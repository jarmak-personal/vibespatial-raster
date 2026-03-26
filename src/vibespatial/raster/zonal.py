"""Zonal statistics via CCCL segmented reduce.

Computes per-zone aggregate statistics (count, sum, mean, min, max, std, median)
over raster values using CCCL segmented reduce primitives.

GPU path: sort by zone_id (CCCL radix_sort), find segment boundaries, then
CCCL segmented_reduce for each requested statistic.  All computation stays
on-device until the final DataFrame transfer.

CPU path: numpy sort + segment loop (baseline).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from vibespatial.raster.buffers import (
    GridSpec,
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    ZonalSpec,
    ZonalStatistic,
)
from vibespatial.residency import Residency, TransferTrigger

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    """Return True if CuPy can be imported."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(zones: OwnedRasterArray, values: OwnedRasterArray) -> bool:
    """Auto-dispatch heuristic: use GPU when available and raster is large enough."""
    try:
        from vibespatial.runtime import has_gpu_runtime

        if not has_gpu_runtime():
            return False
        return zones.pixel_count >= 100_000
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NVRTC kernel sources for median extraction
# ---------------------------------------------------------------------------

_MEDIAN_KERNEL_SOURCE = r"""
extern "C" __global__
void extract_median(
    const double* __restrict__ sorted_values,
    const int* __restrict__ starts,
    const int* __restrict__ ends,
    double* __restrict__ out,
    const int num_segments
) {
    const int stride = blockDim.x * gridDim.x;
    for (int seg = blockIdx.x * blockDim.x + threadIdx.x;
         seg < num_segments;
         seg += stride) {
        const int s = starts[seg];
        const int e = ends[seg];
        const int count = e - s;

        if (count == 0) {
            out[seg] = 0.0 / 0.0;  // NaN
        } else if (count & 1) {
            out[seg] = sorted_values[s + (count >> 1)];
        } else {
            const int mid = count >> 1;
            out[seg] = (sorted_values[s + mid - 1] + sorted_values[s + mid]) * 0.5;
        }
    }
}
"""


# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------


def _zonal_stats_cpu(
    zones: OwnedRasterArray,
    values: OwnedRasterArray,
    requested: tuple[ZonalStatistic, ...],
) -> pd.DataFrame:
    """CPU path: numpy sort + per-segment loop."""
    zone_data = zones.to_numpy().ravel().astype(np.int64)
    value_data = values.to_numpy().ravel().astype(np.float64)

    # Build nodata mask from values raster
    if values.nodata is not None:
        if np.isnan(values.nodata):
            valid = ~np.isnan(value_data)
        else:
            valid = value_data != values.nodata
    else:
        valid = np.ones(len(value_data), dtype=bool)

    # Also exclude nodata zones
    if zones.nodata is not None:
        if np.isnan(zones.nodata):
            valid &= ~np.isnan(zone_data.astype(float))
        else:
            valid &= zone_data != int(zones.nodata)

    # Filter to valid pixels
    zone_valid = zone_data[valid]
    value_valid = value_data[valid]

    # Get unique zone labels
    unique_zones = np.unique(zone_valid)
    if len(unique_zones) == 0:
        return pd.DataFrame({"zone": pd.array([], dtype="int64")})

    results: dict[str, object] = {"zone": unique_zones}

    # Sort by zone for grouped operations
    sort_idx = np.argsort(zone_valid)
    zones_sorted = zone_valid[sort_idx]
    values_sorted = value_valid[sort_idx]

    # Find segment boundaries
    boundaries = np.searchsorted(zones_sorted, unique_zones)
    boundaries = np.append(boundaries, len(zones_sorted))

    for stat in requested:
        col = np.empty(len(unique_zones), dtype=np.float64)
        for i in range(len(unique_zones)):
            segment = values_sorted[boundaries[i] : boundaries[i + 1]]
            if len(segment) == 0:
                col[i] = np.nan
                continue

            if stat is ZonalStatistic.COUNT:
                col[i] = len(segment)
            elif stat is ZonalStatistic.SUM:
                col[i] = np.sum(segment)
            elif stat is ZonalStatistic.MEAN:
                col[i] = np.mean(segment)
            elif stat is ZonalStatistic.MIN:
                col[i] = np.min(segment)
            elif stat is ZonalStatistic.MAX:
                col[i] = np.max(segment)
            elif stat is ZonalStatistic.STD:
                col[i] = np.std(segment, ddof=0)
            elif stat is ZonalStatistic.MEDIAN:
                col[i] = np.median(segment)

        results[str(stat.value)] = col

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# GPU path via CCCL primitives
# ---------------------------------------------------------------------------


def _zonal_stats_gpu(
    zones: OwnedRasterArray,
    values: OwnedRasterArray,
    requested: tuple[ZonalStatistic, ...],
) -> pd.DataFrame:
    """GPU path: CCCL radix sort + segmented reduce, all on-device.

    Pipeline:
    1. Move both rasters to device, get CuPy arrays
    2. Build valid mask on device (both zones and values nodata excluded)
    3. Compact valid pixels to dense (zone_id, value) pairs
    4. Sort by zone_id via CCCL radix_sort
    5. Find segment boundaries via unique_by_key + searchsorted
    6. Segmented reduce for each statistic (CCCL primitives)
    7. Transfer results to host, build DataFrame
    """
    import cupy as cp

    from vibespatial.cccl_primitives import (
        segmented_reduce_max,
        segmented_reduce_min,
        segmented_reduce_sum,
        segmented_sort,
        sort_pairs,
        unique_sorted_pairs,
    )
    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )

    # ---- Step (a): Move both rasters to device, get CuPy arrays ----
    zones.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="zonal_stats_gpu requires device-resident zone data",
    )
    values.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="zonal_stats_gpu requires device-resident value data",
    )
    d_zones = zones.device_data()
    d_values = values.device_data()

    # Flatten to 1D
    d_zones_flat = d_zones.ravel().astype(cp.int64)
    d_values_flat = d_values.ravel().astype(cp.float64)

    # Use .shape[0] instead of int(.size) to avoid implicit D2H sync
    total_pixels = d_zones_flat.shape[0]

    # ---- Step (b): Build valid mask on device ----
    d_valid = cp.ones(total_pixels, dtype=cp.bool_)

    # Exclude values nodata
    if values.nodata is not None:
        if np.isnan(values.nodata):
            d_valid &= ~cp.isnan(d_values_flat)
        else:
            d_valid &= d_values_flat != values.nodata

    # Exclude zones nodata
    if zones.nodata is not None:
        if np.isnan(zones.nodata):
            d_valid &= ~cp.isnan(d_zones_flat.astype(cp.float64))
        else:
            d_valid &= d_zones_flat != int(zones.nodata)

    # ---- Step (c): Compact valid pixels ----
    d_valid_indices = cp.flatnonzero(d_valid)
    # Use .shape[0] instead of int(.size) to avoid implicit D2H sync
    n_valid = d_valid_indices.shape[0]

    if n_valid == 0:
        return pd.DataFrame({"zone": pd.array([], dtype="int64")})

    d_zone_valid = d_zones_flat[d_valid_indices]
    d_value_valid = d_values_flat[d_valid_indices]

    # ---- Step (d): Sort by zone_id via CCCL radix_sort ----
    sort_result = sort_pairs(d_zone_valid, d_value_valid)
    d_sorted_zones = sort_result.keys
    d_sorted_values = sort_result.values

    # ---- Step (e): Find unique zones and segment boundaries ----
    # Use counting indices as items for unique_by_key
    d_counting = cp.arange(n_valid, dtype=cp.int64)
    unique_result = unique_sorted_pairs(d_sorted_zones, d_counting)
    n_zones = unique_result.count

    if n_zones == 0:
        return pd.DataFrame({"zone": pd.array([], dtype="int64")})

    d_unique_zones = unique_result.keys[:n_zones]

    # Compute segment starts and ends using searchsorted on sorted zones
    d_starts = cp.searchsorted(d_sorted_zones, d_unique_zones, side="left").astype(cp.int32)
    d_ends = cp.searchsorted(d_sorted_zones, d_unique_zones, side="right").astype(cp.int32)

    # ---- Step (f): Segmented reduce for each statistic ----
    results_device: dict[str, object] = {}

    # Precompute stats we'll need for derived statistics
    need_count = ZonalStatistic.COUNT in requested
    need_sum = ZonalStatistic.SUM in requested
    need_mean = ZonalStatistic.MEAN in requested
    need_std = ZonalStatistic.STD in requested

    # COUNT is always needed for MEAN and STD
    d_counts = None
    if need_count or need_mean or need_std:
        d_counts = (d_ends - d_starts).astype(cp.float64)

    # SUM is needed for MEAN and STD
    d_sums = None
    if need_sum or need_mean or need_std:
        sum_result = segmented_reduce_sum(
            d_sorted_values,
            d_starts,
            d_ends,
            num_segments=n_zones,
            synchronize=False,
        )
        d_sums = sum_result.values

    # MEAN is needed for STD
    d_means = None
    if need_mean or need_std:
        d_means = d_sums / d_counts

    for stat in requested:
        if stat is ZonalStatistic.COUNT:
            results_device["count"] = d_counts

        elif stat is ZonalStatistic.SUM:
            results_device["sum"] = d_sums

        elif stat is ZonalStatistic.MEAN:
            results_device["mean"] = d_means

        elif stat is ZonalStatistic.MIN:
            min_result = segmented_reduce_min(
                d_sorted_values,
                d_starts,
                d_ends,
                num_segments=n_zones,
                synchronize=False,
            )
            results_device["min"] = min_result.values

        elif stat is ZonalStatistic.MAX:
            max_result = segmented_reduce_max(
                d_sorted_values,
                d_starts,
                d_ends,
                num_segments=n_zones,
                synchronize=False,
            )
            results_device["max"] = max_result.values

        elif stat is ZonalStatistic.STD:
            # STD = sqrt(sum((x - mean)^2) / count)
            #
            # Welford-style two-pass:
            #   pass 1: CCCL segmented_reduce_sum for sum -> mean (done above)
            #   pass 2: CCCL segmented_reduce_sum for sum((x - mean)^2)
            #
            # We compute (x - mean)^2 as a CuPy vectorised op over the
            # sorted values array, using per-segment broadcast of means
            # via fancy-index from d_starts/d_ends offsets.  This avoids
            # the O(n)-per-segment serial loop in the old NVRTC kernel.
            #
            # Build per-pixel means via segment broadcast (all on-device):
            # 1. Create a segment-id array via searchsorted of pixel
            #    indices into d_starts.
            # 2. Gather means[seg_id] to get per-pixel mean.
            # 3. Compute (value - mean)^2 element-wise.
            # 4. CCCL segmented_reduce_sum over the squared diffs.

            # Map each pixel position to its segment id
            d_pixel_idx = cp.arange(n_valid, dtype=cp.int32)
            # searchsorted(starts, pixel_idx, side='right') - 1 gives
            # the segment each pixel belongs to
            d_seg_ids = cp.searchsorted(d_starts, d_pixel_idx, side="right").astype(cp.int32) - 1
            # Broadcast means to per-pixel
            d_pixel_means = d_means[d_seg_ids]
            # Squared differences
            d_sq_diffs = (d_sorted_values - d_pixel_means) ** 2

            # Sum of squared diffs per segment via CCCL
            sq_result = segmented_reduce_sum(
                d_sq_diffs,
                d_starts,
                d_ends,
                num_segments=n_zones,
                synchronize=False,
            )

            d_std = cp.sqrt(sq_result.values / d_counts)
            results_device["std"] = d_std

        elif stat is ZonalStatistic.MEDIAN:
            # sort_pairs sorted by zone_id only; values within each
            # segment are NOT yet sorted.  Use CCCL segmented_sort to
            # sort values within each zone segment before extracting
            # the median.
            seg_sort_result = segmented_sort(
                d_sorted_values,
                starts=d_starts,
                ends=d_ends,
                num_segments=n_zones,
            )
            d_seg_sorted_values = seg_sort_result.keys

            # Extract median via NVRTC kernel with grid-stride loop
            # and occupancy-based launch config.
            runtime = get_cuda_runtime()
            cache_key = make_kernel_cache_key("extract_median", _MEDIAN_KERNEL_SOURCE)
            kernels = runtime.compile_kernels(
                cache_key=cache_key,
                source=_MEDIAN_KERNEL_SOURCE,
                kernel_names=("extract_median",),
            )

            d_median = cp.empty(n_zones, dtype=cp.float64)

            # Occupancy-based launch config
            grid, block = runtime.launch_config(kernels["extract_median"], n_zones)

            params = (
                (
                    d_seg_sorted_values.data.ptr,
                    d_starts.data.ptr,
                    d_ends.data.ptr,
                    d_median.data.ptr,
                    n_zones,
                ),
                (
                    KERNEL_PARAM_PTR,  # sorted_values
                    KERNEL_PARAM_PTR,  # starts
                    KERNEL_PARAM_PTR,  # ends
                    KERNEL_PARAM_PTR,  # out
                    KERNEL_PARAM_I32,  # num_segments
                ),
            )
            runtime.launch(
                kernel=kernels["extract_median"],
                grid=grid,
                block=block,
                params=params,
            )
            results_device["median"] = d_median

    # ---- Step (g): Transfer results to host, build DataFrame ----
    # Single sync before all D2H transfers
    cp.cuda.Stream.null.synchronize()

    host_results: dict[str, object] = {
        "zone": cp.asnumpy(d_unique_zones),
    }
    for stat in requested:
        key = str(stat.value)
        host_results[key] = cp.asnumpy(results_device[key])

    return pd.DataFrame(host_results)


# ---------------------------------------------------------------------------
# Public API with dispatch
# ---------------------------------------------------------------------------


def zonal_stats(
    zones: OwnedRasterArray,
    values: OwnedRasterArray,
    stats: ZonalSpec | list[str] | None = None,
    *,
    use_gpu: bool | None = None,
) -> pd.DataFrame:
    """Compute zonal statistics of values raster over zone labels.

    Parameters
    ----------
    zones : OwnedRasterArray
        Integer label raster defining zones. Each unique nonzero value is a zone.
    values : OwnedRasterArray
        Value raster to aggregate. Must have same height/width as zones.
    stats : ZonalSpec or list[str] or None
        Statistics to compute. Default: count, sum, mean, min, max.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when CuPy is available and pixel count >= 100k.

    Returns
    -------
    pd.DataFrame
        One row per zone, columns for zone label and each requested statistic.
    """
    if zones.band_count != 1:
        raise ValueError("zones raster must be single-band")
    if values.band_count != 1:
        raise ValueError(
            "values raster must be single-band. For multiband, select a band first via indexing."
        )

    if zones.height != values.height or zones.width != values.width:
        raise ValueError(
            f"zones shape ({zones.height}x{zones.width}) must match "
            f"values shape ({values.height}x{values.width})"
        )

    if stats is None:
        spec = ZonalSpec()
    elif isinstance(stats, ZonalSpec):
        spec = stats
    else:
        spec = ZonalSpec(stats=tuple(stats))

    requested = spec.normalized_stats()

    # Dispatch
    if use_gpu is None:
        use_gpu = _should_use_gpu(zones, values)

    t0 = time.perf_counter()
    if use_gpu:
        result = _zonal_stats_gpu(zones, values, requested)
    else:
        result = _zonal_stats_cpu(zones, values, requested)
    elapsed = time.perf_counter() - t0

    n_zones = len(result)
    n_pixels = zones.pixel_count
    backend = "gpu" if use_gpu else "cpu"
    zones.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"zonal_stats {backend} zones={n_zones} pixels={n_pixels} "
                f"stats={[s.value for s in requested]} elapsed={elapsed:.3f}s"
            ),
            residency=zones.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )

    return result


def zonal_stats_gdf(
    gdf,
    values: OwnedRasterArray,
    stats: ZonalSpec | list[str] | None = None,
    *,
    grid_spec: GridSpec | None = None,
    use_gpu: bool | None = None,
) -> pd.DataFrame:
    """Compute zonal statistics of a raster over vector zone boundaries.

    Rasterizes the GeoDataFrame zones to a label grid matching the values
    raster, then runs zonal_stats on the label grid.

    Parameters
    ----------
    gdf : GeoDataFrame
        Vector zones. Each row is a zone.
    values : OwnedRasterArray
        Value raster to aggregate.
    stats : ZonalSpec or list[str] or None
        Statistics to compute.
    grid_spec : GridSpec or None
        Grid to rasterize zones onto. If None, matches the values raster grid.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    pd.DataFrame
        One row per zone with zone index and requested statistics.
    """
    from vibespatial.raster.rasterize import rasterize_owned

    if grid_spec is None:
        grid_spec = GridSpec(
            affine=values.affine,
            width=values.width,
            height=values.height,
            dtype=np.dtype("int32"),
            fill_value=0,
        )

    # Create integer zone labels (1-indexed by GeoDataFrame row position)
    geometries = list(gdf.geometry)
    zone_labels = np.arange(1, len(geometries) + 1, dtype=np.float64)
    zone_raster = rasterize_owned(geometries, zone_labels, grid_spec)

    # Run zonal stats on the rasterized zones
    result = zonal_stats(zone_raster, values, stats=stats, use_gpu=use_gpu)

    # Map zone labels back to GeoDataFrame index
    if len(result) > 0:
        zone_to_idx = {i + 1: idx for i, idx in enumerate(gdf.index)}
        result["gdf_index"] = result["zone"].map(zone_to_idx)

    return result
