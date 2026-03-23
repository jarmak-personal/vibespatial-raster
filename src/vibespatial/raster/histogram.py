"""GPU-accelerated histogram, CDF, equalization, and percentile operations.

GPU path: CCCL histogram_even for bin counting, CCCL exclusive_scan for CDF,
and a custom NVRTC remap kernel for histogram equalization.  All computation
stays on-device until the final result transfer.

CPU path: numpy.histogram, numpy.cumsum, numpy-based equalization.
"""

from __future__ import annotations

import time

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
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


def _should_use_gpu(raster: OwnedRasterArray) -> bool:
    """Auto-dispatch heuristic: use GPU when available and raster is large enough."""
    try:
        from vibespatial.runtime import has_gpu_runtime

        if not has_gpu_runtime():
            return False
        return raster.pixel_count >= 100_000
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CPU baseline
# ---------------------------------------------------------------------------


def _raster_histogram_cpu(
    raster: OwnedRasterArray,
    bins: int = 256,
    range_min: float | None = None,
    range_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU path: numpy.histogram on valid (non-nodata) pixels."""
    data = raster.to_numpy().ravel()

    # Exclude nodata pixels
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            valid = data[~np.isnan(data)]
        else:
            valid = data[data != raster.nodata]
    else:
        valid = data

    if valid.size == 0:
        edges = np.linspace(
            range_min if range_min is not None else 0.0,
            range_max if range_max is not None else 1.0,
            bins + 1,
        )
        return np.zeros(bins, dtype=np.int64), edges

    lo = float(range_min if range_min is not None else np.min(valid))
    hi = float(range_max if range_max is not None else np.max(valid))
    # Ensure hi > lo for linspace
    if hi <= lo:
        hi = lo + 1.0

    counts, edges = np.histogram(valid, bins=bins, range=(lo, hi))
    return counts.astype(np.int64), edges


def _raster_cdf_cpu(
    counts: np.ndarray,
) -> np.ndarray:
    """CPU path: cumulative sum of histogram counts."""
    return np.cumsum(counts).astype(np.int64)


def _raster_histogram_equalize_cpu(
    raster: OwnedRasterArray,
) -> OwnedRasterArray:
    """CPU path: histogram equalization via numpy."""
    data = raster.to_numpy()
    original_shape = data.shape

    # Work with uint8 for equalization
    if data.dtype != np.uint8:
        dmin = (
            float(np.nanmin(data))
            if raster.nodata is None
            else float(np.min(data[data != raster.nodata]) if np.any(data != raster.nodata) else 0)
        )
        dmax = (
            float(np.nanmax(data))
            if raster.nodata is None
            else float(
                np.max(data[data != raster.nodata]) if np.any(data != raster.nodata) else 255
            )
        )
        if dmax <= dmin:
            dmax = dmin + 1.0
        normalized = np.clip((data.astype(np.float64) - dmin) / (dmax - dmin) * 255.0, 0, 255)
        data_u8 = normalized.astype(np.uint8)
    else:
        data_u8 = data.copy()

    flat = data_u8.ravel()

    # Compute histogram of valid pixels only
    if raster.nodata is not None:
        nodata_u8 = int(raster.nodata) if data.dtype == np.uint8 else 0
        nodata_mask = raster.nodata_mask.ravel()
        valid = flat[~nodata_mask]
    else:
        nodata_u8 = 0
        nodata_mask = None
        valid = flat

    if valid.size == 0:
        return from_numpy(
            np.zeros(original_shape, dtype=np.uint8),
            nodata=raster.nodata,
            affine=raster.affine,
            crs=raster.crs,
        )

    counts, _ = np.histogram(valid, bins=256, range=(0, 256))
    cdf = np.cumsum(counts)
    cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
    total_valid = int(valid.size)

    # Build LUT
    lut = np.zeros(256, dtype=np.uint8)
    denom = total_valid - cdf_min
    if denom > 0:
        for i in range(256):
            lut[i] = np.clip(round((cdf[i] - cdf_min) / denom * 255.0), 0, 255)

    # Apply LUT
    result = lut[flat].reshape(original_shape)

    # Restore nodata
    if nodata_mask is not None:
        result_flat = result.ravel()
        result_flat[nodata_mask] = nodata_u8
        result = result_flat.reshape(original_shape)

    out_nodata = (
        int(raster.nodata) if raster.nodata is not None and data.dtype == np.uint8 else None
    )
    return from_numpy(result, nodata=out_nodata, affine=raster.affine, crs=raster.crs)


def _raster_percentile_cpu(
    raster: OwnedRasterArray,
    percentiles: list[float],
    bins: int = 256,
) -> np.ndarray:
    """CPU path: percentile via histogram CDF."""
    data = raster.to_numpy().ravel()

    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            valid = data[~np.isnan(data)]
        else:
            valid = data[data != raster.nodata]
    else:
        valid = data

    if valid.size == 0:
        return np.full(len(percentiles), np.nan, dtype=np.float64)

    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if hi <= lo:
        return np.full(len(percentiles), lo, dtype=np.float64)

    counts, edges = np.histogram(valid, bins=bins, range=(lo, hi))
    cdf = np.cumsum(counts).astype(np.float64)
    total = cdf[-1]

    results = np.empty(len(percentiles), dtype=np.float64)
    for i, pct in enumerate(percentiles):
        target = pct / 100.0 * total
        idx = np.searchsorted(cdf, target, side="left")
        idx = min(idx, len(edges) - 2)
        results[i] = edges[idx]

    return results


# ---------------------------------------------------------------------------
# GPU path
# ---------------------------------------------------------------------------


def _raster_histogram_gpu(
    raster: OwnedRasterArray,
    bins: int = 256,
    range_min: float | None = None,
    range_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU path: CCCL histogram_even, all on-device until final transfer."""
    import cupy as cp
    from cuda.compute import algorithms as cccl_algorithms

    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="raster_histogram requires device-resident data",
    )
    d_data = raster.device_data()
    d_flat = d_data.ravel()

    # Exclude nodata: compact valid pixels on device
    if raster.nodata is not None:
        d_nodata_mask = raster.device_nodata_mask().ravel()
        d_valid_idx = cp.flatnonzero(~d_nodata_mask)
        d_valid = d_flat[d_valid_idx]
    else:
        d_valid = d_flat

    n_valid = int(d_valid.size)
    if n_valid == 0:
        lo = float(range_min if range_min is not None else 0.0)
        hi = float(range_max if range_max is not None else 1.0)
        edges = np.linspace(lo, hi, bins + 1)
        return np.zeros(bins, dtype=np.int64), edges

    # Determine range on device (single min/max kernel, no host round-trip for data)
    if range_min is not None and range_max is not None:
        lo = float(range_min)
        hi = float(range_max)
    else:
        # These are scalar reductions -- small D2H is unavoidable for range
        lo = float(range_min if range_min is not None else float(cp.min(d_valid)))
        hi = float(range_max if range_max is not None else float(cp.max(d_valid)))

    if hi <= lo:
        hi = lo + 1.0

    # CCCL histogram_even uses half-open intervals [lower, upper), so the
    # maximum value would be excluded.  Nudge upper_level past the actual
    # max so the last bin captures it.
    hi_for_cccl = float(np.nextafter(np.float64(hi), np.float64(np.inf)))

    # Cast valid data to float64 for histogram_even (requires numeric type)
    d_samples = d_valid.astype(cp.float64, copy=False)

    # Allocate histogram output on device
    # histogram_even: num_output_levels = bins + 1 (bins + 1 level edges define bins bins)
    # NOTE: CCCL histogram_even requires int32 counters (atomicAdd_block
    # does not support int64).  We cast to int64 after the kernel.
    d_histogram = cp.zeros(bins, dtype=cp.int32)

    cccl_algorithms.histogram_even(
        d_samples=d_samples,
        d_histogram=d_histogram,
        num_output_levels=bins + 1,
        lower_level=np.float64(lo),
        upper_level=np.float64(hi_for_cccl),
        num_samples=n_valid,
    )

    # Compute edges on host (cheap: bins+1 floats)
    # Report the original [lo, hi] range to the caller, not the nudged one
    edges = np.linspace(lo, hi, bins + 1)

    # Single D2H transfer for counts
    cp.cuda.Stream.null.synchronize()
    counts = cp.asnumpy(d_histogram).astype(np.int64)

    return counts, edges


def _raster_cdf_gpu(
    counts_device,
) -> object:
    """GPU path: exclusive_sum on histogram counts for CDF.

    Parameters
    ----------
    counts_device : CuPy array
        Histogram bin counts on device.

    Returns
    -------
    CuPy array of cumulative counts (CDF).
    """
    import cupy as cp

    from vibespatial.cccl_primitives import exclusive_sum

    d_counts = counts_device.astype(cp.int64, copy=False)
    d_prefix = exclusive_sum(d_counts, synchronize=False)
    # CDF = exclusive_sum + counts (inclusive sum)
    d_cdf = d_prefix + d_counts
    return d_cdf


def _raster_histogram_equalize_gpu(
    raster: OwnedRasterArray,
) -> OwnedRasterArray:
    """GPU path: histogram -> CDF -> NVRTC remap kernel.

    Full pipeline stays on-device:
    1. Move data to device, convert to uint8
    2. CCCL histogram_even for 256-bin histogram
    3. CCCL exclusive_sum for CDF
    4. Build LUT from CDF (256-entry -- computed on device)
    5. NVRTC remap kernel applies LUT per-pixel
    6. Single D2H transfer for final result
    """
    import cupy as cp
    from cuda.compute import algorithms as cccl_algorithms

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.histogram import HISTOGRAM_REMAP_KERNEL_SOURCE

    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="raster_histogram_equalize requires device-resident data",
    )
    d_data = raster.device_data()
    original_shape = d_data.shape

    # Convert to uint8 for equalization
    if d_data.dtype != cp.uint8:
        # Build nodata-aware min/max on device
        if raster.nodata is not None:
            d_mask = raster.device_nodata_mask()
            d_valid_mask = ~d_mask
            d_flat = d_data.ravel()
            d_valid_idx = cp.flatnonzero(d_valid_mask.ravel())
            if d_valid_idx.size == 0:
                host_result = cp.asnumpy(cp.zeros(original_shape, dtype=cp.uint8))
                return from_numpy(
                    host_result, nodata=raster.nodata, affine=raster.affine, crs=raster.crs
                )
            d_valid = d_flat[d_valid_idx]
            d_min = cp.min(d_valid)
            d_max = cp.max(d_valid)
        else:
            d_min = cp.min(d_data)
            d_max = cp.max(d_data)

        # Normalize to [0, 255] on device
        d_range = d_max - d_min
        d_range = cp.where(d_range > 0, d_range, cp.ones_like(d_range))
        d_normalized = cp.clip((d_data.astype(cp.float64) - d_min) / d_range * 255.0, 0, 255)
        d_u8 = d_normalized.astype(cp.uint8)
    else:
        d_u8 = d_data

    d_flat_u8 = d_u8.ravel()
    n_pixels = int(d_flat_u8.size)

    # Build nodata mask for the remap kernel
    if raster.nodata is not None:
        d_nodata_mask_u8 = raster.device_nodata_mask().ravel().astype(cp.uint8)
        nodata_val_u8 = np.uint8(0)  # nodata pixels get 0 in equalized output
    else:
        d_nodata_mask_u8 = None
        nodata_val_u8 = np.uint8(0)

    # Step 1: Histogram of valid pixels via CCCL histogram_even
    # Get valid pixels for histogram computation
    if d_nodata_mask_u8 is not None:
        d_valid_idx = cp.flatnonzero(d_nodata_mask_u8 == 0)
        d_valid_u8 = d_flat_u8[d_valid_idx]
    else:
        d_valid_u8 = d_flat_u8

    n_valid = int(d_valid_u8.size)
    if n_valid == 0:
        host_result = cp.asnumpy(cp.zeros(original_shape, dtype=cp.uint8))
        return from_numpy(host_result, nodata=raster.nodata, affine=raster.affine, crs=raster.crs)

    d_samples_i32 = d_valid_u8.astype(cp.int32)
    # NOTE: CCCL histogram_even requires int32 counters (atomicAdd_block
    # does not support int64).
    d_hist = cp.zeros(256, dtype=cp.int32)

    cccl_algorithms.histogram_even(
        d_samples=d_samples_i32,
        d_histogram=d_hist,
        num_output_levels=257,
        lower_level=np.int32(0),
        upper_level=np.int32(256),
        num_samples=n_valid,
    )

    # Step 2: CDF via CCCL exclusive_sum (inclusive sum = exclusive_sum + counts)
    # Cast to int64 for CDF to avoid overflow on large rasters
    d_cdf = _raster_cdf_gpu(d_hist.astype(cp.int64))

    # Step 3: Build LUT on device from CDF
    # equalized[i] = round((cdf[i] - cdf_min) / (n_valid - cdf_min) * 255)
    # cdf_min = first nonzero CDF value
    d_nonzero_mask = d_cdf > 0
    d_nonzero_indices = cp.flatnonzero(d_nonzero_mask)
    if d_nonzero_indices.size > 0:
        d_cdf_min = d_cdf[d_nonzero_indices[0]]
    else:
        d_cdf_min = cp.int64(0)

    d_n_valid = cp.int64(n_valid)
    d_denom = d_n_valid - d_cdf_min
    d_denom = cp.where(d_denom > 0, d_denom, cp.int64(1))

    d_lut = cp.clip(
        cp.around(
            (d_cdf.astype(cp.float64) - d_cdf_min.astype(cp.float64))
            / d_denom.astype(cp.float64)
            * 255.0
        ),
        0,
        255,
    ).astype(cp.uint8)

    # Step 4: Apply LUT via NVRTC remap kernel
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("histogram_remap", HISTOGRAM_REMAP_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=HISTOGRAM_REMAP_KERNEL_SOURCE,
        kernel_names=("histogram_remap",),
    )

    d_output = cp.empty(n_pixels, dtype=cp.uint8)

    nodata_ptr = d_nodata_mask_u8.data.ptr if d_nodata_mask_u8 is not None else 0

    params = (
        (
            d_flat_u8.data.ptr,
            d_output.data.ptr,
            d_lut.data.ptr,
            nodata_ptr,
            n_pixels,
            int(nodata_val_u8),
        ),
        (
            KERNEL_PARAM_PTR,  # input
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # lut
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # n
            KERNEL_PARAM_I32,  # nodata_val (passed as int, kernel reads as unsigned char)
        ),
    )

    grid, block = runtime.launch_config(kernels["histogram_remap"], n_pixels)
    runtime.launch(
        kernel=kernels["histogram_remap"],
        grid=grid,
        block=block,
        params=params,
    )

    d_result = d_output.reshape(original_shape)

    # Single D2H transfer at the end
    host_result = cp.asnumpy(d_result)
    out_nodata = (
        int(raster.nodata) if raster.nodata is not None and raster.dtype == np.uint8 else None
    )
    return from_numpy(host_result, nodata=out_nodata, affine=raster.affine, crs=raster.crs)


def _raster_percentile_gpu(
    raster: OwnedRasterArray,
    percentiles: list[float],
    bins: int = 256,
) -> np.ndarray:
    """GPU path: percentile via histogram CDF.

    Uses CCCL histogram_even + exclusive_sum to build the CDF on device,
    then a single D2H of the CDF array to compute percentile values.
    """
    import cupy as cp
    from cuda.compute import algorithms as cccl_algorithms

    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="raster_percentile requires device-resident data",
    )
    d_data = raster.device_data()
    d_flat = d_data.ravel()

    # Exclude nodata
    if raster.nodata is not None:
        d_nodata_mask = raster.device_nodata_mask().ravel()
        d_valid_idx = cp.flatnonzero(~d_nodata_mask)
        d_valid = d_flat[d_valid_idx]
    else:
        d_valid = d_flat

    n_valid = int(d_valid.size)
    if n_valid == 0:
        return np.full(len(percentiles), np.nan, dtype=np.float64)

    # Get data range on device
    d_lo = cp.min(d_valid)
    d_hi = cp.max(d_valid)
    lo = float(d_lo)
    hi = float(d_hi)

    if hi <= lo:
        return np.full(len(percentiles), lo, dtype=np.float64)

    # Histogram via CCCL
    # CCCL histogram_even uses half-open [lower, upper) so nudge upper past max
    hi_for_cccl = float(np.nextafter(np.float64(hi), np.float64(np.inf)))
    d_samples = d_valid.astype(cp.float64, copy=False)
    # NOTE: CCCL histogram_even requires int32 counters (atomicAdd_block
    # does not support int64).
    d_hist = cp.zeros(bins, dtype=cp.int32)

    cccl_algorithms.histogram_even(
        d_samples=d_samples,
        d_histogram=d_hist,
        num_output_levels=bins + 1,
        lower_level=np.float64(lo),
        upper_level=np.float64(hi_for_cccl),
        num_samples=n_valid,
    )

    # CDF via CCCL exclusive_sum (cast to int64 for large rasters)
    d_cdf = _raster_cdf_gpu(d_hist.astype(cp.int64))

    # Transfer CDF to host (small: bins elements)
    cp.cuda.Stream.null.synchronize()
    cdf = cp.asnumpy(d_cdf).astype(np.float64)
    total = cdf[-1]

    edges = np.linspace(lo, hi, bins + 1)

    results = np.empty(len(percentiles), dtype=np.float64)
    for i, pct in enumerate(percentiles):
        target = pct / 100.0 * total
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(idx, len(edges) - 2)
        results[i] = edges[idx]

    return results


# ---------------------------------------------------------------------------
# Public API with dispatch
# ---------------------------------------------------------------------------


def raster_histogram(
    raster: OwnedRasterArray,
    bins: int = 256,
    *,
    range_min: float | None = None,
    range_max: float | None = None,
    use_gpu: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of raster pixel values.

    Nodata pixels are excluded from the histogram.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    bins : int
        Number of histogram bins (default: 256).
    range_min, range_max : float or None
        Value range for the histogram.  If None, inferred from valid pixels.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (counts, bin_edges) where counts has shape (bins,) and
        bin_edges has shape (bins + 1,).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    t0 = time.perf_counter()
    if use_gpu:
        counts, edges = _raster_histogram_gpu(raster, bins, range_min, range_max)
    else:
        counts, edges = _raster_histogram_cpu(raster, bins, range_min, range_max)
    elapsed = time.perf_counter() - t0

    backend = "gpu" if use_gpu else "cpu"
    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"raster_histogram {backend} pixels={raster.pixel_count} "
                f"bins={bins} elapsed={elapsed:.3f}s"
            ),
            residency=raster.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return counts, edges


def raster_cumulative_distribution(
    raster: OwnedRasterArray,
    bins: int = 256,
    *,
    range_min: float | None = None,
    range_max: float | None = None,
    use_gpu: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cumulative distribution function of raster pixel values.

    Computes the histogram, then takes the cumulative sum to get the CDF.
    Nodata pixels are excluded.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    bins : int
        Number of histogram bins (default: 256).
    range_min, range_max : float or None
        Value range.  If None, inferred from valid pixels.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (cdf, bin_edges) where cdf has shape (bins,) and is monotonically
        non-decreasing, and bin_edges has shape (bins + 1,).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    t0 = time.perf_counter()
    if use_gpu:
        counts, edges = _raster_histogram_gpu(raster, bins, range_min, range_max)
        cdf = np.cumsum(counts)
    else:
        counts, edges = _raster_histogram_cpu(raster, bins, range_min, range_max)
        cdf = _raster_cdf_cpu(counts)
    elapsed = time.perf_counter() - t0

    backend = "gpu" if use_gpu else "cpu"
    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"raster_cumulative_distribution {backend} pixels={raster.pixel_count} "
                f"bins={bins} elapsed={elapsed:.3f}s"
            ),
            residency=raster.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return cdf, edges


def raster_histogram_equalize(
    raster: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Apply histogram equalization to a raster.

    Redistributes pixel values to achieve a roughly uniform histogram
    distribution across the 0-255 range.  Output dtype is uint8.
    Nodata pixels are preserved.

    GPU pipeline: histogram (CCCL) -> CDF (CCCL exclusive_sum) ->
    LUT build (CuPy element-wise) -> remap kernel (NVRTC).
    All computation stays on-device until the final D2H transfer.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    OwnedRasterArray
        Equalized raster with dtype uint8 and values in [0, 255].
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    t0 = time.perf_counter()
    if use_gpu:
        result = _raster_histogram_equalize_gpu(raster)
    else:
        result = _raster_histogram_equalize_cpu(raster)
    elapsed = time.perf_counter() - t0

    backend = "gpu" if use_gpu else "cpu"
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"raster_histogram_equalize {backend} pixels={raster.pixel_count} "
                f"elapsed={elapsed:.3f}s"
            ),
            residency=Residency.HOST,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


def raster_percentile(
    raster: OwnedRasterArray,
    percentiles: list[float] | float,
    *,
    bins: int = 256,
    use_gpu: bool | None = None,
) -> np.ndarray:
    """Compute percentile values from a raster using histogram-based CDF.

    Avoids a full sort by computing percentiles from the histogram CDF,
    which is O(bins) rather than O(n log n).  Nodata pixels are excluded.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    percentiles : list[float] or float
        Percentile(s) to compute, in range [0, 100].
    bins : int
        Number of histogram bins for CDF computation (default: 256).
        More bins = higher precision.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    np.ndarray
        Array of percentile values with shape (len(percentiles),).
    """
    if isinstance(percentiles, (int, float)):
        percentiles = [float(percentiles)]
    else:
        percentiles = [float(p) for p in percentiles]

    for p in percentiles:
        if p < 0 or p > 100:
            raise ValueError(f"percentile must be in [0, 100], got {p}")

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    t0 = time.perf_counter()
    if use_gpu:
        result = _raster_percentile_gpu(raster, percentiles, bins)
    else:
        result = _raster_percentile_cpu(raster, percentiles, bins)
    elapsed = time.perf_counter() - t0

    backend = "gpu" if use_gpu else "cpu"
    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"raster_percentile {backend} pixels={raster.pixel_count} "
                f"percentiles={percentiles} elapsed={elapsed:.3f}s"
            ),
            residency=raster.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result
