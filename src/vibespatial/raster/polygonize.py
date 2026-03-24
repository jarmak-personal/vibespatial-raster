"""Raster-to-vector polygonize pipeline.

Converts labeled raster regions into polygon geometries. CPU baseline uses
rasterio.features.shapes; GPU path uses marching-squares NVRTC kernels.

Beads o17.8.12-8.15.
"""

from __future__ import annotations

import collections
import time

import numpy as np
from shapely.geometry import Polygon, shape

from vibespatial.fusion import (
    FusionPlan,
    PipelineStep,
    StepKind,
    plan_fusion,
)
from vibespatial.raster.buffers import (
    OwnedRasterArray,
    PolygonizeSpec,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
)
from vibespatial.residency import Residency, TransferTrigger

# ---------------------------------------------------------------------------
# CPU baseline via rasterio.features.shapes
# ---------------------------------------------------------------------------


def polygonize_cpu(
    raster: OwnedRasterArray,
    *,
    spec: PolygonizeSpec | None = None,
) -> tuple[list, list]:
    """Polygonize a raster using rasterio (CPU baseline).

    Returns (geometries, values) lists.
    """
    try:
        from rasterio.features import shapes
        from rasterio.transform import Affine
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for CPU polygonize. "
            "Install with: uv sync --group upstream-optional"
        ) from exc

    if spec is None:
        spec = PolygonizeSpec()

    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    transform = Affine(
        raster.affine[0],
        raster.affine[1],
        raster.affine[2],
        raster.affine[3],
        raster.affine[4],
        raster.affine[5],
    )

    # Build mask: exclude nodata pixels
    mask = None
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            mask = ~np.isnan(data)
        else:
            mask = data != raster.nodata

    connectivity = spec.connectivity

    geometries = []
    values = []
    poly_count = 0

    for geom_dict, value in shapes(
        data.astype(np.float32),
        mask=mask,
        transform=transform,
        connectivity=connectivity,
    ):
        poly_count += 1
        if spec.max_polygons is not None and poly_count > spec.max_polygons:
            import warnings

            warnings.warn(
                f"Polygon explosion: {poly_count} polygons exceed "
                f"max_polygons={spec.max_polygons}. Truncating output.",
                stacklevel=2,
            )
            break

        geom = shape(geom_dict)

        if spec.simplify_tolerance is not None and spec.simplify_tolerance > 0:
            geom = geom.simplify(spec.simplify_tolerance)

        if not geom.is_empty:
            geometries.append(geom)
            values.append(value)

    return geometries, values


# ---------------------------------------------------------------------------
# GPU polygonize via marching-squares NVRTC kernels
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    """Check if CuPy is importable."""
    try:
        import cupy as cp  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(raster: OwnedRasterArray) -> bool:
    """Auto-dispatch heuristic for polygonize."""
    try:
        from vibespatial.runtime import has_gpu_runtime

        return has_gpu_runtime() and raster.pixel_count >= 100_000
    except Exception:
        return False


def _chain_edges_to_rings(
    edge_x0: np.ndarray,
    edge_y0: np.ndarray,
    edge_x1: np.ndarray,
    edge_y1: np.ndarray,
    edge_values: np.ndarray,
) -> dict[float, list[list[tuple[float, float]]]]:
    """Chain directed edge segments into closed polygon rings, grouped by value.

    This is the inherently-serial graph traversal step that runs on the host
    after the GPU kernels emit all edge segments.

    Parameters
    ----------
    edge_x0, edge_y0 : start vertex world coordinates
    edge_x1, edge_y1 : end vertex world coordinates
    edge_values : raster value for each edge

    Returns
    -------
    dict mapping raster value -> list of rings (each ring is a list of (x,y) tuples)
    """
    n_edges = len(edge_x0)
    if n_edges == 0:
        return {}

    # Round coordinates to avoid floating-point mismatches when chaining.
    # Use enough precision to distinguish sub-pixel midpoints.
    PRECISION = 10

    # Build adjacency: start_vertex -> (end_vertex, edge_index)
    # Group edges by value first, then chain within each value group.
    value_edges: dict[float, list[int]] = collections.defaultdict(list)
    for i in range(n_edges):
        value_edges[edge_values[i]].append(i)

    result: dict[float, list[list[tuple[float, float]]]] = {}

    for val, edge_indices in value_edges.items():
        # Build a start-vertex -> list of (end_x, end_y, edge_idx) mapping
        adj: dict[tuple[float, float], list[tuple[float, float, int]]] = collections.defaultdict(
            list
        )
        for idx in edge_indices:
            sx = round(edge_x0[idx], PRECISION)
            sy = round(edge_y0[idx], PRECISION)
            ex = round(edge_x1[idx], PRECISION)
            ey = round(edge_y1[idx], PRECISION)
            adj[(sx, sy)].append((ex, ey, idx))

        used = set()
        rings = []

        for idx in edge_indices:
            if idx in used:
                continue

            # Start a new ring from this edge
            ring: list[tuple[float, float]] = []
            sx = round(edge_x0[idx], PRECISION)
            sy = round(edge_y0[idx], PRECISION)
            ring_start = (sx, sy)
            ring.append((edge_x0[idx], edge_y0[idx]))

            current_end = (
                round(edge_x1[idx], PRECISION),
                round(edge_y1[idx], PRECISION),
            )
            ring.append((edge_x1[idx], edge_y1[idx]))
            used.add(idx)

            # Follow the chain until we return to the start or get stuck
            max_steps = len(edge_indices) + 1
            for _ in range(max_steps):
                if current_end == ring_start:
                    # Ring closed
                    break

                # Find an unused edge starting at current_end
                candidates = adj.get(current_end, [])
                found = False
                for ex, ey, next_idx in candidates:
                    if next_idx not in used:
                        used.add(next_idx)
                        current_end = (ex, ey)
                        ring.append((edge_x1[next_idx], edge_y1[next_idx]))
                        found = True
                        break
                if not found:
                    break

            if current_end == ring_start and len(ring) >= 4:
                rings.append(ring)

        if rings:
            result[val] = rings

    return result


def _rings_to_polygons(
    value_rings: dict[float, list[list[tuple[float, float]]]],
    spec: PolygonizeSpec,
) -> tuple[list, list]:
    """Convert ring groups into Shapely polygon geometries.

    For each value, the largest ring (by area) is treated as the exterior
    and smaller rings inside it become holes.  Multiple exterior rings
    for the same value produce separate polygons.

    Parameters
    ----------
    value_rings : dict mapping value -> list of rings
    spec : PolygonizeSpec with simplify_tolerance etc.

    Returns
    -------
    (geometries, values) tuple
    """
    geometries: list = []
    values: list = []
    poly_count = 0

    for val, rings in value_rings.items():
        # Build polygons from rings.
        # Simple heuristic: each ring becomes its own polygon.
        # A more sophisticated approach would detect containment for holes,
        # but for marching-squares output on labeled rasters the rings
        # are typically simple exterior boundaries.
        polys_for_value: list[Polygon] = []

        for ring_coords in rings:
            if len(ring_coords) < 4:
                continue
            try:
                poly = Polygon(ring_coords)
                if poly.is_valid and not poly.is_empty:
                    polys_for_value.append(poly)
            except Exception:
                continue

        # Detect holes: if a smaller polygon is fully within a larger one,
        # make it a hole. We use a simple area-sorted approach.
        if len(polys_for_value) > 1:
            polys_for_value.sort(key=lambda p: p.area, reverse=True)
            final_polys: list[Polygon] = []
            for i, poly in enumerate(polys_for_value):
                is_hole = False
                for j in range(i):
                    if polys_for_value[j].contains(poly):
                        # This is a hole inside polygon j — skip it
                        # (holes belong to other values in labeled rasters)
                        is_hole = True
                        break
                if not is_hole:
                    final_polys.append(poly)
            polys_for_value = final_polys

        for poly in polys_for_value:
            poly_count += 1
            if spec.max_polygons is not None and poly_count > spec.max_polygons:
                import warnings

                warnings.warn(
                    f"Polygon explosion: {poly_count} polygons exceed "
                    f"max_polygons={spec.max_polygons}. Truncating output.",
                    stacklevel=2,
                )
                return geometries, values

            if spec.simplify_tolerance is not None and spec.simplify_tolerance > 0:
                poly = poly.simplify(spec.simplify_tolerance)

            if not poly.is_empty:
                geometries.append(poly)
                values.append(val)

    return geometries, values


def polygonize_gpu(
    raster: OwnedRasterArray,
    *,
    spec: PolygonizeSpec | None = None,
) -> tuple[list, list]:
    """Polygonize a raster using GPU marching-squares kernels.

    Pipeline (per unique raster value):
      1. Transfer raster to device (once, shared across all values)
      2. classify_cells kernel -- binary-classify each 2x2 cell for target value
      3. Compact non-trivial cells via CuPy boolean indexing
      4. edge_count kernel + prefix sum -- compute write offsets
      5. emit_edges kernel -- emit directed edges with precomputed affine offsets
      6. Accumulate edges across all unique values
      7. Single bulk D2H transfer of concatenated edge arrays
      8. Chain edges into rings, convert to Shapely geometries (host)

    Multi-value correctness: marching-squares classifies cells as in/out
    relative to a single target value.  To produce complete closed rings for
    every distinct raster value, we iterate over each unique non-nodata value
    and run the classify/compact/count/emit pipeline once per value.

    Optimizations (o18.3.22):
      - Occupancy-based launch config (no hardcoded block sizes)
      - Precomputed affine half-step offsets eliminate per-thread recomputation
      - Grid-stride loops with ILP=4 on all kernels
      - __constant__ edge lookup tables in emit_edges and edge_count kernels
      - Reused device buffers (d_cell_class, d_cell_indices) across value iters
      - No intermediate syncs between same-stream kernel launches
      - Single bulk D2H transfer of concatenated results at the end

    Returns (geometries, values) lists, same format as CPU path.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.polygonize import (
        CLASSIFY_CELLS_KERNEL_SOURCE,
        EDGE_COUNT_KERNEL_SOURCE,
        EMIT_EDGES_KERNEL_SOURCE,
    )

    if spec is None:
        spec = PolygonizeSpec()

    t0 = time.perf_counter()

    # --- Prepare raster data on device (zero-copy: no D->H->D) ---
    # Move raster to device using the standard residency pattern.
    # Shape metadata is available without any transfer.
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="polygonize_gpu requires device-resident data",
    )
    d_data = raster.device_data()
    if d_data.ndim == 3:
        d_data = d_data[0]

    orig_height, orig_width = d_data.shape

    if orig_height < 2 or orig_width < 2:
        return [], []

    # --- Pad the raster with a 1-pixel nodata border (all on device) ---
    # This is the standard marching-squares padding technique: by surrounding
    # the raster with a border of "outside" values, boundary cells at the
    # raster edge naturally become mixed cells and produce edges.  Without
    # padding, homogeneous regions touching the raster border generate only
    # class-15 (all-match) cells that emit zero edges, so rings can never
    # close and no polygons are produced.
    d_data_f64 = d_data.astype(cp.float64)

    # Choose a sentinel value for the padding border that is guaranteed to
    # differ from every real data pixel.  The min/max are computed on device.
    if raster.nodata is not None and not np.isnan(raster.nodata):
        pad_value = float(raster.nodata)
    else:
        # Use a value outside the data range (works even for NaN nodata).
        # Compute on device; single scalar D2H is unavoidable for the branch.
        d_finite = d_data_f64[cp.isfinite(d_data_f64)]
        if d_finite.size > 0:
            pad_value = float(d_finite.min().item())  # single scalar D2H
            pad_value -= 1.0
        else:
            pad_value = 0.0

    # Pad on device: allocate padded buffer and copy interior via slice
    d_padded = cp.full((orig_height + 2, orig_width + 2), pad_value, dtype=cp.float64)
    d_padded[1:-1, 1:-1] = d_data_f64

    height, width = d_padded.shape
    cell_width = width - 1
    cell_height = height - 1
    total_cells = cell_width * cell_height

    if total_cells == 0:
        return [], []

    # d_padded is the device raster used by kernels (replaces old d_raster)
    d_raster = d_padded

    # Nodata mask for the padded raster (all on device):
    # Border pixels are always nodata; interior pixels inherit the original mask.
    d_nodata_mask = cp.ones((height, width), dtype=cp.uint8)  # border = nodata
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            d_nodata_mask[1:-1, 1:-1] = cp.isnan(d_data).astype(cp.uint8)
        else:
            d_nodata_mask[1:-1, 1:-1] = (d_data == raster.nodata).astype(cp.uint8)
    else:
        d_nodata_mask[1:-1, 1:-1] = 0  # no nodata in original data
    nodata_mask_ptr = d_nodata_mask.data.ptr

    # --- Find unique non-nodata values on device ---
    # Each unique value gets its own binary marching-squares pass so that
    # every region produces a complete set of closed boundary rings.
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            d_valid = d_data_f64[~cp.isnan(d_data_f64)]
        else:
            d_valid = d_data_f64[d_data_f64 != raster.nodata]
        d_unique = cp.unique(d_valid)
    else:
        d_unique = cp.unique(d_data_f64)

    # Exclude the padding sentinel from unique values (it is never a real value)
    d_unique = d_unique[d_unique != pad_value]

    # Single bulk D2H for the (small) unique-values array -- needed to drive
    # the per-value Python loop below.
    unique_values = cp.asnumpy(d_unique)

    if len(unique_values) == 0:
        return [], []

    # --- Compile kernels (once, reused across all value iterations) ---
    runtime = get_cuda_runtime()

    classify_key = make_kernel_cache_key("classify_cells", CLASSIFY_CELLS_KERNEL_SOURCE)
    classify_kernels = runtime.compile_kernels(
        cache_key=classify_key,
        source=CLASSIFY_CELLS_KERNEL_SOURCE,
        kernel_names=("classify_cells",),
    )

    edge_count_key = make_kernel_cache_key("edge_count", EDGE_COUNT_KERNEL_SOURCE)
    edge_count_kernels = runtime.compile_kernels(
        cache_key=edge_count_key,
        source=EDGE_COUNT_KERNEL_SOURCE,
        kernel_names=("edge_count",),
    )

    emit_key = make_kernel_cache_key("emit_edges", EMIT_EDGES_KERNEL_SOURCE)
    emit_kernels = runtime.compile_kernels(
        cache_key=emit_key,
        source=EMIT_EDGES_KERNEL_SOURCE,
        kernel_names=("emit_edges",),
    )

    # Occupancy-based launch config for classify (total_cells items)
    classify_grid, classify_block = runtime.launch_config(
        classify_kernels["classify_cells"], total_cells
    )

    # Precompute affine transform parameters (shared across all value iters)
    a, b, c_aff, d_aff, e_aff, f_aff = raster.affine

    # Adjust the affine translation to account for the 1-pixel padding.
    # In the padded raster, pixel (col_p, row_p) corresponds to original
    # pixel (col_p - 1, row_p - 1).  The kernel computes:
    #   world_x = a * col_p + b * row_p + c_pad
    # We need this to equal the original:
    #   world_x = a * (col_p - 1) + b * (row_p - 1) + c_aff
    #           = a * col_p + b * row_p + (c_aff - a - b)
    # Same logic for the y components.
    c_pad = c_aff - a - b
    f_pad = f_aff - d_aff - e_aff

    # Precompute affine half-step offsets on the host (4 scalar doubles).
    half_dx_x = 0.5 * a  # x-shift for half column step
    half_dx_y = 0.5 * d_aff  # y-shift for half column step
    half_dy_x = 0.5 * b  # x-shift for half row step
    half_dy_y = 0.5 * e_aff  # y-shift for half row step

    # Pre-allocate reusable device buffers for classification
    d_cell_class = cp.zeros(total_cells, dtype=cp.int32)
    d_cell_indices = cp.arange(total_cells, dtype=cp.int64)

    # Accumulate edge arrays across all unique-value passes
    all_edge_x0 = []
    all_edge_y0 = []
    all_edge_x1 = []
    all_edge_y1 = []
    all_edge_value = []
    total_nontrivial = 0
    grand_total_edges = 0

    # --- Per-value marching-squares pass ---
    for target_val in unique_values:
        target_val_f64 = float(target_val)

        # 1. Binary classify cells for this target value
        classify_params = (
            (
                d_raster.data.ptr,
                nodata_mask_ptr,
                d_cell_class.data.ptr,
                width,
                cell_width,
                cell_height,
                target_val_f64,
            ),
            (
                KERNEL_PARAM_PTR,  # raster
                KERNEL_PARAM_PTR,  # nodata_mask
                KERNEL_PARAM_PTR,  # cell_class
                KERNEL_PARAM_I32,  # raster_width
                KERNEL_PARAM_I32,  # cell_width
                KERNEL_PARAM_I32,  # cell_height
                KERNEL_PARAM_F64,  # target_value
            ),
        )

        runtime.launch(
            kernel=classify_kernels["classify_cells"],
            grid=classify_grid,
            block=classify_block,
            params=classify_params,
        )

        # 2. Compact non-trivial cells (class != 0 and class != 15)
        nontrivial_mask = (d_cell_class != 0) & (d_cell_class != 15)
        compact_idx = d_cell_indices[nontrivial_mask]
        compact_class = d_cell_class[nontrivial_mask]
        n_nontrivial = compact_idx.shape[0]

        if n_nontrivial == 0:
            continue

        total_nontrivial += n_nontrivial

        # 3. Compute edge counts and prefix sum for write offsets
        d_edge_counts = cp.zeros(n_nontrivial, dtype=cp.int32)

        ec_grid, ec_block = runtime.launch_config(edge_count_kernels["edge_count"], n_nontrivial)
        ec_params = (
            (
                compact_class.data.ptr,
                d_edge_counts.data.ptr,
                n_nontrivial,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
            ),
        )
        runtime.launch(
            kernel=edge_count_kernels["edge_count"],
            grid=ec_grid,
            block=ec_block,
            params=ec_params,
        )

        # Prefix sum for write offsets (CuPy cumsum + shift)
        d_edge_offsets = cp.empty(n_nontrivial, dtype=cp.int32)
        d_edge_offsets[0] = 0
        if n_nontrivial > 1:
            cp.cumsum(d_edge_counts[:-1], out=d_edge_offsets[1:])

        total_edges = int(cp.sum(d_edge_counts).item())

        if total_edges == 0:
            continue

        grand_total_edges += total_edges

        # 4. Allocate and emit edges for this value
        d_edge_x0 = cp.empty(total_edges, dtype=cp.float64)
        d_edge_y0 = cp.empty(total_edges, dtype=cp.float64)
        d_edge_x1 = cp.empty(total_edges, dtype=cp.float64)
        d_edge_y1 = cp.empty(total_edges, dtype=cp.float64)
        d_edge_value = cp.empty(total_edges, dtype=cp.float64)

        emit_grid, emit_block = runtime.launch_config(emit_kernels["emit_edges"], n_nontrivial)
        emit_params = (
            (
                compact_idx.data.ptr,
                compact_class.data.ptr,
                d_edge_offsets.data.ptr,
                d_edge_x0.data.ptr,
                d_edge_y0.data.ptr,
                d_edge_x1.data.ptr,
                d_edge_y1.data.ptr,
                d_edge_value.data.ptr,
                cell_width,
                n_nontrivial,
                target_val_f64,
                a,
                b,
                c_pad,
                d_aff,
                e_aff,
                f_pad,
                half_dx_x,
                half_dx_y,
                half_dy_x,
                half_dy_y,
            ),
            (
                KERNEL_PARAM_PTR,  # compact_cell_idx
                KERNEL_PARAM_PTR,  # compact_cell_class
                KERNEL_PARAM_PTR,  # edge_offsets
                KERNEL_PARAM_PTR,  # edge_x0
                KERNEL_PARAM_PTR,  # edge_y0
                KERNEL_PARAM_PTR,  # edge_x1
                KERNEL_PARAM_PTR,  # edge_y1
                KERNEL_PARAM_PTR,  # edge_value
                KERNEL_PARAM_I32,  # cell_width
                KERNEL_PARAM_I32,  # n_cells
                KERNEL_PARAM_F64,  # target_value
                KERNEL_PARAM_F64,  # aff_a
                KERNEL_PARAM_F64,  # aff_b
                KERNEL_PARAM_F64,  # aff_c (pad-adjusted)
                KERNEL_PARAM_F64,  # aff_d
                KERNEL_PARAM_F64,  # aff_e
                KERNEL_PARAM_F64,  # aff_f (pad-adjusted)
                KERNEL_PARAM_F64,  # half_dx_x
                KERNEL_PARAM_F64,  # half_dx_y
                KERNEL_PARAM_F64,  # half_dy_x
                KERNEL_PARAM_F64,  # half_dy_y
            ),
        )

        runtime.launch(
            kernel=emit_kernels["emit_edges"],
            grid=emit_grid,
            block=emit_block,
            params=emit_params,
        )

        # Accumulate on-device arrays (concatenated after the loop)
        all_edge_x0.append(d_edge_x0)
        all_edge_y0.append(d_edge_y0)
        all_edge_x1.append(d_edge_x1)
        all_edge_y1.append(d_edge_y1)
        all_edge_value.append(d_edge_value)

    if grand_total_edges == 0:
        return [], []

    # --- Concatenate all edge arrays and transfer to host ---
    # Single bulk D2H transfer point; all prior GPU work completes
    # implicitly via cp.asnumpy's internal synchronization.
    if len(all_edge_x0) == 1:
        # Fast path: single value, no concatenation needed
        h_edge_x0 = cp.asnumpy(all_edge_x0[0])
        h_edge_y0 = cp.asnumpy(all_edge_y0[0])
        h_edge_x1 = cp.asnumpy(all_edge_x1[0])
        h_edge_y1 = cp.asnumpy(all_edge_y1[0])
        h_edge_value = cp.asnumpy(all_edge_value[0])
    else:
        h_edge_x0 = cp.asnumpy(cp.concatenate(all_edge_x0))
        h_edge_y0 = cp.asnumpy(cp.concatenate(all_edge_y0))
        h_edge_x1 = cp.asnumpy(cp.concatenate(all_edge_x1))
        h_edge_y1 = cp.asnumpy(cp.concatenate(all_edge_y1))
        h_edge_value = cp.asnumpy(cp.concatenate(all_edge_value))

    # --- Chain edges into rings and build polygons ---
    value_rings = _chain_edges_to_rings(
        h_edge_x0,
        h_edge_y0,
        h_edge_x1,
        h_edge_y1,
        h_edge_value,
    )

    geometries, values = _rings_to_polygons(value_rings, spec)

    elapsed = time.perf_counter() - t0

    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_polygonize classify_cells={total_cells} "
                f"unique_values={len(unique_values)} "
                f"nontrivial={total_nontrivial} edges={grand_total_edges} "
                f"polygons={len(geometries)} elapsed={elapsed:.3f}s"
            ),
            residency=raster.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )

    return geometries, values


# ---------------------------------------------------------------------------
# Polygonize pipeline plan
# ---------------------------------------------------------------------------


def plan_polygonize_pipeline() -> FusionPlan:
    """Create a fusion plan for the polygonize pipeline."""
    steps = [
        PipelineStep(
            name="read_raster",
            kind=StepKind.RASTER,
            output_name="raster_data",
            materializes_host_output=True,
        ),
        PipelineStep(
            name="nodata_mask",
            kind=StepKind.DERIVED,
            output_name="valid_mask",
        ),
        PipelineStep(
            name="sieve_filter",
            kind=StepKind.FILTER,
            output_name="sieved_data",
        ),
        PipelineStep(
            name="connected_components",
            kind=StepKind.DERIVED,
            output_name="labels",
            reusable_output=True,
        ),
        PipelineStep(
            name="contour_extraction",
            kind=StepKind.GEOMETRY,
            output_name="raw_rings",
        ),
        PipelineStep(
            name="ring_assembly",
            kind=StepKind.GEOMETRY,
            output_name="oriented_polygons",
        ),
        PipelineStep(
            name="build_geodataframe",
            kind=StepKind.MATERIALIZATION,
            output_name="output_gdf",
            materializes_host_output=True,
        ),
    ]
    return plan_fusion(steps)


# ---------------------------------------------------------------------------
# Full polygonize entry point
# ---------------------------------------------------------------------------


def polygonize_owned(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
    simplify_tolerance: float | None = None,
    max_polygons: int | None = 1_000_000,
    value_field: str = "value",
    use_gpu: bool | None = None,
) -> tuple[list, list]:
    """Polygonize a raster into geometries and values.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster to polygonize.
    connectivity : int
        4 or 8 neighbor connectivity.
    simplify_tolerance : float or None
        Optional Douglas-Peucker simplification tolerance.
    max_polygons : int or None
        Maximum polygon count (explosion guardrail).
    value_field : str
        Name for the value attribute.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    (geometries, values)
        Lists of Shapely geometries and their associated raster values.
    """
    spec = PolygonizeSpec(
        connectivity=connectivity,
        value_field=value_field,
        simplify_tolerance=simplify_tolerance,
        max_polygons=max_polygons,
    )

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    t0 = time.perf_counter()
    if use_gpu:
        geometries, values = polygonize_gpu(raster, spec=spec)
    else:
        geometries, values = polygonize_cpu(raster, spec=spec)
    elapsed = time.perf_counter() - t0

    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"polygonize {'gpu' if use_gpu else 'cpu'} "
                f"connectivity={connectivity} polygons={len(geometries)} "
                f"elapsed={elapsed:.3f}s"
            ),
            residency=raster.residency,
        )
    )

    return geometries, values


def polygonize_to_gdf(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
    simplify_tolerance: float | None = None,
    max_polygons: int | None = 1_000_000,
    value_field: str = "value",
):
    """Polygonize a raster into a GeoDataFrame.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    connectivity, simplify_tolerance, max_polygons, value_field
        See polygonize_owned.

    Returns
    -------
    GeoDataFrame
        With geometry column and a value column named `value_field`.
    """
    import geopandas as gpd

    geometries, values = polygonize_owned(
        raster,
        connectivity=connectivity,
        simplify_tolerance=simplify_tolerance,
        max_polygons=max_polygons,
        value_field=value_field,
    )

    return gpd.GeoDataFrame(
        {value_field: values},
        geometry=geometries,
        crs=raster.crs,
    )
