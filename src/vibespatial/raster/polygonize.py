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

            if len(ring) >= 4:  # minimum polygon: 3 vertices + closing vertex
                # Ensure ring is closed
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
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

    Pipeline:
      1. Transfer raster to device
      2. classify_cells kernel — classify each 2x2 cell (one thread per cell)
      3. Compact non-trivial cells via CuPy boolean indexing
      4. edge_count kernel + exclusive prefix sum — compute write offsets
      5. emit_edges kernel — emit directed edge segments with affine transform
      6. Transfer edges to host for ring assembly (graph traversal)
      7. Chain edges into rings, convert to Shapely geometries

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

    # --- Prepare raster data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape
    cell_width = width - 1
    cell_height = height - 1
    total_cells = cell_width * cell_height

    if total_cells == 0:
        return [], []

    # Transfer to device
    d_raster = cp.asarray(np.ascontiguousarray(data.astype(np.float64)))

    # Nodata mask
    d_nodata_mask = None
    nodata_mask_ptr = 0
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            host_mask = np.isnan(data).astype(np.uint8)
        else:
            host_mask = (data == raster.nodata).astype(np.uint8)
        d_nodata_mask = cp.asarray(host_mask)
        nodata_mask_ptr = d_nodata_mask.data.ptr

    # Allocate classification outputs
    d_cell_class = cp.zeros(total_cells, dtype=cp.int32)
    d_cell_value = cp.zeros(total_cells, dtype=cp.float64)

    # --- Compile and launch classify_cells kernel ---
    runtime = get_cuda_runtime()

    classify_key = make_kernel_cache_key("classify_cells", CLASSIFY_CELLS_KERNEL_SOURCE)
    classify_kernels = runtime.compile_kernels(
        cache_key=classify_key,
        source=CLASSIFY_CELLS_KERNEL_SOURCE,
        kernel_names=("classify_cells",),
    )

    block_size = 256
    grid_size = (total_cells + block_size - 1) // block_size

    classify_params = (
        (
            d_raster.data.ptr,
            nodata_mask_ptr,
            d_cell_class.data.ptr,
            d_cell_value.data.ptr,
            width,
            height,
            cell_width,
            cell_height,
        ),
        (
            KERNEL_PARAM_PTR,  # raster
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_PTR,  # cell_class
            KERNEL_PARAM_PTR,  # cell_value
            KERNEL_PARAM_I32,  # raster_width
            KERNEL_PARAM_I32,  # raster_height
            KERNEL_PARAM_I32,  # cell_width
            KERNEL_PARAM_I32,  # cell_height
        ),
    )

    runtime.launch(
        kernel=classify_kernels["classify_cells"],
        grid=(grid_size, 1, 1),
        block=(block_size, 1, 1),
        params=classify_params,
    )

    # --- Compact non-trivial cells (class != 0 and class != 15) ---
    # Use CuPy boolean indexing (zero-copy on device)
    nontrivial_mask = (d_cell_class != 0) & (d_cell_class != 15)
    d_cell_indices = cp.arange(total_cells, dtype=cp.int32)

    compact_idx = d_cell_indices[nontrivial_mask]
    compact_class = d_cell_class[nontrivial_mask]
    compact_value = d_cell_value[nontrivial_mask]

    n_nontrivial = int(compact_idx.size)

    if n_nontrivial == 0:
        # All cells are trivial — the raster is homogeneous or all-nodata.
        # Still need to produce polygons for homogeneous regions.
        # Fall through to the ring assembly with no edges.
        elapsed = time.perf_counter() - t0
        raster.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=f"gpu_polygonize classify_cells={total_cells} nontrivial=0 elapsed={elapsed:.3f}s",
                residency=raster.residency,
                visible_to_user=True,
                elapsed_seconds=elapsed,
            )
        )
        # For a homogeneous raster with no boundary cells, produce a single polygon
        # covering the entire raster extent (if not all nodata).
        unique_vals = np.unique(data)
        if raster.nodata is not None:
            if np.isnan(raster.nodata):
                unique_vals = unique_vals[~np.isnan(unique_vals)]
            else:
                unique_vals = unique_vals[unique_vals != raster.nodata]
        if len(unique_vals) == 0:
            return [], []

        a, b, c_aff, d_aff, e_aff, f_aff = raster.affine
        geometries = []
        values_out = []
        for v in unique_vals:
            # Build a polygon from the raster extent corners
            corners = [
                (c_aff, f_aff),  # TL: col=0, row=0
                (a * width + c_aff, d_aff * width + f_aff),  # TR: col=width, row=0
                (a * width + b * height + c_aff, d_aff * width + e_aff * height + f_aff),  # BR
                (b * height + c_aff, e_aff * height + f_aff),  # BL: col=0, row=height
                (c_aff, f_aff),  # close ring
            ]
            poly = Polygon(corners)
            if not poly.is_empty and poly.is_valid:
                geometries.append(poly)
                values_out.append(float(v))
        return geometries, values_out

    # --- Compute edge counts and prefix sum for write offsets ---
    d_edge_counts = cp.zeros(n_nontrivial, dtype=cp.int32)

    edge_count_key = make_kernel_cache_key("edge_count", EDGE_COUNT_KERNEL_SOURCE)
    edge_count_kernels = runtime.compile_kernels(
        cache_key=edge_count_key,
        source=EDGE_COUNT_KERNEL_SOURCE,
        kernel_names=("edge_count",),
    )

    ec_grid = (n_nontrivial + block_size - 1) // block_size
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
        grid=(ec_grid, 1, 1),
        block=(block_size, 1, 1),
        params=ec_params,
    )

    # Exclusive prefix sum for write offsets (CuPy cumsum)
    d_edge_offsets = cp.zeros(n_nontrivial, dtype=cp.int32)
    if n_nontrivial > 0:
        cumsum = cp.cumsum(d_edge_counts)
        # exclusive sum: shift right by 1, first element = 0
        d_edge_offsets[1:] = cumsum[:-1]
        d_edge_offsets[0] = 0

    total_edges = int(cp.sum(d_edge_counts).get())

    if total_edges == 0:
        return [], []

    # --- Allocate edge output arrays ---
    d_edge_x0 = cp.zeros(total_edges, dtype=cp.float64)
    d_edge_y0 = cp.zeros(total_edges, dtype=cp.float64)
    d_edge_x1 = cp.zeros(total_edges, dtype=cp.float64)
    d_edge_y1 = cp.zeros(total_edges, dtype=cp.float64)
    d_edge_value = cp.zeros(total_edges, dtype=cp.float64)

    # --- Compile and launch emit_edges kernel ---
    emit_key = make_kernel_cache_key("emit_edges", EMIT_EDGES_KERNEL_SOURCE)
    emit_kernels = runtime.compile_kernels(
        cache_key=emit_key,
        source=EMIT_EDGES_KERNEL_SOURCE,
        kernel_names=("emit_edges",),
    )

    a, b, c_aff, d_aff, e_aff, f_aff = raster.affine

    emit_grid = (n_nontrivial + block_size - 1) // block_size
    emit_params = (
        (
            compact_idx.data.ptr,
            compact_class.data.ptr,
            compact_value.data.ptr,
            d_edge_offsets.data.ptr,
            d_edge_x0.data.ptr,
            d_edge_y0.data.ptr,
            d_edge_x1.data.ptr,
            d_edge_y1.data.ptr,
            d_edge_value.data.ptr,
            cell_width,
            n_nontrivial,
            a,
            b,
            c_aff,
            d_aff,
            e_aff,
            f_aff,
        ),
        (
            KERNEL_PARAM_PTR,  # compact_cell_idx
            KERNEL_PARAM_PTR,  # compact_cell_class
            KERNEL_PARAM_PTR,  # compact_cell_value
            KERNEL_PARAM_PTR,  # edge_offsets
            KERNEL_PARAM_PTR,  # edge_x0
            KERNEL_PARAM_PTR,  # edge_y0
            KERNEL_PARAM_PTR,  # edge_x1
            KERNEL_PARAM_PTR,  # edge_y1
            KERNEL_PARAM_PTR,  # edge_value
            KERNEL_PARAM_I32,  # cell_width
            KERNEL_PARAM_I32,  # n_cells
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,  # aff a, b, c
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,
            KERNEL_PARAM_F64,  # aff d, e, f
        ),
    )

    runtime.launch(
        kernel=emit_kernels["emit_edges"],
        grid=(emit_grid, 1, 1),
        block=(block_size, 1, 1),
        params=emit_params,
    )

    # --- Transfer edges to host for ring assembly ---
    h_edge_x0 = cp.asnumpy(d_edge_x0)
    h_edge_y0 = cp.asnumpy(d_edge_y0)
    h_edge_x1 = cp.asnumpy(d_edge_x1)
    h_edge_y1 = cp.asnumpy(d_edge_y1)
    h_edge_value = cp.asnumpy(d_edge_value)

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
                f"nontrivial={n_nontrivial} edges={total_edges} "
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
