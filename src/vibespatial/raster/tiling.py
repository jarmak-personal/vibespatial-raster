"""Tiling execution engine for raster operations.

Phase 1 (vibeSpatial-fx3.2): pointwise (zero-overlap) tiling.  Processes
rasters in spatial chunks that fit in VRAM, enabling large-raster GPU
processing without OOM.  Each tile undergoes a HOST->DEVICE->HOST round
trip independently so that the full raster never needs to reside on the
GPU at once.

Phase 2 (vibeSpatial-fx3.3): halo tiling for stencil/focal operations.
Extends Phase 1 with overlap (halo) pixels so that each tile includes
enough border context for neighbourhood kernels (convolution, slope,
hillshade, morphology, focal statistics).  The result is trimmed back to
the effective tile region and stitched seamlessly.

Phase 3 (vibeSpatial-fx3.4): accumulator tiling for reduce operations.
Map-reduce pattern for operations that summarize a raster into non-raster
data (histograms, percentiles, zonal statistics).  Each tile produces a
partial accumulator which is merged pairwise via a caller-provided
``merge_fn``.

Phase 4 (vibeSpatial-fx3.5): multi-pass tiling for CCL and distance
transform.  Two-pass pattern for operations that require cross-tile
communication: (1) local pass processes each tile independently,
(2) boundary merge reconciles results across tile seams on host.

ADR: vibeSpatial-fx3.2  Phase 1: Trivial tiling for pointwise operations
ADR: vibeSpatial-fx3.3  Phase 2: Halo tiling for stencil operations
ADR: vibeSpatial-fx3.4  Phase 3: Accumulator tiling for histogram/zonal
ADR: vibeSpatial-fx3.5  Phase 4: Multi-pass tiling for CCL/distance
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vibespatial.raster.buffers import OwnedRasterArray, RasterPlan

__all__ = [
    "dispatch_tiled",
    "dispatch_tiled_accumulator",
    "dispatch_tiled_binary",
    "dispatch_tiled_halo",
    "dispatch_tiled_multipass",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tile_bounds(
    tile_row: int,
    tile_col: int,
    tile_h: int,
    tile_w: int,
    raster_h: int,
    raster_w: int,
) -> tuple[int, int, int, int]:
    """Return (row_start, row_end, col_start, col_end) clamped to raster bounds.

    Parameters
    ----------
    tile_row, tile_col:
        0-indexed tile position in the tile grid.
    tile_h, tile_w:
        Nominal tile dimensions (height, width).
    raster_h, raster_w:
        Full raster spatial dimensions.

    Returns
    -------
    tuple[int, int, int, int]
        ``(row_start, row_end, col_start, col_end)`` where slicing with
        ``data[..., row_start:row_end, col_start:col_end]`` extracts the
        tile, correctly clamped for edge tiles.
    """
    row_start = tile_row * tile_h
    row_end = min(row_start + tile_h, raster_h)
    col_start = tile_col * tile_w
    col_end = min(col_start + tile_w, raster_w)
    return row_start, row_end, col_start, col_end


def _adjust_affine(
    affine: tuple[float, float, float, float, float, float],
    row_offset: int,
    col_offset: int,
) -> tuple[float, float, float, float, float, float]:
    """Shift affine transform origin to account for tile position.

    The affine is ``(a, b, c, d, e, f)`` where::

        world_x = a * col + b * row + c
        world_y = d * col + e * row + f

    For a tile starting at pixel ``(row_offset, col_offset)`` the new
    origin is::

        new_c = c + col_offset * a + row_offset * b
        new_f = f + col_offset * d + row_offset * e

    Parameters
    ----------
    affine:
        6-element GDAL-style affine of the full raster.
    row_offset, col_offset:
        Pixel offset of the tile's upper-left corner within the full raster.

    Returns
    -------
    tuple[float, float, float, float, float, float]
        Adjusted affine for the tile.
    """
    a, b, c, d, e, f = affine
    new_c = c + col_offset * a + row_offset * b
    new_f = f + col_offset * d + row_offset * e
    return (a, b, new_c, d, e, new_f)


def _ensure_host_resident(raster: OwnedRasterArray, *, label: str) -> np.ndarray:
    """Return the host numpy array, raising if the raster is DEVICE-resident.

    Phase 1 tiling requires HOST-resident input because the whole point of
    tiling is that the full raster does not fit in VRAM.  A DEVICE-resident
    raster would require a full D->H transfer via ``to_numpy()``, defeating
    the OOM-avoidance goal.

    Raises
    ------
    ValueError
        If the raster is DEVICE-resident.
    """
    from vibespatial.residency import Residency

    if raster.residency == Residency.DEVICE:
        raise ValueError(
            f"{label} requires HOST-resident input; a DEVICE-resident raster "
            "cannot be tiled because the full D->H transfer would defeat the "
            "OOM-avoidance goal.  Call raster.move_to(Residency.HOST) first, "
            "or use WHOLE strategy for device-resident data."
        )
    return raster.to_numpy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dispatch_tiled(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    plan: RasterPlan,
) -> OwnedRasterArray:
    """Execute a unary pointwise operation using spatial tiling.

    Parameters
    ----------
    raster:
        Input raster (single- or multi-band).  Must be HOST-resident for
        the TILED path; the WHOLE fast path accepts any residency.
    op_fn:
        The operation to apply per tile.  Receives a tile-sized
        ``OwnedRasterArray`` and returns a result of the same spatial
        dimensions.  The operation may internally transfer the tile to
        device and back -- this is the expected pattern for tiled GPU
        processing.
    plan:
        A frozen ``RasterPlan`` produced by ``analyze_raster_plan()``.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, CRS, and nodata as the
        input.  The dtype is determined by ``op_fn``'s output.

    Raises
    ------
    ValueError
        If ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if the raster is DEVICE-resident on the TILED path.
    """
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
        TilingStrategy,
        from_numpy,
    )
    from vibespatial.residency import Residency

    # -- WHOLE fast path: no tiling overhead --
    if plan.strategy == TilingStrategy.WHOLE:
        return op_fn(raster)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()

    tile_h, tile_w = plan.tile_shape
    host = _ensure_host_resident(raster, label="dispatch_tiled")
    raster_h = raster.height
    raster_w = raster.width

    # Lazy output allocation: deferred until first tile result so that the
    # output dtype matches op_fn's actual return dtype (which may differ
    # from the input dtype, e.g. classify returning uint8 from float32).
    output: np.ndarray | None = None

    # Compute tile grid dimensions.
    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    # TODO(fx3-perf): Overlap H->D transfer of tile N+1 with GPU compute
    # on tile N using two CUDA streams (double-buffering).  Current serial
    # execution leaves the GPU idle during host-side tile preparation.
    tiles_processed = 0
    result_nodata: float | int | None = raster.nodata
    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            rs, re, cs, ce = _tile_bounds(tr, tc, tile_h, tile_w, raster_h, raster_w)

            # Slice tile from host array.  ascontiguousarray() is a no-op
            # for already-contiguous slices (e.g. full-row tiles) and
            # ensures contiguity for interior tiles where the slice has
            # non-unit stride along the column axis.
            if host.ndim == 3:
                tile_data = np.ascontiguousarray(host[:, rs:re, cs:ce])
            else:
                tile_data = np.ascontiguousarray(host[rs:re, cs:ce])

            # Adjust affine for this tile's spatial position.
            tile_affine = _adjust_affine(raster.affine, row_offset=rs, col_offset=cs)

            # Wrap tile as OwnedRasterArray (HOST-resident).
            tile_raster = from_numpy(
                tile_data,
                nodata=raster.nodata,
                affine=tile_affine,
                crs=raster.crs,
            )

            # Apply the operation (op may H->D->H internally).
            tile_result = op_fn(tile_raster)

            # Retrieve result to host.
            tile_host = tile_result.to_numpy()

            # Lazy allocation on first tile result.
            if output is None:
                if host.ndim == 3:
                    output = np.empty(
                        (host.shape[0], raster_h, raster_w),
                        dtype=tile_host.dtype,
                    )
                else:
                    output = np.empty((raster_h, raster_w), dtype=tile_host.dtype)
                result_nodata = tile_result.nodata

            if output.ndim == 3:
                output[:, rs:re, cs:ce] = tile_host
            else:
                output[rs:re, cs:ce] = tile_host

            # Release tile references.  CPython's reference counting
            # triggers immediate deallocation, and both the RMM pool and
            # CuPy default pool immediately reclaim freed device blocks
            # for reuse by the next tile's allocation.
            del tile_raster, tile_result, tile_host, tile_data

            tiles_processed += 1

    elapsed = time.perf_counter() - t0

    # Guard against zero-tile degenerate rasters.
    if output is None:
        raise ValueError("Zero tiles processed; input raster has degenerate dimensions")

    result = from_numpy(
        output,
        nodata=result_nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_tiled unary tiles={tiles_processed} "
                f"tile_shape=({tile_h},{tile_w}) "
                f"raster_shape={raster.shape} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result


def dispatch_tiled_binary(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray, OwnedRasterArray], OwnedRasterArray],
    plan: RasterPlan,
) -> OwnedRasterArray:
    """Execute a binary pointwise operation using spatial tiling.

    Same tiling strategy as :func:`dispatch_tiled` but extracts the same
    tile region from both input rasters and passes them to a binary
    ``op_fn``.

    Parameters
    ----------
    a, b:
        Input rasters.  Must have the same spatial dimensions.  Must be
        HOST-resident for the TILED path.
    op_fn:
        Binary operation.  Receives two tile-sized ``OwnedRasterArray``
        objects and returns a result of the same spatial dimensions.
    plan:
        A frozen ``RasterPlan`` produced by ``analyze_raster_plan()``.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, CRS, and nodata as ``a``.
        The dtype is determined by ``op_fn``'s output.

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different spatial dimensions, if
        ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if either input is DEVICE-resident on the TILED path.
    """
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
        TilingStrategy,
        from_numpy,
    )
    from vibespatial.residency import Residency

    if a.height != b.height or a.width != b.width:
        raise ValueError(
            f"Spatial dimension mismatch: a=({a.height},{a.width}), b=({b.height},{b.width})"
        )

    # -- WHOLE fast path: no tiling overhead --
    if plan.strategy == TilingStrategy.WHOLE:
        return op_fn(a, b)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()

    tile_h, tile_w = plan.tile_shape
    host_a = _ensure_host_resident(a, label="dispatch_tiled_binary (input a)")
    host_b = _ensure_host_resident(b, label="dispatch_tiled_binary (input b)")
    raster_h = a.height
    raster_w = a.width

    # Lazy output allocation: deferred until first tile result.
    output: np.ndarray | None = None

    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    # TODO(fx3-perf): Double-buffer with two CUDA streams for overlap.
    tiles_processed = 0
    result_nodata: float | int | None = a.nodata

    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            rs, re, cs, ce = _tile_bounds(tr, tc, tile_h, tile_w, raster_h, raster_w)

            # Slice tiles from host arrays (contiguous for DMA).
            if host_a.ndim == 3:
                tile_a_data = np.ascontiguousarray(host_a[:, rs:re, cs:ce])
            else:
                tile_a_data = np.ascontiguousarray(host_a[rs:re, cs:ce])

            if host_b.ndim == 3:
                tile_b_data = np.ascontiguousarray(host_b[:, rs:re, cs:ce])
            else:
                tile_b_data = np.ascontiguousarray(host_b[rs:re, cs:ce])

            tile_affine = _adjust_affine(a.affine, row_offset=rs, col_offset=cs)

            tile_a = from_numpy(
                tile_a_data,
                nodata=a.nodata,
                affine=tile_affine,
                crs=a.crs,
            )
            tile_b = from_numpy(
                tile_b_data,
                nodata=b.nodata,
                affine=tile_affine,
                crs=b.crs,
            )

            tile_result = op_fn(tile_a, tile_b)
            tile_host = tile_result.to_numpy()

            # Lazy output allocation: use first tile's result dtype.
            if output is None:
                if host_a.ndim == 3:
                    output = np.empty(
                        (host_a.shape[0], raster_h, raster_w),
                        dtype=tile_host.dtype,
                    )
                else:
                    output = np.empty((raster_h, raster_w), dtype=tile_host.dtype)
                result_nodata = tile_result.nodata

            if output.ndim == 3:
                output[:, rs:re, cs:ce] = tile_host
            else:
                output[rs:re, cs:ce] = tile_host

            del tile_a, tile_b, tile_result, tile_host, tile_a_data, tile_b_data

            tiles_processed += 1

    elapsed = time.perf_counter() - t0

    # Guard against zero-tile degenerate rasters.
    if output is None:
        raise ValueError("Zero tiles processed; input rasters have degenerate dimensions")

    result = from_numpy(
        output,
        nodata=result_nodata,
        affine=a.affine,
        crs=a.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_tiled_binary tiles={tiles_processed} "
                f"tile_shape=({tile_h},{tile_w}) "
                f"raster_shape={a.shape} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result


def dispatch_tiled_halo(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    plan: RasterPlan,
) -> OwnedRasterArray:
    """Execute a unary stencil/focal operation using halo-aware spatial tiling.

    Similar to :func:`dispatch_tiled` but each tile is extracted with extra
    *halo* border pixels so that neighbourhood kernels (convolution, slope,
    hillshade, morphology, focal statistics) have sufficient context at tile
    edges.  After ``op_fn`` processes the full physical tile the halo border
    is trimmed and only the interior (effective) region is stitched into the
    output.

    Parameters
    ----------
    raster:
        Input raster (single- or multi-band).  Must be HOST-resident for
        the TILED path; the WHOLE fast path accepts any residency.
    op_fn:
        The stencil operation to apply per tile.  Receives a tile-sized
        ``OwnedRasterArray`` that includes halo border pixels and returns a
        result of the **same** spatial dimensions as its input (i.e. the
        operation does not strip the halo itself).  The dispatcher handles
        trimming.
    plan:
        A frozen ``RasterPlan`` produced by ``analyze_raster_plan()``.
        ``plan.halo`` specifies the overlap pixels.  ``plan.tile_shape``
        specifies the **effective** (output) tile dimensions, *not*
        including the halo.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, CRS, and nodata as the
        input.  The dtype is determined by ``op_fn``'s output.

    Raises
    ------
    ValueError
        If ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if the raster is DEVICE-resident on the TILED path.
    """
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
        TilingStrategy,
        from_numpy,
    )
    from vibespatial.residency import Residency

    # -- WHOLE fast path: no tiling overhead --
    if plan.strategy == TilingStrategy.WHOLE:
        return op_fn(raster)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()

    tile_h, tile_w = plan.tile_shape
    halo = plan.halo
    host = _ensure_host_resident(raster, label="dispatch_tiled_halo")
    raster_h = raster.height
    raster_w = raster.width

    # Lazy output allocation: deferred until first tile result so that the
    # output dtype matches op_fn's actual return dtype.
    output: np.ndarray | None = None

    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    tiles_processed = 0
    result_nodata: float | int | None = raster.nodata

    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            # Effective bounds: the output region for this tile (same as
            # dispatch_tiled's non-overlapping partitioning).
            eff_rs, eff_re, eff_cs, eff_ce = _tile_bounds(
                tr,
                tc,
                tile_h,
                tile_w,
                raster_h,
                raster_w,
            )

            # Physical bounds: expand each edge by halo, clamped to raster
            # bounds.  This is the region that gets extracted and fed to
            # op_fn so that the stencil has sufficient neighbourhood context
            # at tile edges.
            phys_rs = max(0, eff_rs - halo)
            phys_re = min(raster_h, eff_re + halo)
            phys_cs = max(0, eff_cs - halo)
            phys_ce = min(raster_w, eff_ce + halo)

            # Slice the physical tile from host data.
            if host.ndim == 3:
                tile_data = np.ascontiguousarray(host[:, phys_rs:phys_re, phys_cs:phys_ce])
            else:
                tile_data = np.ascontiguousarray(host[phys_rs:phys_re, phys_cs:phys_ce])

            # Adjust affine to the physical tile's origin (not the
            # effective origin).  This ensures world-coordinate lookups
            # inside op_fn (e.g. cell_size from affine for slope) are
            # correct for the data that op_fn receives.
            tile_affine = _adjust_affine(raster.affine, row_offset=phys_rs, col_offset=phys_cs)

            tile_raster = from_numpy(
                tile_data,
                nodata=raster.nodata,
                affine=tile_affine,
                crs=raster.crs,
            )

            # Apply the stencil operation on the full physical tile.
            tile_result = op_fn(tile_raster)

            tile_host = tile_result.to_numpy()

            # Trim halo from the result to extract only the interior
            # (effective) region.  At raster boundaries the actual halo
            # may be less than the requested halo because the physical
            # bounds were clamped.
            top_trim = eff_rs - phys_rs
            left_trim = eff_cs - phys_cs
            eff_height = eff_re - eff_rs
            eff_width = eff_ce - eff_cs

            if tile_host.ndim == 3:
                interior = tile_host[
                    :,
                    top_trim : top_trim + eff_height,
                    left_trim : left_trim + eff_width,
                ]
            else:
                interior = tile_host[
                    top_trim : top_trim + eff_height,
                    left_trim : left_trim + eff_width,
                ]

            # Lazy allocation on first tile result.
            if output is None:
                if host.ndim == 3:
                    output = np.empty(
                        (host.shape[0], raster_h, raster_w),
                        dtype=interior.dtype,
                    )
                else:
                    output = np.empty((raster_h, raster_w), dtype=interior.dtype)
                result_nodata = tile_result.nodata

            if output.ndim == 3:
                output[:, eff_rs:eff_re, eff_cs:eff_ce] = interior
            else:
                output[eff_rs:eff_re, eff_cs:eff_ce] = interior

            # Release tile references so RMM/CuPy can reclaim memory.
            del tile_raster, tile_result, tile_host, tile_data, interior

            tiles_processed += 1

    elapsed = time.perf_counter() - t0

    if output is None:
        raise ValueError("Zero tiles processed; input raster has degenerate dimensions")

    result = from_numpy(
        output,
        nodata=result_nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_tiled_halo tiles={tiles_processed} "
                f"tile_shape=({tile_h},{tile_w}) halo={halo} "
                f"raster_shape={raster.shape} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result


def dispatch_tiled_accumulator[T](
    raster: OwnedRasterArray,
    tile_fn: Callable[[OwnedRasterArray], T],
    merge_fn: Callable[[T, T], T],
    plan: RasterPlan,
) -> T:
    """Execute a reduce operation over spatial tiles using map-reduce.

    Unlike :func:`dispatch_tiled` which produces a raster output of the same
    spatial dimensions, this function produces an aggregated result by:

    1. Splitting the raster into non-overlapping tiles (using
       ``plan.tile_shape``).
    2. Applying ``tile_fn`` to each tile to get a partial accumulator.
    3. Merging partial accumulators left-to-right using ``merge_fn``.

    Parameters
    ----------
    raster:
        Input raster (single- or multi-band).  Must be HOST-resident for
        the TILED path; the WHOLE fast path accepts any residency.
    tile_fn:
        Maps a tile ``OwnedRasterArray`` to a partial accumulator of type T.
        For histogram: returns ``(counts_array, bin_edges_array)``.
        For zonal stats: returns ``{zone_id: {stat: value}}``.
    merge_fn:
        Combines two partial accumulators into one.  Must be associative
        (caller's responsibility).
        For histogram: sums counts, keeps bin_edges.
        For zonal stats: merges per-zone sums/counts/min/max.
    plan:
        A frozen ``RasterPlan``.  Uses ``plan.tile_shape`` for tiling
        dimensions.  ``plan.halo`` is ignored (accumulator ops don't
        need overlap).

    Returns
    -------
    T
        The final merged accumulator.

    Raises
    ------
    ValueError
        If ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if the raster is DEVICE-resident on the TILED path, or if the
        raster has degenerate (zero) spatial dimensions.
    """
    from vibespatial.raster.buffers import TilingStrategy, from_numpy

    # -- WHOLE fast path: no tiling overhead --
    if plan.strategy == TilingStrategy.WHOLE:
        return tile_fn(raster)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    tile_h, tile_w = plan.tile_shape
    host = _ensure_host_resident(raster, label="dispatch_tiled_accumulator")
    raster_h = raster.height
    raster_w = raster.width

    # Compute tile grid dimensions.
    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    accumulator: T | None = None
    tiles_processed = 0

    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            rs, re, cs, ce = _tile_bounds(tr, tc, tile_h, tile_w, raster_h, raster_w)

            # Slice tile from host array (contiguous for DMA).
            if host.ndim == 3:
                tile_data = np.ascontiguousarray(host[:, rs:re, cs:ce])
            else:
                tile_data = np.ascontiguousarray(host[rs:re, cs:ce])

            # Adjust affine for this tile's spatial position.
            tile_affine = _adjust_affine(raster.affine, row_offset=rs, col_offset=cs)

            # Wrap tile as OwnedRasterArray (HOST-resident).
            tile_raster = from_numpy(
                tile_data,
                nodata=raster.nodata,
                affine=tile_affine,
                crs=raster.crs,
            )

            # Apply tile_fn to get partial accumulator.
            partial = tile_fn(tile_raster)

            # Merge with running accumulator.
            if accumulator is None:
                accumulator = partial
            else:
                accumulator = merge_fn(accumulator, partial)

            # Release tile references.
            del tile_raster, tile_data, partial

            tiles_processed += 1

    if accumulator is None:
        raise ValueError("Zero tiles processed; input raster has degenerate dimensions")

    return accumulator


def dispatch_tiled_multipass(
    raster: OwnedRasterArray,
    local_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    merge_fn: Callable[[np.ndarray, list[tuple[int, int, int, int]]], np.ndarray],
    plan: RasterPlan,
) -> OwnedRasterArray:
    """Execute a multi-pass operation using spatial tiling with boundary merge.

    Designed for operations that require cross-tile communication (CCL,
    distance transform).  The executor runs two passes:

    1. **Local pass**: Process each tile independently via ``local_fn``.
       Results are assembled into a full-size intermediate array on host.
    2. **Boundary merge**: Call ``merge_fn`` on the full intermediate array
       with tile boundary information so that cross-tile inconsistencies
       (disconnected labels, truncated distances) can be reconciled.

    Parameters
    ----------
    raster:
        Input raster (single- or multi-band).  Must be HOST-resident for
        the TILED path; the WHOLE fast path accepts any residency.
    local_fn:
        Applied to each tile independently in the first pass.  Takes a
        tile-sized ``OwnedRasterArray`` and returns a result of the same
        spatial dimensions.  The operation may internally transfer the tile
        to device and back -- this is the expected pattern for tiled GPU
        processing.
    merge_fn:
        Applied to the full assembled intermediate result (on host) to
        reconcile tile boundaries.  Takes:

        - ``intermediate``: The assembled numpy array from all local_fn
          results, shape ``(H, W)`` or ``(bands, H, W)``.
        - ``tile_bounds_list``: List of ``(row_start, row_end, col_start,
          col_end)`` tuples for each tile (row-major order), so merge_fn
          knows where tile boundaries are.

        Returns the corrected numpy array (same shape).  merge_fn runs
        entirely on host.
    plan:
        A frozen ``RasterPlan`` produced by ``analyze_raster_plan()``.
        ``plan.halo`` is used for the local pass if > 0 (same semantics
        as ``dispatch_tiled_halo``).

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, CRS, and nodata as the
        input.  The dtype is determined by ``local_fn``'s output.

    Raises
    ------
    ValueError
        If ``plan.strategy`` is ``TILED`` but ``plan.tile_shape`` is None,
        or if the raster is DEVICE-resident on the TILED path.
    """
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
        TilingStrategy,
        from_numpy,
    )
    from vibespatial.residency import Residency

    # -- WHOLE fast path: no tiling overhead, no merge needed --
    if plan.strategy == TilingStrategy.WHOLE:
        return local_fn(raster)

    # -- TILED path --
    if plan.tile_shape is None:
        raise ValueError(
            "RasterPlan has strategy=TILED but tile_shape is None; this indicates a malformed plan"
        )

    t0 = time.perf_counter()

    tile_h, tile_w = plan.tile_shape
    halo = plan.halo
    host = _ensure_host_resident(raster, label="dispatch_tiled_multipass")
    raster_h = raster.height
    raster_w = raster.width

    # Lazy output allocation: deferred until first tile result so that the
    # output dtype matches local_fn's actual return dtype.
    intermediate: np.ndarray | None = None

    # Compute tile grid dimensions.
    rows_of_tiles = (raster_h + tile_h - 1) // tile_h
    cols_of_tiles = (raster_w + tile_w - 1) // tile_w

    tiles_processed = 0
    result_nodata: float | int | None = raster.nodata
    tile_bounds_list: list[tuple[int, int, int, int]] = []

    # Pass 1 (Local): Process each tile independently.
    for tr in range(rows_of_tiles):
        for tc in range(cols_of_tiles):
            # Effective bounds: the output region for this tile (non-overlapping).
            eff_rs, eff_re, eff_cs, eff_ce = _tile_bounds(
                tr,
                tc,
                tile_h,
                tile_w,
                raster_h,
                raster_w,
            )
            tile_bounds_list.append((eff_rs, eff_re, eff_cs, eff_ce))

            if halo > 0:
                # Physical bounds: expand each edge by halo, clamped to
                # raster bounds (same as dispatch_tiled_halo).
                phys_rs = max(0, eff_rs - halo)
                phys_re = min(raster_h, eff_re + halo)
                phys_cs = max(0, eff_cs - halo)
                phys_ce = min(raster_w, eff_ce + halo)
            else:
                phys_rs, phys_re = eff_rs, eff_re
                phys_cs, phys_ce = eff_cs, eff_ce

            # Slice tile from host array (contiguous for DMA).
            if host.ndim == 3:
                tile_data = np.ascontiguousarray(host[:, phys_rs:phys_re, phys_cs:phys_ce])
            else:
                tile_data = np.ascontiguousarray(host[phys_rs:phys_re, phys_cs:phys_ce])

            # Adjust affine to the physical tile's origin.
            tile_affine = _adjust_affine(raster.affine, row_offset=phys_rs, col_offset=phys_cs)

            tile_raster = from_numpy(
                tile_data,
                nodata=raster.nodata,
                affine=tile_affine,
                crs=raster.crs,
            )

            # Apply the local operation on the tile.
            tile_result = local_fn(tile_raster)
            tile_host = tile_result.to_numpy()

            # Trim halo if present, extracting only the effective region.
            if halo > 0:
                top_trim = eff_rs - phys_rs
                left_trim = eff_cs - phys_cs
                eff_height = eff_re - eff_rs
                eff_width = eff_ce - eff_cs

                if tile_host.ndim == 3:
                    interior = tile_host[
                        :,
                        top_trim : top_trim + eff_height,
                        left_trim : left_trim + eff_width,
                    ]
                else:
                    interior = tile_host[
                        top_trim : top_trim + eff_height,
                        left_trim : left_trim + eff_width,
                    ]
            else:
                interior = tile_host

            # Lazy allocation on first tile result.
            if intermediate is None:
                if host.ndim == 3:
                    intermediate = np.empty(
                        (host.shape[0], raster_h, raster_w),
                        dtype=interior.dtype,
                    )
                else:
                    intermediate = np.empty((raster_h, raster_w), dtype=interior.dtype)
                result_nodata = tile_result.nodata

            if intermediate.ndim == 3:
                intermediate[:, eff_rs:eff_re, eff_cs:eff_ce] = interior
            else:
                intermediate[eff_rs:eff_re, eff_cs:eff_ce] = interior

            # Release tile references so RMM/CuPy can reclaim memory.
            del tile_raster, tile_result, tile_host, tile_data
            if halo > 0:
                del interior

            tiles_processed += 1

    elapsed_pass1 = time.perf_counter() - t0

    # Guard against zero-tile degenerate rasters.
    if intermediate is None:
        raise ValueError("Zero tiles processed; input raster has degenerate dimensions")

    # Pass 2 (Merge): Reconcile tile boundaries on host.
    t1 = time.perf_counter()
    corrected = merge_fn(intermediate, tile_bounds_list)
    elapsed_pass2 = time.perf_counter() - t1

    elapsed = time.perf_counter() - t0

    result = from_numpy(
        corrected,
        nodata=result_nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_tiled_multipass tiles={tiles_processed} "
                f"tile_shape=({tile_h},{tile_w}) halo={halo} "
                f"raster_shape={raster.shape} "
                f"pass1={elapsed_pass1:.4f}s pass2={elapsed_pass2:.4f}s "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result
