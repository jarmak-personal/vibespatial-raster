"""Euclidean Distance Transform via Jump Flooding Algorithm.

CPU baseline uses scipy.ndimage.distance_transform_edt. GPU path uses custom
NVRTC kernels implementing the Jump Flooding Algorithm (JFA) for O(log N)
parallel distance computation.

The EDT computes, for each background (zero/False) pixel, the Euclidean
distance to the nearest foreground (nonzero/True) pixel. Foreground pixels
have distance 0. Nodata pixels propagate as nodata in the output.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(raster: OwnedRasterArray, threshold: int = 100_000) -> bool:
    """Auto-dispatch heuristic: use GPU when available and image is large enough."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# CPU baseline: Euclidean Distance Transform
# ---------------------------------------------------------------------------


def _distance_transform_cpu(raster: OwnedRasterArray) -> OwnedRasterArray:
    """CPU Euclidean Distance Transform via scipy.ndimage."""
    from scipy.ndimage import distance_transform_edt

    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("distance transform requires a single-band raster")
        data = data[0]

    # Build foreground mask (nonzero and non-nodata = foreground)
    foreground = data != 0
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            foreground &= ~np.isnan(data)
        else:
            foreground &= data != raster.nodata

    # Build nodata mask
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            nodata_mask = np.isnan(data)
        else:
            nodata_mask = data == raster.nodata
    else:
        nodata_mask = None

    # scipy distance_transform_edt(input): for each nonzero element,
    # returns the distance to the nearest zero-valued element.
    # We want: distance from each background pixel to nearest foreground.
    # Pass ~foreground so background=1, foreground=0. EDT then gives
    # distance from each nonzero (background) to nearest zero (foreground).
    # Foreground pixels are zero in the input so get distance 0.
    #
    # Edge case: if no foreground exists, return all zeros (no reference
    # point, consistent with scipy EDT of all-zeros).
    if not foreground.any():
        distances = np.zeros(data.shape, dtype=np.float64)
    else:
        edt_input = (~foreground).astype(np.float64)
        distances = distance_transform_edt(edt_input)

    # Apply nodata mask
    output_nodata = np.nan
    if nodata_mask is not None and nodata_mask.any():
        distances[nodata_mask] = output_nodata

    result = from_numpy(
        distances.astype(np.float64),
        nodata=output_nodata if (nodata_mask is not None and nodata_mask.any()) else None,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"distance_transform_cpu pixels={raster.pixel_count}",
            residency=result.residency,
        )
    )
    return result


# ---------------------------------------------------------------------------
# GPU: Euclidean Distance Transform via Jump Flooding Algorithm
# ---------------------------------------------------------------------------


def _distance_transform_gpu(raster: OwnedRasterArray) -> OwnedRasterArray:
    """GPU Euclidean Distance Transform using Jump Flooding Algorithm.

    Three-phase NVRTC kernel pipeline:
    1. jfa_init: seed buffer initialization
    2. jfa_step: iterative jump flooding (log2(N) passes, ping-pong buffers)
    3. distance_compute: convert seed coordinates to Euclidean distances

    All processing stays on device. Only transfers: H->D at start, D->H at end.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.distance import (
        DISTANCE_COMPUTE_NAMES,
        DISTANCE_COMPUTE_SOURCE,
        JFA_INIT_NAMES,
        JFA_INIT_SOURCE,
        JFA_STEP_NAMES,
        JFA_STEP_SOURCE,
    )

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Prepare data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("distance transform requires a single-band raster")
        data = data[0]

    height, width = data.shape
    n = height * width

    # Build foreground mask on host, then transfer once
    foreground_host = (data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            foreground_host &= (~np.isnan(data)).astype(np.uint8)
        else:
            foreground_host &= (data != raster.nodata).astype(np.uint8)

    # Build nodata mask on host
    has_nodata = raster.nodata is not None
    if has_nodata:
        if np.isnan(raster.nodata):
            nodata_mask_host = np.isnan(data).astype(np.uint8)
        else:
            nodata_mask_host = (data == raster.nodata).astype(np.uint8)
        any_nodata = nodata_mask_host.any()
    else:
        any_nodata = False

    # H->D transfer (only transfers at start)
    d_foreground = cp.asarray(np.ascontiguousarray(foreground_host.ravel()))

    if any_nodata:
        d_nodata_mask = cp.asarray(np.ascontiguousarray(nodata_mask_host.ravel()))
    else:
        d_nodata_mask = None

    # Allocate device buffers: two seed buffers for ping-pong (SoA layout)
    d_seed_x_a = cp.empty(n, dtype=np.int32)
    d_seed_y_a = cp.empty(n, dtype=np.int32)
    d_seed_x_b = cp.empty(n, dtype=np.int32)
    d_seed_y_b = cp.empty(n, dtype=np.int32)
    d_distance = cp.empty(n, dtype=np.float64)

    # --- Compile kernels ---
    init_key = make_kernel_cache_key("jfa_init", JFA_INIT_SOURCE)
    init_kernels = runtime.compile_kernels(
        cache_key=init_key,
        source=JFA_INIT_SOURCE,
        kernel_names=JFA_INIT_NAMES,
    )

    step_key = make_kernel_cache_key("jfa_step", JFA_STEP_SOURCE)
    step_kernels = runtime.compile_kernels(
        cache_key=step_key,
        source=JFA_STEP_SOURCE,
        kernel_names=JFA_STEP_NAMES,
    )

    dist_key = make_kernel_cache_key("distance_compute", DISTANCE_COMPUTE_SOURCE)
    dist_kernels = runtime.compile_kernels(
        cache_key=dist_key,
        source=DISTANCE_COMPUTE_SOURCE,
        kernel_names=DISTANCE_COMPUTE_NAMES,
    )

    # --- Phase 1: Init seeds ---
    grid_1d, block_1d = runtime.launch_config(init_kernels["jfa_init"], n)

    runtime.launch(
        kernel=init_kernels["jfa_init"],
        grid=grid_1d,
        block=block_1d,
        params=(
            (d_foreground.data.ptr, d_seed_x_a.data.ptr, d_seed_y_a.data.ptr, width, height),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # --- Phase 2: JFA iterations (ping-pong between A and B buffers) ---
    # Step sizes: k = N/2, N/4, ..., 2, 1 where N = next_pow2(max(H, W))
    max_dim = max(height, width)
    N = _next_power_of_2(max_dim)

    block_2d = (16, 16, 1)
    grid_2d = ((width + 15) // 16, (height + 15) // 16, 1)

    # Current input/output seed buffers
    cur_sx, cur_sy = d_seed_x_a, d_seed_y_a
    out_sx, out_sy = d_seed_x_b, d_seed_y_b

    step_k = N // 2
    jfa_iterations = 0
    while step_k >= 1:
        runtime.launch(
            kernel=step_kernels["jfa_step"],
            grid=grid_2d,
            block=block_2d,
            params=(
                (
                    cur_sx.data.ptr,
                    cur_sy.data.ptr,
                    out_sx.data.ptr,
                    out_sy.data.ptr,
                    width,
                    height,
                    step_k,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        # Ping-pong swap
        cur_sx, out_sx = out_sx, cur_sx
        cur_sy, out_sy = out_sy, cur_sy
        step_k //= 2
        jfa_iterations += 1

    # JFA+2 refinement: two extra passes at step_k=2 and step_k=1
    # to fix approximation artifacts at Voronoi boundaries.
    for refine_k in (2, 1):
        runtime.launch(
            kernel=step_kernels["jfa_step"],
            grid=grid_2d,
            block=block_2d,
            params=(
                (
                    cur_sx.data.ptr,
                    cur_sy.data.ptr,
                    out_sx.data.ptr,
                    out_sy.data.ptr,
                    width,
                    height,
                    refine_k,
                ),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )
        cur_sx, out_sx = out_sx, cur_sx
        cur_sy, out_sy = out_sy, cur_sy
        jfa_iterations += 1

    # After the loop, cur_sx/cur_sy hold the final seed coordinates

    # --- Phase 3: Compute distances ---
    grid_dist, block_dist = runtime.launch_config(dist_kernels["distance_compute"], n)

    nodata_value = np.nan
    nodata_mask_ptr = d_nodata_mask.data.ptr if d_nodata_mask is not None else 0

    runtime.launch(
        kernel=dist_kernels["distance_compute"],
        grid=grid_dist,
        block=block_dist,
        params=(
            (
                cur_sx.data.ptr,
                cur_sy.data.ptr,
                d_distance.data.ptr,
                nodata_mask_ptr,
                nodata_value,
                width,
                height,
            ),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_F64,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # --- D->H transfer (final) ---
    host_distance = cp.asnumpy(d_distance).reshape(height, width)

    elapsed = time.perf_counter() - t0

    result = from_numpy(
        host_distance,
        nodata=np.nan if any_nodata else None,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"distance_transform_gpu jfa_iterations={jfa_iterations} "
                f"pixels={n} grid_2d={grid_2d} elapsed={elapsed:.3f}s"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    logger.debug(
        "distance_transform_gpu iterations=%d pixels=%d elapsed=%.4fs",
        jfa_iterations,
        n,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Public API: dispatcher (GPU/CPU auto-selection)
# ---------------------------------------------------------------------------


def raster_distance_transform(
    raster: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute Euclidean Distance Transform of a raster.

    For each background (zero/False) pixel, computes the Euclidean distance
    (in pixel units) to the nearest foreground (nonzero/True) pixel.
    Foreground pixels have distance 0. Nodata pixels propagate as NaN.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster. Nonzero (and non-nodata) values are foreground.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when CuPy is available and pixel count exceeds
        the internal threshold.

    Returns
    -------
    OwnedRasterArray
        Float64 raster of Euclidean distances. Foreground pixels = 0.0,
        nodata pixels = NaN (if input has nodata).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return _distance_transform_gpu(raster)
    else:
        return _distance_transform_cpu(raster)
