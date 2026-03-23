"""GPU-accelerated raster resampling/warping.

Resamples a raster from its native grid to a target grid defined by a
GridSpec (affine transform + width/height).  Each output pixel independently
maps back to the source via an inverse affine and samples the source raster.

Interpolation methods:
  - nearest:  snap to closest source pixel
  - bilinear: 2x2 weighted linear interpolation
  - bicubic:  4x4 Catmull-Rom spline interpolation

GPU path uses custom NVRTC kernels (one thread per output pixel,
grid-stride loop, occupancy-based launch config).  CPU fallback uses
numpy-based interpolation matching the same coordinate conventions.

All methods propagate nodata: if any contributing source pixel is nodata
or falls outside source bounds, the output pixel is set to nodata.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from vibespatial.raster.buffers import (
    GridSpec,
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)

logger = logging.getLogger(__name__)

_VALID_METHODS = ("nearest", "bilinear", "bicubic")


# ---------------------------------------------------------------------------
# Affine math helpers
# ---------------------------------------------------------------------------


def _invert_affine(
    affine: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Invert a 6-element GDAL-style affine transform.

    The affine maps pixel (col, row) to world (x, y):
        x = a * col + b * row + c
        y = d * col + e * row + f

    The inverse maps world (x, y) back to pixel (col, row).
    """
    a, b, c, d, e, f = affine
    det = a * e - b * d
    if abs(det) < 1e-15:
        raise ValueError("Affine transform is singular (determinant ~ 0)")
    inv_a = e / det
    inv_b = -b / det
    inv_c = (b * f - e * c) / det
    inv_d = -d / det
    inv_e = a / det
    inv_f = (d * c - a * f) / det
    return (inv_a, inv_b, inv_c, inv_d, inv_e, inv_f)


def _compose_pixel_to_pixel(
    src_affine: tuple[float, float, float, float, float, float],
    dst_affine: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Compute the composed transform: dst pixel -> world -> src pixel.

    The result maps destination pixel coordinates directly to source
    pixel coordinates, avoiding two separate transforms in the kernel.
    """
    # dst pixel -> world via dst_affine
    da, db, dc, dd, de, df = dst_affine
    # world -> src pixel via inverse of src_affine
    ia, ib, ic, id_, ie, if_ = _invert_affine(src_affine)

    # Compose: M_inv_src @ M_dst (2x3 matrix multiply)
    ca = ia * da + ib * dd
    cb = ia * db + ib * de
    cc = ia * dc + ib * df + ic
    cd = id_ * da + ie * dd
    ce = id_ * db + ie * de
    cf = id_ * dc + ie * df + if_
    return (ca, cb, cc, cd, ce, cf)


# ---------------------------------------------------------------------------
# GPU dispatch helpers
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(raster: OwnedRasterArray, threshold: int = 10_000) -> bool:
    """Auto-dispatch heuristic for resampling."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


def _dtype_to_kernel_param_type(dtype: np.dtype):
    """Return the kernel parameter type constant for the given numpy dtype."""
    from vibespatial.cuda_runtime import KERNEL_PARAM_F64, KERNEL_PARAM_I32

    # Map numpy dtypes to ctypes for the nodata_val kernel parameter.
    # Integer dtypes use c_int (KERNEL_PARAM_I32), floats use c_double
    # (KERNEL_PARAM_F64).
    if np.issubdtype(dtype, np.floating):
        return KERNEL_PARAM_F64
    elif dtype == np.uint8:
        # unsigned char fits in c_int for kernel param passing
        return KERNEL_PARAM_I32
    elif dtype == np.int16:
        return KERNEL_PARAM_I32
    else:
        return KERNEL_PARAM_F64  # fallback for upcast-to-f64 path


# ---------------------------------------------------------------------------
# GPU resample
# ---------------------------------------------------------------------------


def _resample_gpu(
    raster: OwnedRasterArray,
    target_grid: GridSpec,
    method: str,
) -> OwnedRasterArray:
    """GPU-accelerated resampling via NVRTC kernel."""
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.resample import get_resample_source
    from vibespatial.residency import Residency, TransferTrigger

    t0 = time.perf_counter()

    # Move source to device
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="raster_resample requires device-resident source data",
    )
    d_src = raster.device_data()

    # Ensure 2D (single-band)
    if d_src.ndim == 3:
        if d_src.shape[0] != 1:
            raise ValueError("raster_resample requires a single-band raster")
        d_src = d_src[0]

    src_height, src_width = d_src.shape
    dst_height = target_grid.height
    dst_width = target_grid.width

    # Determine kernel source for native dtype
    dtype_name = str(raster.dtype)
    kernel_source, c_type = get_resample_source(dtype_name)

    # If dtype is not directly supported, cast to float64 on device
    needs_cast = dtype_name not in ("float64", "float32", "int16", "uint8")
    if needs_cast:
        d_src = d_src.astype(cp.float64)
        working_dtype = np.dtype("float64")
    else:
        working_dtype = raster.dtype

    # Allocate output on device
    d_dst = cp.empty((dst_height, dst_width), dtype=working_dtype)

    # Nodata mask on device
    has_nodata = 1 if raster.nodata is not None else 0
    if has_nodata:
        d_nodata_mask = raster.device_nodata_mask()
        if d_nodata_mask.ndim == 3:
            d_nodata_mask = d_nodata_mask[0]
        d_nodata_mask = d_nodata_mask.astype(cp.uint8)
        nodata_ptr = d_nodata_mask.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    # Compute composed inverse affine: dst pixel -> src pixel
    inv = _compose_pixel_to_pixel(raster.affine, target_grid.affine)

    # Nodata value for the kernel (cast to working dtype)
    if raster.nodata is not None:
        nodata_val = working_dtype.type(raster.nodata)
    else:
        # Use 0 as placeholder when no nodata (never written since has_nodata=0)
        nodata_val = working_dtype.type(0)

    # Compile kernels
    runtime = get_cuda_runtime()
    kernel_name = f"resample_{method}"
    cache_key = make_kernel_cache_key(f"resample_{method}_{c_type}", kernel_source)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=kernel_source,
        kernel_names=(kernel_name,),
    )

    # Occupancy-based launch config (1D grid-stride loop)
    total_pixels = dst_height * dst_width
    grid, block = runtime.launch_config(kernels[kernel_name], total_pixels)

    # Determine parameter type for nodata_val
    nodata_param_type = _dtype_to_kernel_param_type(working_dtype)

    params = (
        (
            d_src.data.ptr,  # src
            d_dst.data.ptr,  # dst
            nodata_ptr,  # src_nodata_mask (nullable)
            src_width,  # src_width
            src_height,  # src_height
            dst_width,  # dst_width
            dst_height,  # dst_height
            inv[0],  # inv_a
            inv[1],  # inv_b
            inv[2],  # inv_c
            inv[3],  # inv_d
            inv[4],  # inv_e
            inv[5],  # inv_f
            nodata_val,  # nodata_val
            has_nodata,  # has_nodata
        ),
        (
            KERNEL_PARAM_PTR,  # src
            KERNEL_PARAM_PTR,  # dst
            KERNEL_PARAM_PTR,  # src_nodata_mask
            KERNEL_PARAM_I32,  # src_width
            KERNEL_PARAM_I32,  # src_height
            KERNEL_PARAM_I32,  # dst_width
            KERNEL_PARAM_I32,  # dst_height
            KERNEL_PARAM_F64,  # inv_a
            KERNEL_PARAM_F64,  # inv_b
            KERNEL_PARAM_F64,  # inv_c
            KERNEL_PARAM_F64,  # inv_d
            KERNEL_PARAM_F64,  # inv_e
            KERNEL_PARAM_F64,  # inv_f
            nodata_param_type,  # nodata_val
            KERNEL_PARAM_I32,  # has_nodata
        ),
    )

    runtime.launch(
        kernel=kernels[kernel_name],
        grid=grid,
        block=block,
        params=params,
    )

    # Transfer result to host and build OwnedRasterArray
    host_result = cp.asnumpy(d_dst)

    # If we upcast, convert back to original dtype
    if needs_cast and np.issubdtype(raster.dtype, np.integer):
        host_result = np.round(host_result).astype(raster.dtype)
    elif needs_cast:
        host_result = host_result.astype(raster.dtype)

    elapsed = time.perf_counter() - t0

    result = from_numpy(
        host_result,
        nodata=raster.nodata,
        affine=target_grid.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_resample_{method} "
                f"src={src_width}x{src_height} dst={dst_width}x{dst_height} "
                f"dtype={raster.dtype} blocks={grid[0]}"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


# ---------------------------------------------------------------------------
# CPU resample (scipy fallback)
# ---------------------------------------------------------------------------


def _resample_cpu(
    raster: OwnedRasterArray,
    target_grid: GridSpec,
    method: str,
) -> OwnedRasterArray:
    """CPU-based resampling via numpy (matches GPU kernel coordinate conventions).

    The coordinate system follows GDAL conventions:
    - Pixel (col, row) covers area [col, col+1) x [row, row+1)
    - Pixel center is at (col + 0.5, row + 0.5) in pixel coordinates
    - The composed inverse transform maps dst pixel centers to src pixel coords
    - For nearest: floor(src_coord) gives the source array index
    - For bilinear/bicubic: subtract 0.5 to get array-index-space fractional
      position, then interpolate
    """
    t0 = time.perf_counter()

    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("raster_resample requires a single-band raster")
        data = data[0]

    src_height, src_width = data.shape
    dst_height = target_grid.height
    dst_width = target_grid.width

    # Build coordinate arrays for every output pixel
    dst_rows, dst_cols = np.mgrid[0:dst_height, 0:dst_width]
    # Pixel centers in pixel coordinate space
    pc = dst_cols.astype(np.float64) + 0.5
    pr = dst_rows.astype(np.float64) + 0.5

    # Compose dst pixel -> src pixel (matches GPU kernel convention)
    inv = _compose_pixel_to_pixel(raster.affine, target_grid.affine)
    src_col_f = inv[0] * pc + inv[1] * pr + inv[2]
    src_row_f = inv[3] * pc + inv[4] * pr + inv[5]

    # Use float64 for interpolation
    src_f64 = data.astype(np.float64)

    nodata_val = raster.nodata
    has_nodata = nodata_val is not None
    nodata_mask_src = None
    if has_nodata:
        nodata_mask_src = raster.nodata_mask
        if nodata_mask_src.ndim == 3:
            nodata_mask_src = nodata_mask_src[0]

    # Initialize output with nodata fill
    fill = float(nodata_val) if has_nodata else 0.0
    result = np.full((dst_height, dst_width), fill, dtype=np.float64)

    if method == "nearest":
        # floor(src_coord) gives the source array index (same as GPU kernel)
        sc = np.floor(src_col_f).astype(np.intp)
        sr = np.floor(src_row_f).astype(np.intp)
        # In-bounds mask
        valid = (sc >= 0) & (sc < src_width) & (sr >= 0) & (sr < src_height)
        if has_nodata:
            sc_safe = np.clip(sc, 0, src_width - 1)
            sr_safe = np.clip(sr, 0, src_height - 1)
            valid = valid & ~nodata_mask_src[sr_safe, sc_safe]
        result[valid] = src_f64[sr[valid], sc[valid]]

    elif method == "bilinear":
        # Shift to array-index space where pixel center i is at i+0.5
        sx = src_col_f - 0.5
        sy = src_row_f - 0.5
        x0 = np.floor(sx).astype(np.intp)
        y0 = np.floor(sy).astype(np.intp)
        fx = sx - x0.astype(np.float64)
        fy = sy - y0.astype(np.float64)
        x1 = x0 + 1
        y1 = y0 + 1

        # All 4 pixels must be in bounds (matches GPU kernel)
        valid = (x0 >= 0) & (x1 < src_width) & (y0 >= 0) & (y1 < src_height)
        if has_nodata:
            x0s = np.clip(x0, 0, src_width - 1)
            x1s = np.clip(x1, 0, src_width - 1)
            y0s = np.clip(y0, 0, src_height - 1)
            y1s = np.clip(y1, 0, src_height - 1)
            nd = (
                nodata_mask_src[y0s, x0s]
                | nodata_mask_src[y0s, x1s]
                | nodata_mask_src[y1s, x0s]
                | nodata_mask_src[y1s, x1s]
            )
            valid = valid & ~nd

        v = valid
        v00 = src_f64[y0[v], x0[v]]
        v01 = src_f64[y0[v], x1[v]]
        v10 = src_f64[y1[v], x0[v]]
        v11 = src_f64[y1[v], x1[v]]
        fxv = fx[v]
        fyv = fy[v]
        result[v] = (
            v00 * (1.0 - fxv) * (1.0 - fyv)
            + v01 * fxv * (1.0 - fyv)
            + v10 * (1.0 - fxv) * fyv
            + v11 * fxv * fyv
        )

    else:  # bicubic
        sx = src_col_f - 0.5
        sy = src_row_f - 0.5
        x0 = np.floor(sx).astype(np.intp) - 1
        y0 = np.floor(sy).astype(np.intp) - 1
        fx = sx - np.floor(sx)
        fy = sy - np.floor(sy)

        # All 16 pixels must be in bounds (matches GPU kernel)
        valid = (x0 >= 0) & ((x0 + 3) < src_width) & (y0 >= 0) & ((y0 + 3) < src_height)
        if has_nodata:
            nd = np.zeros_like(valid)
            for jj in range(4):
                for ii in range(4):
                    cy = np.clip(y0 + jj, 0, src_height - 1)
                    cx = np.clip(x0 + ii, 0, src_width - 1)
                    nd |= nodata_mask_src[cy, cx]
            valid = valid & ~nd

        def _cubic_w(t):
            """Catmull-Rom cubic weight (vectorized)."""
            at = np.abs(t)
            return np.where(
                at <= 1.0,
                1.5 * at**3 - 2.5 * at**2 + 1.0,
                np.where(at < 2.0, -0.5 * at**3 + 2.5 * at**2 - 4.0 * at + 2.0, 0.0),
            )

        v = valid
        val = np.zeros(v.sum(), dtype=np.float64)
        fxv = fx[v]
        fyv = fy[v]
        for jj in range(4):
            wy = _cubic_w(fyv - (jj - 1))
            for ii in range(4):
                wx = _cubic_w(fxv - (ii - 1))
                cy = y0[v] + jj
                cx = x0[v] + ii
                val += src_f64[cy, cx] * wx * wy
        result[v] = val

    # Cast back to original dtype
    if np.issubdtype(raster.dtype, np.integer):
        result = np.round(result).astype(raster.dtype)
    elif raster.dtype != np.float64:
        result = result.astype(raster.dtype)

    elapsed = time.perf_counter() - t0

    out = from_numpy(
        result,
        nodata=raster.nodata,
        affine=target_grid.affine,
        crs=raster.crs,
    )
    out.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"cpu_resample_{method} "
                f"src={src_width}x{src_height} dst={dst_width}x{dst_height} "
                f"dtype={raster.dtype}"
            ),
            residency=out.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def raster_resample(
    raster: OwnedRasterArray,
    target_grid: GridSpec,
    method: str = "bilinear",
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Resample a raster to a new grid.

    Each output pixel independently samples the source raster via the
    inverse affine transform mapping target pixel coordinates to source
    pixel coordinates.

    Parameters
    ----------
    raster : OwnedRasterArray
        Source raster (single-band).
    target_grid : GridSpec
        Target grid defining the output affine, width, and height.
    method : str
        Interpolation method: ``'nearest'``, ``'bilinear'``, or
        ``'bicubic'``.  Default is ``'bilinear'``.
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-detect (None).

    Returns
    -------
    OwnedRasterArray
        Resampled raster on the target grid, HOST-resident.

    Raises
    ------
    ValueError
        If the method is not one of the supported interpolation methods,
        or if the raster is multi-band.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return _resample_gpu(raster, target_grid, method)
    else:
        return _resample_cpu(raster, target_grid, method)
