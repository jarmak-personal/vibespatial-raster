"""GPU raster algebra: local and focal operations.

Local operations use CuPy element-wise broadcasting.
Focal operations use custom NVRTC shared-memory tiled stencil kernels.

ADR-0039: GPU Raster Algebra Dispatch
"""

from __future__ import annotations

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)
from vibespatial.residency import Residency, TransferTrigger

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


def _to_device_data(raster: OwnedRasterArray):
    """Ensure raster is device-resident and return device data."""
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="raster algebra requires device-resident data",
    )
    return raster.device_data()


def _binary_op(a: OwnedRasterArray, b: OwnedRasterArray, op_name: str, op_func):
    """Apply a binary element-wise operation on two rasters."""
    if a.shape != b.shape:
        raise ValueError(f"raster shapes must match for {op_name}: {a.shape} vs {b.shape}")

    import cupy as cp

    da = _to_device_data(a)
    db = _to_device_data(b)

    result_device = op_func(da, db)

    # Nodata propagation: if either input is nodata, output is nodata
    nodata = a.nodata if a.nodata is not None else b.nodata
    if nodata is not None:
        mask_a = a.device_nodata_mask()
        mask_b = b.device_nodata_mask()
        combined_mask = cp.logical_or(mask_a, mask_b)
        if combined_mask.any():
            result_device = cp.where(combined_mask, nodata, result_device)

    # Build result as HOST with device state already populated
    host_data = cp.asnumpy(result_device)
    result = from_numpy(host_data, nodata=nodata, affine=a.affine, crs=a.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_{op_name} shape={a.shape} dtype={a.dtype}",
            residency=Residency.HOST,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Local raster algebra (element-wise via CuPy)
# ---------------------------------------------------------------------------


def raster_add(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise addition of two rasters."""
    import cupy as cp

    return _binary_op(a, b, "add", cp.add)


def raster_subtract(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise subtraction of two rasters."""
    import cupy as cp

    return _binary_op(a, b, "subtract", cp.subtract)


def raster_multiply(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise multiplication of two rasters."""
    import cupy as cp

    return _binary_op(a, b, "multiply", cp.multiply)


def raster_divide(a: OwnedRasterArray, b: OwnedRasterArray) -> OwnedRasterArray:
    """Element-wise division of two rasters. Division by zero yields nodata."""
    import cupy as cp

    def safe_divide(da, db):
        with np.errstate(divide="ignore", invalid="ignore"):
            result = cp.true_divide(da, db)
        # Replace inf/nan from div-by-zero with nodata
        nodata_val = (
            a.nodata if a.nodata is not None else (b.nodata if b.nodata is not None else 0.0)
        )
        bad = cp.logical_or(cp.isinf(result), cp.isnan(result))
        result = cp.where(bad, nodata_val, result)
        return result

    return _binary_op(a, b, "divide", safe_divide)


def raster_apply(
    raster: OwnedRasterArray,
    func,
    *,
    nodata: float | int | None = None,
) -> OwnedRasterArray:
    """Apply an arbitrary element-wise function to a raster on GPU.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    func : callable
        Function that accepts a CuPy array and returns a CuPy array.
    nodata : float | int | None
        Nodata value for the output. If None, inherits from input.
    """
    import cupy as cp

    d = _to_device_data(raster)
    result_device = func(d)

    out_nodata = nodata if nodata is not None else raster.nodata
    if out_nodata is not None and raster.nodata is not None:
        mask = raster.device_nodata_mask()
        if mask.any():
            result_device = cp.where(mask, out_nodata, result_device)

    host_data = cp.asnumpy(result_device)
    return from_numpy(host_data, nodata=out_nodata, affine=raster.affine, crs=raster.crs)


def raster_where(
    condition: OwnedRasterArray,
    true_val: OwnedRasterArray | float | int,
    false_val: OwnedRasterArray | float | int,
) -> OwnedRasterArray:
    """Element-wise conditional selection.

    Parameters
    ----------
    condition : OwnedRasterArray
        Boolean-like raster (nonzero = True).
    true_val, false_val : OwnedRasterArray or scalar
        Values to use where condition is True/False.
    """
    import cupy as cp

    cond_d = _to_device_data(condition)
    cond_bool = cond_d.astype(cp.bool_)

    if isinstance(true_val, OwnedRasterArray):
        tv = _to_device_data(true_val)
    else:
        tv = true_val

    if isinstance(false_val, OwnedRasterArray):
        fv = _to_device_data(false_val)
    else:
        fv = false_val

    result_device = cp.where(cond_bool, tv, fv)
    host_data = cp.asnumpy(result_device)

    nodata = condition.nodata
    return from_numpy(host_data, nodata=nodata, affine=condition.affine, crs=condition.crs)


def raster_classify(
    raster: OwnedRasterArray,
    bins: list[float],
    labels: list[int | float],
) -> OwnedRasterArray:
    """Reclassify raster values into discrete classes.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    bins : list[float]
        Bin edges (N edges define N-1 bins). Values below bins[0] get labels[0],
        values in [bins[i], bins[i+1]) get labels[i+1], etc.
    labels : list[int | float]
        Class labels. Must have len(bins) + 1 elements.
    """
    import cupy as cp

    if len(labels) != len(bins) + 1:
        raise ValueError(
            f"labels must have len(bins)+1={len(bins) + 1} elements, got {len(labels)}"
        )

    d = _to_device_data(raster)
    bins_d = cp.asarray(bins, dtype=d.dtype)
    labels_d = cp.asarray(labels, dtype=cp.float64)

    indices = cp.digitize(d.ravel(), bins_d).reshape(d.shape)
    result_device = labels_d[indices]

    # Preserve nodata
    if raster.nodata is not None:
        mask = raster.device_nodata_mask()
        if mask.any():
            result_device = cp.where(mask, raster.nodata, result_device)

    host_data = cp.asnumpy(result_device)
    return from_numpy(
        host_data.astype(np.float64),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


# ---------------------------------------------------------------------------
# Focal raster operations (NVRTC stencil kernels)
# ---------------------------------------------------------------------------

# Tile dimensions must match the #define TILE_W/TILE_H in kernel sources
_TILE_W = 16
_TILE_H = 16


def _convolve_shared_mem_bytes(kw: int, kh: int) -> int:
    """Calculate shared memory bytes needed for the tiled convolution kernel.

    Layout (contiguous in ``extern __shared__ char _smem[]``):
      data tile   : (TILE_H + 2*pad_y) * (TILE_W + 2*pad_x + 1) doubles  (+1 bank padding)
      kweights    : kh * kw doubles
      nodata tile : (TILE_H + 2*pad_y) * (TILE_W + 2*pad_x) uint8s
    """
    pad_x = kw // 2
    pad_y = kh // 2
    smem_cols = _TILE_W + 2 * pad_x + 1  # +1 for bank conflict avoidance
    tile_doubles = (_TILE_H + 2 * pad_y) * smem_cols
    kweights_doubles = kh * kw
    # nodata tile: 1 byte per element, no bank padding needed
    nodata_tile_bytes = (_TILE_H + 2 * pad_y) * (_TILE_W + 2 * pad_x)
    return (tile_doubles + kweights_doubles) * 8 + nodata_tile_bytes


def _gpu_convolve(raster: OwnedRasterArray, kernel_weights: np.ndarray) -> OwnedRasterArray:
    """Run a 2D convolution on GPU via shared-memory tiled NVRTC kernel.

    The kernel uses shared memory for both the input tile (with halo) and the
    kernel weights. Each thread block loads a TILE_WxTILE_H region plus
    pad_x/pad_y halo cells, then all threads read from shared memory for the
    convolution, achieving O(1) global memory reads per output pixel regardless
    of kernel size.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import CONVOLVE_NORMALIZED_KERNEL_SOURCE

    # Move to device and cast to float64 for computation
    d_data = _to_device_data(raster).astype(cp.float64)
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape
    kh, kw = kernel_weights.shape
    pad_y, pad_x = kh // 2, kw // 2

    d_input = d_data
    d_output = cp.zeros_like(d_input)
    d_kernel = cp.asarray(kernel_weights.astype(np.float64))

    nodata_val = float(raster.nodata) if raster.nodata is not None else 0.0

    if raster.nodata is not None:
        d_nodata = raster.device_nodata_mask().astype(cp.uint8)
        nodata_ptr = d_nodata.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("convolve_normalized", CONVOLVE_NORMALIZED_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=CONVOLVE_NORMALIZED_KERNEL_SOURCE,
        kernel_names=("convolve_normalized",),
    )

    # Block size is fixed at (_TILE_W, _TILE_H) because the kernel source
    # uses #define TILE_W/TILE_H and indexes shared memory relative to
    # threadIdx.  Validate via occupancy API that the hardware can schedule
    # this block size; fall back to _TILE_W * _TILE_H if the API is
    # unavailable.
    shared_mem_bytes = _convolve_shared_mem_bytes(kw, kh)
    optimal = runtime.optimal_block_size(
        kernels["convolve_normalized"], shared_mem_bytes=shared_mem_bytes
    )
    required = _TILE_W * _TILE_H
    if optimal < required:
        raise RuntimeError(
            f"Kernel requires block size {required} ({_TILE_W}x{_TILE_H}) but "
            f"occupancy API reports max {optimal} threads/block for "
            f"{shared_mem_bytes} bytes shared memory.  Reduce TILE_W/TILE_H "
            f"or kernel size (current: {kw}x{kh})."
        )

    block = (_TILE_W, _TILE_H, 1)
    grid = (
        (width + _TILE_W - 1) // _TILE_W,
        (height + _TILE_H - 1) // _TILE_H,
        1,
    )

    params = (
        (
            d_input.data.ptr,
            d_output.data.ptr,
            d_kernel.data.ptr,
            nodata_ptr,
            width,
            height,
            kw,
            kh,
            pad_x,
            pad_y,
            nodata_val,
        ),
        (
            KERNEL_PARAM_PTR,  # input
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # kernel_weights
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_I32,  # kw
            KERNEL_PARAM_I32,  # kh
            KERNEL_PARAM_I32,  # pad_x
            KERNEL_PARAM_I32,  # pad_y
            KERNEL_PARAM_F64,  # nodata_val
        ),
    )

    runtime.launch(
        kernel=kernels["convolve_normalized"],
        grid=grid,
        block=block,
        params=params,
        shared_mem_bytes=shared_mem_bytes,
    )

    host_result = cp.asnumpy(d_output)
    result = from_numpy(
        host_result,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(f"gpu_convolve {width}x{height} kernel={kw}x{kh} smem={shared_mem_bytes}B"),
            residency=Residency.HOST,
        )
    )
    return result


def raster_convolve(
    raster: OwnedRasterArray,
    kernel: np.ndarray,
) -> OwnedRasterArray:
    """Apply a 2D convolution kernel to a raster on GPU.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    kernel : np.ndarray
        2D convolution kernel (e.g., 3x3, 5x5).
    """
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 2:
        raise ValueError(f"kernel must be 2D, got {kernel.ndim}D")
    return _gpu_convolve(raster, kernel)


def raster_gaussian_filter(
    raster: OwnedRasterArray,
    sigma: float,
    *,
    kernel_size: int | None = None,
) -> OwnedRasterArray:
    """Apply a Gaussian filter to a raster on GPU.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    sigma : float
        Standard deviation of the Gaussian.
    kernel_size : int or None
        Size of the kernel. Default: 2 * ceil(3*sigma) + 1.
    """
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    ax = np.arange(kernel_size) - kernel_size // 2
    gauss_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_2d = np.outer(gauss_1d, gauss_1d)
    kernel_2d /= kernel_2d.sum()

    return _gpu_convolve(raster, kernel_2d)


# ---------------------------------------------------------------------------
# CPU slope/aspect via numpy Horn method (fallback)
# ---------------------------------------------------------------------------


def _cpu_slope_aspect(
    dem: OwnedRasterArray,
    *,
    compute_slope: bool,
    compute_aspect: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute slope and/or aspect on CPU using the Horn method with numpy.

    Uses np.gradient for central-difference computation, equivalent to
    a 3x3 Horn stencil. Nodata pixels are propagated: any pixel whose
    3x3 neighbourhood contains nodata receives nodata in the output.

    Returns (slope_host, aspect_host) numpy arrays, either may be None.
    """
    data = dem.to_numpy()
    if data.ndim == 3:
        data = data[0]

    data = data.astype(np.float64)
    height, width = data.shape

    cell_x = abs(dem.affine[0]) if dem.affine[0] != 0 else 1.0
    cell_y = abs(dem.affine[4]) if dem.affine[4] != 0 else 1.0

    nodata_val = float(dem.nodata) if dem.nodata is not None else None

    # Build nodata mask (True where nodata)
    if nodata_val is not None:
        nodata_mask = data == nodata_val
    else:
        nodata_mask = np.zeros_like(data, dtype=bool)

    # Pad with edge values for gradient computation
    padded = np.pad(data, 1, mode="edge")
    # Propagate nodata into padded array's mask
    nodata_padded = np.pad(nodata_mask, 1, mode="edge")

    # Horn method: compute dz/dx and dz/dy using 3x3 neighbourhood
    # dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_x)
    # dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_y)
    # where the 3x3 window is:
    #   a b c
    #   d e f
    #   g h i
    a = padded[0:-2, 0:-2]
    b = padded[0:-2, 1:-1]
    c = padded[0:-2, 2:]
    d = padded[1:-1, 0:-2]
    # e = padded[1:-1, 1:-1]  # center, not used in Horn gradients
    f = padded[1:-1, 2:]
    g = padded[2:, 0:-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_x)
    dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_y)

    # Any pixel whose 3x3 window touches nodata gets nodata in output
    nd_a = nodata_padded[0:-2, 0:-2]
    nd_b = nodata_padded[0:-2, 1:-1]
    nd_c = nodata_padded[0:-2, 2:]
    nd_d = nodata_padded[1:-1, 0:-2]
    nd_e = nodata_padded[1:-1, 1:-1]
    nd_f = nodata_padded[1:-1, 2:]
    nd_g = nodata_padded[2:, 0:-2]
    nd_h = nodata_padded[2:, 1:-1]
    nd_i = nodata_padded[2:, 2:]
    neighbourhood_nodata = nd_a | nd_b | nd_c | nd_d | nd_e | nd_f | nd_g | nd_h | nd_i

    slope_host = None
    aspect_host = None

    if compute_slope:
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)
        if nodata_val is not None:
            slope_deg[neighbourhood_nodata] = nodata_val
        slope_host = slope_deg

    if compute_aspect:
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        aspect_deg = np.degrees(aspect_rad)
        # Convert from math convention (0=east, CCW) to geographic (0=north, CW)
        aspect_deg = (90.0 - aspect_deg) % 360.0
        if nodata_val is not None:
            aspect_deg[neighbourhood_nodata] = nodata_val
        aspect_host = aspect_deg

    return slope_host, aspect_host


# ---------------------------------------------------------------------------
# Fused slope/aspect via NVRTC kernel (zero-copy, single-pass)
# ---------------------------------------------------------------------------


def _gpu_slope_aspect(
    dem: OwnedRasterArray,
    *,
    compute_slope: bool,
    compute_aspect: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run fused slope+aspect NVRTC kernel on device-resident DEM data.

    Computes Horn method gradient in a single pass using shared-memory tiling.
    No D->H->D round-trips. No cp.pad allocation. Nodata handled on device.

    Returns (slope_host, aspect_host) numpy arrays, either may be None if
    not requested.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import SLOPE_ASPECT_KERNEL_SOURCE

    # Keep data on device -- no to_numpy() round-trip
    d_data = _to_device_data(dem).astype(cp.float64)
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    # Allocate output buffers on device
    d_slope = cp.zeros_like(d_data) if compute_slope else cp.empty(1, dtype=cp.float64)
    d_aspect = cp.zeros_like(d_data) if compute_aspect else cp.empty(1, dtype=cp.float64)

    # Nodata mask on device (no host round-trip)
    nodata_val = float(dem.nodata) if dem.nodata is not None else 0.0
    if dem.nodata is not None:
        d_nodata = dem.device_nodata_mask().astype(cp.uint8)
        nodata_ptr = d_nodata.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    # Cell size from affine transform
    cell_x = abs(dem.affine[0]) if dem.affine[0] != 0 else 1.0
    cell_y = abs(dem.affine[4]) if dem.affine[4] != 0 else 1.0

    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("slope_aspect", SLOPE_ASPECT_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=SLOPE_ASPECT_KERNEL_SOURCE,
        kernel_names=("slope_aspect",),
    )

    # Block size is fixed at (_TILE_W, _TILE_H) because the kernel source
    # uses #define TILE_W/TILE_H and indexes shared memory relative to
    # threadIdx.  The slope/aspect kernel uses statically-sized shared
    # memory: __shared__ double tile[TILE_H+2][TILE_W+2+1].
    # Validate via occupancy API that the hardware can schedule this block.
    slope_smem = ((_TILE_H + 2) * (_TILE_W + 2 + 1)) * 8  # doubles
    optimal = runtime.optimal_block_size(kernels["slope_aspect"], shared_mem_bytes=slope_smem)
    required = _TILE_W * _TILE_H
    if optimal < required:
        raise RuntimeError(
            f"slope_aspect kernel requires block size {required} "
            f"({_TILE_W}x{_TILE_H}) but occupancy API reports max "
            f"{optimal} threads/block.  Reduce TILE_W/TILE_H."
        )

    block = (_TILE_W, _TILE_H, 1)
    grid = (
        (width + _TILE_W - 1) // _TILE_W,
        (height + _TILE_H - 1) // _TILE_H,
        1,
    )

    slope_ptr = d_slope.data.ptr if compute_slope else 0
    aspect_ptr = d_aspect.data.ptr if compute_aspect else 0

    params = (
        (
            d_data.data.ptr,
            slope_ptr,
            aspect_ptr,
            nodata_ptr,
            width,
            height,
            cell_x,
            cell_y,
            nodata_val,
            1 if compute_slope else 0,
            1 if compute_aspect else 0,
        ),
        (
            KERNEL_PARAM_PTR,  # dem
            KERNEL_PARAM_PTR,  # slope_out
            KERNEL_PARAM_PTR,  # aspect_out
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_F64,  # cell_x
            KERNEL_PARAM_F64,  # cell_y
            KERNEL_PARAM_F64,  # nodata_val
            KERNEL_PARAM_I32,  # compute_slope
            KERNEL_PARAM_I32,  # compute_aspect
        ),
    )

    runtime.launch(
        kernel=kernels["slope_aspect"],
        grid=grid,
        block=block,
        params=params,
    )

    # Transfer results to host only at the end
    slope_host = cp.asnumpy(d_slope) if compute_slope else None
    aspect_host = cp.asnumpy(d_aspect) if compute_aspect else None

    return slope_host, aspect_host


def raster_slope(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute slope (degrees) from a DEM raster.

    Uses a fused NVRTC kernel with shared-memory tiled 3x3 Horn method
    when GPU is available, or a numpy CPU fallback otherwise.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when CuPy is available.
    """
    if use_gpu is None:
        use_gpu = _has_cupy()

    orig_dtype = dem.dtype

    if use_gpu:
        slope_host, _ = _gpu_slope_aspect(dem, compute_slope=True, compute_aspect=False)
        backend = "gpu"
    else:
        slope_host, _ = _cpu_slope_aspect(dem, compute_slope=True, compute_aspect=False)
        backend = "cpu"

    # Restore original dtype for float inputs (float32 in -> float32 out)
    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        slope_host = slope_host.astype(orig_dtype)

    result = from_numpy(
        slope_host,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"{backend}_slope_fused {slope_host.shape[1]}x{slope_host.shape[0]}",
            residency=Residency.HOST,
        )
    )
    return result


def raster_aspect(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute aspect (degrees, 0=north, clockwise) from a DEM raster.

    Uses a fused NVRTC kernel with shared-memory tiled 3x3 Horn method
    when GPU is available, or a numpy CPU fallback otherwise.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when CuPy is available.
    """
    if use_gpu is None:
        use_gpu = _has_cupy()

    orig_dtype = dem.dtype

    if use_gpu:
        _, aspect_host = _gpu_slope_aspect(dem, compute_slope=False, compute_aspect=True)
        backend = "gpu"
    else:
        _, aspect_host = _cpu_slope_aspect(dem, compute_slope=False, compute_aspect=True)
        backend = "cpu"

    # Restore original dtype for float inputs (float32 in -> float32 out)
    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        aspect_host = aspect_host.astype(orig_dtype)

    result = from_numpy(
        aspect_host,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"{backend}_aspect_fused {aspect_host.shape[1]}x{aspect_host.shape[0]}",
            residency=Residency.HOST,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Hillshade (fused Horn-method slope/aspect + illumination)
# ---------------------------------------------------------------------------


def _should_use_gpu(raster: OwnedRasterArray) -> bool:
    """Auto-dispatch heuristic: use GPU if CuPy is available and raster is large enough."""
    try:
        import cupy as cp  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= 10_000
    except (ImportError, RuntimeError):
        return False


def _hillshade_cpu(
    dem: OwnedRasterArray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> OwnedRasterArray:
    """CPU hillshade using Horn method (numpy)."""
    data = dem.to_numpy().astype(np.float64)
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape

    # Pixel size from affine transform
    cell_x = abs(dem.affine[0]) if abs(dem.affine[0]) > 0 else 1.0
    cell_y = abs(dem.affine[4]) if abs(dem.affine[4]) > 0 else 1.0

    # Pad with edge replication for border handling
    padded = np.pad(data, 1, mode="edge")

    # Horn method partial derivatives
    dz_dx = (
        (
            (padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
            - (padded[0:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
        )
        / (8.0 * cell_x)
        * z_factor
    )

    dz_dy = (
        (
            (padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
            - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
        )
        / (8.0 * cell_y)
        * z_factor
    )

    # Slope and aspect
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)

    # Sun position in radians
    zenith_rad = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(azimuth)

    # Hillshade formula
    hs = np.cos(zenith_rad) * np.cos(slope) + np.sin(zenith_rad) * np.sin(slope) * np.cos(
        azimuth_rad - aspect
    )

    # Clamp and scale to uint8
    hs = np.clip(hs, 0.0, 1.0)
    result_data = (hs * 255.0 + 0.5).astype(np.uint8)

    # Nodata propagation: any 3x3 neighbor touching nodata -> nodata output
    nodata_out = 0
    if dem.nodata is not None:
        nd_mask = dem.nodata_mask
        if nd_mask.ndim == 3:
            nd_mask = nd_mask[0]
        # Dilate nodata mask by 1 pixel (any neighbor nodata -> output nodata)
        nd_padded = np.pad(nd_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
        nodata_expanded = np.zeros((height, width), dtype=bool)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nodata_expanded |= nd_padded[
                    1 + dy : height + 1 + dy, 1 + dx : width + 1 + dx
                ].astype(bool)
        result_data[nodata_expanded] = nodata_out

    return from_numpy(
        result_data,
        nodata=nodata_out if dem.nodata is not None else None,
        affine=dem.affine,
        crs=dem.crs,
    )


def _hillshade_gpu(
    dem: OwnedRasterArray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> OwnedRasterArray:
    """GPU hillshade using NVRTC shared-memory 3x3 stencil kernel."""
    import time as _time

    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import HILLSHADE_KERNEL_SOURCE

    # Move DEM to device
    dem.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="hillshade requires device-resident data",
    )

    d_data = dem.device_data()
    # Ensure fp64 for stencil precision (kernel expects double*)
    if d_data.dtype != cp.float64:
        d_data = d_data.astype(cp.float64)
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    # Pixel size from affine
    cell_x = abs(dem.affine[0]) if abs(dem.affine[0]) > 0 else 1.0
    cell_y = abs(dem.affine[4]) if abs(dem.affine[4]) > 0 else 1.0

    # Pre-convert angles to radians on host (avoids per-thread trig overhead)
    zenith_rad = float(np.radians(90.0 - altitude))
    azimuth_rad = float(np.radians(azimuth))

    # Nodata mask (device-side, nullable)
    nodata_out = np.uint8(0)
    if dem.nodata is not None:
        d_nodata = dem.device_nodata_mask()
        if d_nodata.ndim == 3:
            d_nodata = d_nodata[0]
        d_nodata_u8 = d_nodata.astype(cp.uint8)
        nodata_ptr = d_nodata_u8.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    # Allocate output on device
    d_output = cp.zeros((height, width), dtype=cp.uint8)

    # Compile kernel
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("hillshade", HILLSHADE_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=HILLSHADE_KERNEL_SOURCE,
        kernel_names=("hillshade",),
    )

    # 2D launch config: tile size must match kernel defines (TILE_W=16, TILE_H=16)
    tile_w, tile_h = 16, 16
    block = (tile_w, tile_h, 1)
    grid = ((width + tile_w - 1) // tile_w, (height + tile_h - 1) // tile_h, 1)

    params = (
        (
            d_data.data.ptr,
            d_output.data.ptr,
            nodata_ptr,
            width,
            height,
            float(cell_x),
            float(cell_y),
            float(z_factor),
            zenith_rad,
            azimuth_rad,
            int(nodata_out),
        ),
        (
            KERNEL_PARAM_PTR,  # input
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # nodata_mask (nullable)
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_F64,  # cell_x
            KERNEL_PARAM_F64,  # cell_y
            KERNEL_PARAM_F64,  # z_factor
            KERNEL_PARAM_F64,  # zenith_rad
            KERNEL_PARAM_F64,  # azimuth_rad
            KERNEL_PARAM_I32,  # nodata_out (uint8 passed as int)
        ),
    )

    # Shared memory: (TILE_H+2) * (TILE_W+2+1) * sizeof(double)
    smem_bytes = (tile_h + 2) * (tile_w + 2 + 1) * 8

    t0 = _time.perf_counter()
    runtime.launch(
        kernel=kernels["hillshade"],
        grid=grid,
        block=block,
        params=params,
        shared_mem_bytes=smem_bytes,
    )

    # D2H transfer (final result only -- zero-copy compliant)
    host_result = cp.asnumpy(d_output)
    elapsed = _time.perf_counter() - t0

    result = from_numpy(
        host_result,
        nodata=int(nodata_out) if dem.nodata is not None else None,
        affine=dem.affine,
        crs=dem.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_hillshade {width}x{height} blocks={grid[0]}x{grid[1]} smem={smem_bytes}B"
            ),
            residency=Residency.HOST,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


def raster_hillshade(
    dem: OwnedRasterArray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute hillshade from a DEM raster.

    Uses the Horn method to compute slope and aspect from a 3x3 neighborhood,
    then applies the standard hillshade illumination formula.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster (single-band).
    azimuth : float
        Direction of the light source in degrees (0=north, clockwise).
        Default 315 (northwest).
    altitude : float
        Altitude of the light source in degrees above the horizon.
        Default 45.
    z_factor : float
        Vertical exaggeration factor. Default 1.0.
    use_gpu : bool or None
        Force GPU (True) or CPU (False). None auto-detects.

    Returns
    -------
    OwnedRasterArray
        Hillshade raster with uint8 dtype (0-255). Nodata value is 0
        when the input has nodata, otherwise None.
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(dem)

    if use_gpu:
        return _hillshade_gpu(dem, azimuth=azimuth, altitude=altitude, z_factor=z_factor)
    else:
        return _hillshade_cpu(dem, azimuth=azimuth, altitude=altitude, z_factor=z_factor)


# ---------------------------------------------------------------------------
# Terrain derivatives: TRI, TPI, curvature
# ---------------------------------------------------------------------------

# Derivative type constants (must match DERIV_* in kernel source)
_DERIV_TRI = 0
_DERIV_TPI = 1
_DERIV_CURV = 2


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu_terrain(raster: OwnedRasterArray, threshold: int = 100_000) -> bool:
    """Auto-dispatch heuristic: use GPU when available and image is large enough."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


def _terrain_derivative_gpu(
    dem: OwnedRasterArray,
    deriv_type: int,
) -> OwnedRasterArray:
    """Run terrain derivative kernel on GPU via NVRTC.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    deriv_type : int
        0=TRI, 1=TPI, 2=curvature.
    """
    import time

    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import TERRAIN_DERIVATIVES_KERNEL_SOURCE

    t0 = time.perf_counter()

    # Move to device; work in float64 for accuracy
    dem.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="terrain derivative requires device-resident data",
    )
    d_data = dem.device_data().astype(cp.float64)
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    # Output buffer
    d_output = cp.empty((height, width), dtype=cp.float64)

    # Nodata
    nodata_val = float(dem.nodata) if dem.nodata is not None else -9999.0
    if dem.nodata is not None:
        d_nodata = dem.device_nodata_mask().astype(cp.uint8)
        nodata_ptr = d_nodata.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    # Cell sizes from affine
    cellsize_x = abs(dem.affine[0])
    cellsize_y = abs(dem.affine[4])
    if cellsize_x == 0.0:
        cellsize_x = 1.0
    if cellsize_y == 0.0:
        cellsize_y = 1.0

    # Compile kernel
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("terrain_derivatives", TERRAIN_DERIVATIVES_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=TERRAIN_DERIVATIVES_KERNEL_SOURCE,
        kernel_names=("terrain_derivatives",),
    )

    # 2D launch config — tile size matches kernel's TILE_W/TILE_H (16x16)
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)

    params = (
        (
            d_data.data.ptr,
            d_output.data.ptr,
            nodata_ptr,
            width,
            height,
            cellsize_x,
            cellsize_y,
            nodata_val,
            deriv_type,
        ),
        (
            KERNEL_PARAM_PTR,  # input
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_F64,  # cellsize_x
            KERNEL_PARAM_F64,  # cellsize_y
            KERNEL_PARAM_F64,  # nodata_val
            KERNEL_PARAM_I32,  # deriv_type
        ),
    )

    runtime.launch(
        kernel=kernels["terrain_derivatives"],
        grid=grid,
        block=block,
        params=params,
    )

    host_result = cp.asnumpy(d_output)
    elapsed = time.perf_counter() - t0

    deriv_names = {_DERIV_TRI: "TRI", _DERIV_TPI: "TPI", _DERIV_CURV: "curvature"}
    result = from_numpy(
        host_result,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_terrain_{deriv_names.get(deriv_type, 'unknown')} "
                f"{width}x{height} blocks={grid[0]}x{grid[1]}"
            ),
            residency=Residency.HOST,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


def _terrain_derivative_cpu(
    dem: OwnedRasterArray,
    deriv_type: int,
) -> OwnedRasterArray:
    """Compute terrain derivative on CPU using numpy 3x3 window operations.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    deriv_type : int
        0=TRI, 1=TPI, 2=curvature.
    """
    data = dem.to_numpy().astype(np.float64)
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape
    nodata_val = float(dem.nodata) if dem.nodata is not None else -9999.0

    # Pad with edge replication for boundary handling
    padded = np.pad(data, 1, mode="edge")

    # Extract 3x3 neighborhood elements (named per kernel convention)
    z0 = padded[0:-2, 0:-2]  # top-left
    z1 = padded[0:-2, 1:-1]  # top-center
    z2 = padded[0:-2, 2:]  # top-right
    z3 = padded[1:-1, 0:-2]  # mid-left
    z4 = padded[1:-1, 1:-1]  # center
    z5 = padded[1:-1, 2:]  # mid-right
    z6 = padded[2:, 0:-2]  # bot-left
    z7 = padded[2:, 1:-1]  # bot-center
    z8 = padded[2:, 2:]  # bot-right

    if deriv_type == _DERIV_TRI:
        # TRI: mean absolute difference between center and 8 neighbors
        result = (
            np.abs(z0 - z4)
            + np.abs(z1 - z4)
            + np.abs(z2 - z4)
            + np.abs(z3 - z4)
            + np.abs(z5 - z4)
            + np.abs(z6 - z4)
            + np.abs(z7 - z4)
            + np.abs(z8 - z4)
        ) / 8.0
    elif deriv_type == _DERIV_TPI:
        # TPI: center minus mean of 8 neighbors
        neighbor_mean = (z0 + z1 + z2 + z3 + z5 + z6 + z7 + z8) / 8.0
        result = z4 - neighbor_mean
    else:
        # Profile curvature — Zevenbergen & Thorne (1987)
        cellsize_x = abs(dem.affine[0])
        cellsize_y = abs(dem.affine[4])
        if cellsize_x == 0.0:
            cellsize_x = 1.0
        if cellsize_y == 0.0:
            cellsize_y = 1.0

        D = ((z3 + z5) / 2.0 - z4) / (cellsize_x * cellsize_x)
        E = ((z1 + z7) / 2.0 - z4) / (cellsize_y * cellsize_y)
        result = -2.0 * (D + E) * 100.0

    # Border pixels -> nodata (GPU kernel also marks them as nodata)
    output = np.full((height, width), nodata_val, dtype=np.float64)
    output[1:-1, 1:-1] = result[1:-1, 1:-1]

    # Nodata propagation: if center or any neighbor is nodata
    if dem.nodata is not None:
        nodata_mask = dem.nodata_mask
        if nodata_mask.ndim == 3:
            nodata_mask = nodata_mask[0]
        # Dilate the nodata mask by 1 pixel (any neighbor nodata -> output nodata)
        padded_mask = np.pad(nodata_mask, 1, mode="constant", constant_values=False)
        any_nodata = np.zeros((height, width), dtype=bool)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                any_nodata |= padded_mask[1 + dy : height + 1 + dy, 1 + dx : width + 1 + dx]
        output[any_nodata] = nodata_val

    return from_numpy(
        output,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )


def raster_tri(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute Terrain Ruggedness Index (TRI) from a DEM raster.

    TRI is the mean absolute difference between the center cell and its
    8 neighbors. Riley et al. (1999).

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-dispatch (None).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu_terrain(dem)

    if use_gpu:
        return _terrain_derivative_gpu(dem, _DERIV_TRI)
    else:
        return _terrain_derivative_cpu(dem, _DERIV_TRI)


def raster_tpi(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute Topographic Position Index (TPI) from a DEM raster.

    TPI is the center cell elevation minus the mean of its 8 neighbors.
    Positive values indicate ridges/hilltops, negative values indicate
    valleys.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-dispatch (None).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu_terrain(dem)

    if use_gpu:
        return _terrain_derivative_gpu(dem, _DERIV_TPI)
    else:
        return _terrain_derivative_cpu(dem, _DERIV_TPI)


def raster_curvature(
    dem: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute profile curvature from a DEM raster.

    Uses second-order finite differences from the 3x3 window per
    Zevenbergen & Thorne (1987). Positive curvature indicates concave
    surfaces, negative indicates convex.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-dispatch (None).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu_terrain(dem)

    if use_gpu:
        return _terrain_derivative_gpu(dem, _DERIV_CURV)
    else:
        return _terrain_derivative_cpu(dem, _DERIV_CURV)
