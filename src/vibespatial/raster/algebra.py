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


def _gpu_convolve(raster: OwnedRasterArray, kernel_weights: np.ndarray) -> OwnedRasterArray:
    """Run a 2D convolution on GPU via NVRTC kernel."""
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import CONVOLVE_NORMALIZED_KERNEL_SOURCE

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

    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)

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
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,  # width, height
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,  # kw, kh
            KERNEL_PARAM_I32,
            KERNEL_PARAM_I32,  # pad_x, pad_y
            KERNEL_PARAM_F64,  # nodata_val
        ),
    )

    runtime.launch(
        kernel=kernels["convolve_normalized"],
        grid=grid,
        block=block,
        params=params,
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
            detail=f"gpu_convolve {width}x{height} kernel={kw}x{kh}",
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


def raster_slope(dem: OwnedRasterArray) -> OwnedRasterArray:
    """Compute slope (degrees) from a DEM raster on GPU.

    Uses a 3x3 Horn method kernel.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    """
    import cupy as cp

    orig_dtype = dem.dtype
    data = dem.to_numpy().astype(np.float64)
    if data.ndim == 3:
        data = data[0]

    d = cp.asarray(data)
    height, width = d.shape

    # Pad with edge replication
    padded = cp.pad(d, 1, mode="edge")

    # Horn method partial derivatives
    dz_dx = (
        (padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[0:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / 8.0

    dz_dy = (
        (padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    ) / 8.0

    # Account for pixel size from affine
    cell_x = abs(dem.affine[0])
    cell_y = abs(dem.affine[4])
    if cell_x > 0:
        dz_dx /= cell_x
    if cell_y > 0:
        dz_dy /= cell_y

    slope_rad = cp.arctan(cp.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = cp.degrees(slope_rad)

    host_result = cp.asnumpy(slope_deg)

    # Propagate nodata
    if dem.nodata is not None:
        nodata_mask = dem.nodata_mask
        host_result[nodata_mask] = dem.nodata

    # Restore original dtype for float inputs (float32 in -> float32 out)
    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        host_result = host_result.astype(orig_dtype)

    return from_numpy(
        host_result,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )


def raster_aspect(dem: OwnedRasterArray) -> OwnedRasterArray:
    """Compute aspect (degrees, 0=north, clockwise) from a DEM raster on GPU.

    Parameters
    ----------
    dem : OwnedRasterArray
        Digital Elevation Model raster.
    """
    import cupy as cp

    orig_dtype = dem.dtype
    data = dem.to_numpy().astype(np.float64)
    if data.ndim == 3:
        data = data[0]

    d = cp.asarray(data)
    padded = cp.pad(d, 1, mode="edge")

    dz_dx = (
        (padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[0:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / 8.0

    dz_dy = (
        (padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
        - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
    ) / 8.0

    aspect_rad = cp.arctan2(-dz_dy, dz_dx)
    aspect_deg = cp.degrees(aspect_rad)
    # Convert from math angle (east=0, CCW) to compass (north=0, CW)
    aspect_deg = (90.0 - aspect_deg) % 360.0

    host_result = cp.asnumpy(aspect_deg)

    if dem.nodata is not None:
        nodata_mask = dem.nodata_mask
        host_result[nodata_mask] = dem.nodata

    # Restore original dtype for float inputs (float32 in -> float32 out)
    if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
        host_result = host_result.astype(orig_dtype)

    return from_numpy(
        host_result,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )


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
