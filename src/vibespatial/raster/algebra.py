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
        nodata_val = a.nodata if a.nodata is not None else 0.0
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

    data = raster.to_numpy().astype(np.float64)
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape
    kh, kw = kernel_weights.shape
    pad_y, pad_x = kh // 2, kw // 2

    d_input = cp.asarray(data)
    d_output = cp.zeros_like(d_input)
    d_kernel = cp.asarray(kernel_weights.astype(np.float64))

    nodata_val = float(raster.nodata) if raster.nodata is not None else 0.0

    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            nodata_host = np.isnan(data).astype(np.uint8)
        else:
            nodata_host = (data == raster.nodata).astype(np.uint8)
        d_nodata = cp.asarray(nodata_host)
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

    return from_numpy(
        host_result,
        nodata=dem.nodata,
        affine=dem.affine,
        crs=dem.crs,
    )
