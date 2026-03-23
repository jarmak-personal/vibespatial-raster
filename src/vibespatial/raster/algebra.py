"""GPU raster algebra: local and focal operations.

Local operations use CuPy element-wise broadcasting.
Focal operations use custom NVRTC shared-memory tiled stencil kernels.
Fused expressions use custom NVRTC kernels compiled at runtime from
user-supplied expression strings (raster_expression).

ADR-0039: GPU Raster Algebra Dispatch
"""

from __future__ import annotations

import re
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
# Fused element-wise expression (NVRTC kernel)
# ---------------------------------------------------------------------------

# Expression validation: allowed tokens are variable names, numbers, operators,
# parentheses, commas, whitespace, and whitelisted function names.
_EXPR_TOKEN_RE = re.compile(
    r"""
    [a-h](?![a-zA-Z_0-9])      # single-letter variable (a-h), not part of longer identifier
    | \d+\.?\d*(?:[eE][+-]?\d+)? # numeric literal (int or float with optional exponent)
    | \.\d+(?:[eE][+-]?\d+)?     # numeric literal starting with dot
    | [+\-*/(),\s]               # operators, parens, comma, whitespace
    | abs|sqrt|pow|min|max       # allowed math functions
    | clamp                      # clamp(x, lo, hi) -> fmin(fmax(x, lo), hi)
    | log|log2|log10|exp         # transcendental functions
    | sin|cos|tan                # trig functions
    | floor|ceil                 # rounding functions
    """,
    re.VERBOSE,
)


def _validate_expression(expression: str, var_names: tuple[str, ...]) -> str:
    """Validate and translate expression to CUDA-compatible C code.

    Raises ValueError on invalid tokens or references to undefined variables.
    Returns the C expression string.
    """
    if not expression or not expression.strip():
        raise ValueError("expression must not be empty")

    # Check for disallowed tokens by removing all valid ones
    cleaned = _EXPR_TOKEN_RE.sub("", expression)
    if cleaned.strip():
        raise ValueError(
            f"expression contains invalid tokens: {cleaned.strip()!r}. "
            f"Allowed: variables {var_names}, numbers, +, -, *, /, "
            f"parentheses, and functions (abs, sqrt, pow, min, max, clamp, "
            f"log, log2, log10, exp, sin, cos, tan, floor, ceil)"
        )

    # Check that all single-letter identifiers reference defined variables
    found_vars = set(re.findall(r"\b([a-h])\b", expression))
    undefined = found_vars - set(var_names)
    if undefined:
        raise ValueError(
            f"expression references undefined variables: {sorted(undefined)}. "
            f"Defined: {list(var_names)}"
        )

    return expression


def _translate_expression(expression: str, dtype_name: str) -> str:
    """Translate high-level function names to CUDA math builtins.

    Handles clamp(x, lo, hi) -> fmin(fmax(x, lo), hi) and maps
    function names to their dtype-appropriate CUDA equivalents.
    """
    from vibespatial.raster.kernels.algebra import (
        ALLOWED_FUNCTIONS,
        ALLOWED_FUNCTIONS_F32,
    )

    func_map = ALLOWED_FUNCTIONS_F32 if dtype_name == "float" else ALLOWED_FUNCTIONS

    result = expression

    # Handle clamp(x, lo, hi) -> fmin(fmax(x, lo), hi)
    # Simple non-nested replacement (regex handles one level of nesting)
    result = re.sub(
        r"\bclamp\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)",
        lambda m: f"{func_map['min']}({func_map['max']}({m.group(1)}, {m.group(2)}), {m.group(3)})",
        result,
    )

    # Map remaining function names
    for py_name, cuda_name in func_map.items():
        result = re.sub(rf"\b{py_name}\b", cuda_name, result)

    return result


def _should_use_gpu_expr(*rasters: OwnedRasterArray) -> bool:
    """Auto-dispatch heuristic for raster_expression."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        if not runtime.available():
            return False
        return any(r.pixel_count >= 100_000 for r in rasters)
    except (ImportError, RuntimeError):
        return False


def _raster_expression_gpu(
    expression: str,
    rasters: dict[str, OwnedRasterArray],
    nodata: float | int | None,
    out_dtype: np.dtype,
) -> OwnedRasterArray:
    """GPU path: compile and launch fused expression kernel via NVRTC."""
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.algebra import build_expression_kernel_source

    t0 = time.perf_counter()

    # Determine compute dtype
    if np.issubdtype(out_dtype, np.floating) and out_dtype.itemsize <= 4:
        dtype_name = "float"
        compute_dtype = np.float32
    else:
        dtype_name = "double"
        compute_dtype = np.float64

    # Ordered variable names
    var_names = tuple(sorted(rasters.keys()))
    first_raster = rasters[var_names[0]]
    shape = first_raster.shape
    # Flatten to 2D if 3D
    flat_shape = shape if len(shape) == 2 else shape[-2:]
    n_pixels = int(np.prod(flat_shape))

    # Move all rasters to device and get pointers
    input_ptrs = []
    mask_ptrs = []
    # Keep references alive for the duration of the kernel launch
    _device_refs = []

    for name in var_names:
        r = rasters[name]
        r.move_to(
            Residency.DEVICE,
            trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
            reason="raster_expression requires device-resident data",
        )
        d = r.device_data()
        if d.ndim == 3:
            d = d[0]
        d = d.astype(compute_dtype)
        _device_refs.append(d)
        input_ptrs.append(d.data.ptr)

        mask = r.device_nodata_mask()
        if mask is not None and mask.any():
            m = mask.astype(cp.uint8)
            if m.ndim == 3:
                m = m[0]
            _device_refs.append(m)
            mask_ptrs.append(m.data.ptr)
        else:
            mask_ptrs.append(0)  # nullptr

    # Allocate output
    d_output = cp.empty(flat_shape, dtype=compute_dtype)
    d_output_mask = cp.zeros(flat_shape, dtype=cp.uint8)

    # Nodata value for the kernel
    nodata_val = float(nodata) if nodata is not None else 0.0

    # Translate expression to CUDA
    c_expression = _translate_expression(expression, dtype_name)

    # Build and compile kernel
    source = build_expression_kernel_source(c_expression, var_names, dtype_name)
    # Cache key includes the expression hash so different expressions get different kernels
    cache_key = make_kernel_cache_key(f"raster_expr_{dtype_name}", source)

    runtime = get_cuda_runtime()
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=("raster_expression",),
    )

    # Build parameter tuple: input_ptrs + mask_ptrs + output + output_mask + nodata_val + n
    # The kernel expects {dtype} nodata_val. For float32 kernels we need c_float since
    # KERNEL_PARAM_F32 is not defined in cuda_runtime.
    import ctypes

    from vibespatial.cuda_runtime import KERNEL_PARAM_F64

    if dtype_name == "float":
        nodata_param_type = ctypes.c_float
    else:
        nodata_param_type = KERNEL_PARAM_F64

    param_values = (
        tuple(input_ptrs)
        + tuple(mask_ptrs)
        + (
            d_output.data.ptr,
            d_output_mask.data.ptr,
            nodata_val,
            n_pixels,
        )
    )
    param_types = (
        tuple(KERNEL_PARAM_PTR for _ in var_names)  # input ptrs
        + tuple(KERNEL_PARAM_PTR for _ in var_names)  # mask ptrs
        + (
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # output_nodata_mask
            nodata_param_type,  # nodata_val
            KERNEL_PARAM_I32,  # n
        )
    )
    params = (param_values, param_types)

    # Occupancy-based launch config
    grid, block = runtime.launch_config(kernels["raster_expression"], n_pixels)

    runtime.launch(
        kernel=kernels["raster_expression"],
        grid=grid,
        block=block,
        params=params,
    )

    # Transfer result to host
    host_result = cp.asnumpy(d_output)
    host_mask = cp.asnumpy(d_output_mask)

    # Apply nodata where output mask is set
    if nodata is not None:
        host_result[host_mask.astype(bool)] = nodata

    elapsed = time.perf_counter() - t0

    result = from_numpy(
        host_result,
        nodata=nodata,
        affine=first_raster.affine,
        crs=first_raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_kernel=raster_expression pixels={n_pixels} "
                f"inputs={len(var_names)} expr={expression!r}"
            ),
            residency=Residency.HOST,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


def _raster_expression_cpu(
    expression: str,
    rasters: dict[str, OwnedRasterArray],
    nodata: float | int | None,
    out_dtype: np.dtype,
) -> OwnedRasterArray:
    """CPU fallback: evaluate expression using numpy."""
    var_names = tuple(sorted(rasters.keys()))
    first_raster = rasters[var_names[0]]
    shape = first_raster.shape
    flat_shape = shape if len(shape) == 2 else shape[-2:]

    # Determine compute dtype
    if np.issubdtype(out_dtype, np.floating) and out_dtype.itemsize <= 4:
        compute_dtype = np.float32
    else:
        compute_dtype = np.float64

    # Build numpy namespace for eval
    numpy_funcs = {
        "abs": np.abs,
        "sqrt": np.sqrt,
        "pow": np.power,
        "min": np.minimum,
        "max": np.maximum,
        "clamp": lambda x, lo, hi: np.clip(x, lo, hi),
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "floor": np.floor,
        "ceil": np.ceil,
    }

    # Prepare data arrays and combined nodata mask
    eval_ns = dict(numpy_funcs)
    combined_mask = np.zeros(flat_shape, dtype=bool)

    for name in var_names:
        r = rasters[name]
        data = r.to_numpy().astype(compute_dtype)
        if data.ndim == 3:
            data = data[0]
        eval_ns[name] = data
        if r.nodata is not None:
            combined_mask |= r.nodata_mask if r.nodata_mask.ndim == 2 else r.nodata_mask[0]

    # Evaluate with suppressed warnings for div-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = eval(expression, {"__builtins__": {}}, eval_ns)

    result = np.asarray(result, dtype=compute_dtype)
    if result.shape != flat_shape:
        result = np.broadcast_to(result, flat_shape).copy()

    # Replace inf/nan with nodata
    if nodata is not None:
        bad = np.logical_or(np.isinf(result), np.isnan(result))
        combined_mask |= bad
        result[combined_mask] = nodata

    return from_numpy(
        result,
        nodata=nodata,
        affine=first_raster.affine,
        crs=first_raster.crs,
    )


def raster_expression(
    expression: str,
    *,
    use_gpu: bool | None = None,
    **rasters: OwnedRasterArray,
) -> OwnedRasterArray:
    """Evaluate a fused element-wise expression over one or more rasters.

    Compiles the expression into a single NVRTC kernel at runtime so
    that multiple arithmetic operations execute in a single GPU pass
    with one read and one write to global memory.

    Parameters
    ----------
    expression : str
        Arithmetic expression using single-letter variable names that
        correspond to the keyword arguments.  Supported operators:
        ``+``, ``-``, ``*``, ``/``.  Supported functions: ``abs``,
        ``sqrt``, ``pow``, ``min``, ``max``, ``clamp``, ``log``,
        ``log2``, ``log10``, ``exp``, ``sin``, ``cos``, ``tan``,
        ``floor``, ``ceil``.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).
    **rasters : OwnedRasterArray
        Named input rasters.  Variable names must be single lowercase
        letters a-h.  All rasters must have the same shape.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same shape, affine, and CRS as the first
        input.  Output dtype is float64 unless all inputs are float32.

    Examples
    --------
    NDVI:

    >>> result = raster_expression("(a - b) / (a + b)", a=nir, b=red)

    Scaled difference with clamp:

    >>> result = raster_expression("clamp((a - b) * 2.0, 0.0, 1.0)", a=r1, b=r2)

    Raises
    ------
    ValueError
        If the expression is invalid, variable names are out of range,
        or raster shapes do not match.
    """
    if not rasters:
        raise ValueError("at least one input raster is required")

    # Validate variable names
    from vibespatial.raster.kernels.algebra import INPUT_VAR_NAMES

    for name in rasters:
        if name not in INPUT_VAR_NAMES:
            raise ValueError(
                f"variable name {name!r} is not valid; "
                f"use single letters: {INPUT_VAR_NAMES[: len(rasters)]}"
            )

    # Validate shapes match
    var_names = tuple(sorted(rasters.keys()))
    shapes = [r.shape for r in rasters.values()]
    ref_shape = shapes[0]
    for name, shape in zip(rasters.keys(), shapes):
        if shape != ref_shape:
            raise ValueError(
                f"all rasters must have the same shape: "
                f"{next(iter(rasters.keys()))} has {ref_shape}, "
                f"{name} has {shape}"
            )

    # Validate expression
    _validate_expression(expression, var_names)

    # Determine output dtype (float32 if all inputs are float32, else float64)
    dtypes = [r.dtype for r in rasters.values()]
    if all(np.issubdtype(d, np.floating) and np.dtype(d).itemsize <= 4 for d in dtypes):
        out_dtype = np.dtype(np.float32)
    else:
        out_dtype = np.dtype(np.float64)

    # Determine nodata
    nodata_vals = [r.nodata for r in rasters.values() if r.nodata is not None]
    nodata = nodata_vals[0] if nodata_vals else None

    # Dispatch
    if use_gpu is None:
        use_gpu = _should_use_gpu_expr(*rasters.values())

    if use_gpu:
        return _raster_expression_gpu(expression, rasters, nodata, out_dtype)
    else:
        return _raster_expression_cpu(expression, rasters, nodata, out_dtype)


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
