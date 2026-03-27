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
import warnings

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_device,
    from_numpy,
)
from vibespatial.raster.dispatch import dispatch_per_band_cpu, dispatch_per_band_gpu
from vibespatial.residency import Residency, TransferTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warn_multiband_squeeze(arr, *, stacklevel: int = 3):
    """Emit a UserWarning when silently discarding extra bands from a 3D array.

    Call this *before* squeezing ``arr = arr[0]``.  ``stacklevel`` should point
    the warning back to the caller's caller (the public dispatch function).
    Only warns when there are genuinely multiple bands (shape[0] > 1).
    """
    if arr.shape[0] > 1:
        warnings.warn(
            f"Multiband raster with {arr.shape[0]} bands received; "
            "only band 1 will be processed. Multiband support is planned.",
            UserWarning,
            stacklevel=stacklevel,
        )


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


def _should_use_gpu_algebra(
    *rasters: OwnedRasterArray,
    threshold: int = 100_000,
) -> bool:
    """Auto-dispatch heuristic for local algebra operations."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        if not runtime.available():
            return False
        return any(r.pixel_count >= threshold for r in rasters)
    except (ImportError, RuntimeError):
        return False


def _binary_op_gpu(a: OwnedRasterArray, b: OwnedRasterArray, op_name: str, op_func):
    """GPU path for binary element-wise operations using CuPy."""
    import cupy as cp

    t0 = time.perf_counter()

    da = _to_device_data(a)
    db = _to_device_data(b)

    result_device = op_func(da, db)

    # Determine the output nodata value
    nodata = a.nodata if a.nodata is not None else b.nodata

    # Build a single combined mask covering:
    #   1. Input nodata pixels (either input is nodata -> output is nodata)
    #   2. Inf/NaN produced by the operation (e.g., division by zero)
    # Applying this in one pass avoids double-masking and ensures that
    # legitimate computed values matching the nodata sentinel are preserved.
    bad_values = cp.logical_or(cp.isinf(result_device), cp.isnan(result_device))
    if nodata is not None:
        mask_a = a.device_nodata_mask()
        mask_b = b.device_nodata_mask()
        combined_mask = cp.logical_or(cp.logical_or(mask_a, mask_b), bad_values)
        result_device = cp.where(combined_mask, nodata, result_device)
    elif cp.any(bad_values):
        # No nodata sentinel available — use NaN for float, 0 for integer
        if cp.issubdtype(result_device.dtype, cp.floating):
            result_device = cp.where(bad_values, cp.nan, result_device)
            nodata = float("nan")
        else:
            result_device = cp.where(bad_values, 0, result_device)
            nodata = 0

    elapsed = time.perf_counter() - t0

    # Build result as DEVICE-resident (zero-copy, no D→H transfer)
    result = from_device(result_device, nodata=nodata, affine=a.affine, crs=a.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_{op_name} [GPU] shape={a.shape} dtype={a.dtype} elapsed={elapsed:.4f}s",
            residency=Residency.DEVICE,
        )
    )
    return result


def _binary_op_cpu(a: OwnedRasterArray, b: OwnedRasterArray, op_name: str, np_op_func):
    """CPU fallback for binary element-wise operations using numpy."""
    t0 = time.perf_counter()

    da = a.to_numpy()
    db = b.to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        result_data = np_op_func(da, db)

    # Determine the output nodata value
    nodata = a.nodata if a.nodata is not None else b.nodata

    # Build a single combined mask covering:
    #   1. Input nodata pixels (either input is nodata -> output is nodata)
    #   2. Inf/NaN produced by the operation (e.g., division by zero)
    bad_values = np.logical_or(np.isinf(result_data), np.isnan(result_data))
    if nodata is not None:
        mask_a = a.nodata_mask
        mask_b = b.nodata_mask
        combined_mask = np.logical_or(np.logical_or(mask_a, mask_b), bad_values)
        result_data = np.where(combined_mask, nodata, result_data)
    elif np.any(bad_values):
        # No nodata sentinel available — use NaN for float, 0 for integer
        if np.issubdtype(result_data.dtype, np.floating):
            result_data = np.where(bad_values, np.nan, result_data)
            nodata = float("nan")
        else:
            result_data = np.where(bad_values, 0, result_data)
            nodata = 0

    elapsed = time.perf_counter() - t0

    result = from_numpy(result_data, nodata=nodata, affine=a.affine, crs=a.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_{op_name} [CPU] shape={a.shape} dtype={a.dtype} elapsed={elapsed:.4f}s",
            residency=Residency.HOST,
        )
    )
    return result


def _binary_op(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    op_name: str,
    gpu_op_func,
    np_op_func,
    *,
    use_gpu: bool | None = None,
):
    """Apply a binary element-wise operation on two rasters.

    Dispatches to GPU or CPU path based on ``use_gpu``.  Nodata masking is
    applied in a single pass on both paths.  The ``op_func`` must **not**
    replace inf/nan with the nodata sentinel — that is handled uniformly
    so that legitimate computed values equal to the nodata sentinel are
    never spuriously masked.
    """
    if a.shape != b.shape:
        raise ValueError(f"raster shapes must match for {op_name}: {a.shape} vs {b.shape}")

    if use_gpu is None:
        use_gpu = _should_use_gpu_algebra(a, b)

    if use_gpu:
        return _binary_op_gpu(a, b, op_name, gpu_op_func)
    else:
        return _binary_op_cpu(a, b, op_name, np_op_func)


# ---------------------------------------------------------------------------
# Local raster algebra (element-wise via CuPy)
# ---------------------------------------------------------------------------


def raster_add(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Element-wise addition of two rasters."""

    def _gpu_add(da, db):
        import cupy as cp

        return cp.add(da, db)

    return _binary_op(a, b, "add", _gpu_add, np.add, use_gpu=use_gpu)


def raster_subtract(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Element-wise subtraction of two rasters."""

    def _gpu_subtract(da, db):
        import cupy as cp

        return cp.subtract(da, db)

    return _binary_op(a, b, "subtract", _gpu_subtract, np.subtract, use_gpu=use_gpu)


def raster_multiply(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Element-wise multiplication of two rasters."""

    def _gpu_multiply(da, db):
        import cupy as cp

        return cp.multiply(da, db)

    return _binary_op(a, b, "multiply", _gpu_multiply, np.multiply, use_gpu=use_gpu)


def raster_divide(
    a: OwnedRasterArray,
    b: OwnedRasterArray,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Element-wise division of two rasters. Division by zero yields nodata."""

    def _gpu_safe_divide(da, db):
        import cupy as cp

        return cp.true_divide(da, db)

    def _cpu_safe_divide(da, db):
        return np.true_divide(da, db)

    return _binary_op(a, b, "divide", _gpu_safe_divide, _cpu_safe_divide, use_gpu=use_gpu)


def _raster_apply_gpu(
    raster: OwnedRasterArray,
    func,
    *,
    nodata: float | int | None = None,
) -> OwnedRasterArray:
    """GPU path: apply an element-wise function using CuPy."""
    import cupy as cp

    t0 = time.perf_counter()

    d = _to_device_data(raster)
    result_device = func(d)

    out_nodata = nodata if nodata is not None else raster.nodata
    if out_nodata is not None and raster.nodata is not None:
        mask = raster.device_nodata_mask()
        if mask is not None:
            result_device = cp.where(mask, out_nodata, result_device)

    elapsed = time.perf_counter() - t0

    result = from_device(result_device, nodata=out_nodata, affine=raster.affine, crs=raster.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_apply [GPU] shape={raster.shape} dtype={raster.dtype} elapsed={elapsed:.4f}s",
            residency=Residency.DEVICE,
        )
    )
    return result


def _raster_apply_cpu(
    raster: OwnedRasterArray,
    func,
    *,
    nodata: float | int | None = None,
) -> OwnedRasterArray:
    """CPU fallback: apply an element-wise function using numpy."""
    t0 = time.perf_counter()

    data = raster.to_numpy()
    result_data = func(data)

    out_nodata = nodata if nodata is not None else raster.nodata
    if out_nodata is not None and raster.nodata is not None:
        mask = raster.nodata_mask
        result_data = np.where(mask, out_nodata, result_data)

    elapsed = time.perf_counter() - t0

    result = from_numpy(result_data, nodata=out_nodata, affine=raster.affine, crs=raster.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_apply [CPU] shape={raster.shape} dtype={raster.dtype} elapsed={elapsed:.4f}s",
            residency=Residency.HOST,
        )
    )
    return result


def raster_apply(
    raster: OwnedRasterArray,
    func,
    *,
    nodata: float | int | None = None,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Apply an arbitrary element-wise function to a raster.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster.
    func : callable
        Function that accepts a numpy or CuPy array and returns an array
        of the same type. When ``use_gpu=True`` this must accept CuPy
        arrays; when ``use_gpu=False`` it must accept numpy arrays.
    nodata : float | int | None
        Nodata value for the output. If None, inherits from input.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).
    """
    if use_gpu is None:
        # raster_apply uses a zero threshold because the caller controls the
        # function.  A CuPy callable (e.g. cp.sqrt) will fail on numpy arrays,
        # so we must route to GPU whenever CUDA is available regardless of
        # raster size.  The standard pixel-count threshold is designed for ops
        # where *we* control both paths -- it does not apply here.
        use_gpu = _should_use_gpu_algebra(raster, threshold=0)

    if use_gpu:
        return _raster_apply_gpu(raster, func, nodata=nodata)
    else:
        return _raster_apply_cpu(raster, func, nodata=nodata)


def _raster_where_gpu(
    condition: OwnedRasterArray,
    true_val: OwnedRasterArray | float | int,
    false_val: OwnedRasterArray | float | int,
) -> OwnedRasterArray:
    """GPU path: element-wise conditional selection using CuPy."""
    import cupy as cp

    t0 = time.perf_counter()

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

    # Propagate nodata from condition and raster operands
    nodata = condition.nodata
    if nodata is not None:
        mask = condition.device_nodata_mask()
        if mask is not None:
            result_device = cp.where(mask, nodata, result_device)
    if isinstance(true_val, OwnedRasterArray) and true_val.nodata is not None:
        tmask = true_val.device_nodata_mask()
        if tmask is not None and nodata is not None:
            result_device = cp.where(cp.logical_and(cond_bool, tmask), nodata, result_device)
    if isinstance(false_val, OwnedRasterArray) and false_val.nodata is not None:
        fmask = false_val.device_nodata_mask()
        if fmask is not None and nodata is not None:
            result_device = cp.where(cp.logical_and(~cond_bool, fmask), nodata, result_device)

    elapsed = time.perf_counter() - t0

    result = from_device(result_device, nodata=nodata, affine=condition.affine, crs=condition.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_where [GPU] shape={condition.shape} elapsed={elapsed:.4f}s",
            residency=Residency.DEVICE,
        )
    )
    return result


def _raster_where_cpu(
    condition: OwnedRasterArray,
    true_val: OwnedRasterArray | float | int,
    false_val: OwnedRasterArray | float | int,
) -> OwnedRasterArray:
    """CPU fallback: element-wise conditional selection using numpy."""
    t0 = time.perf_counter()

    cond_data = condition.to_numpy()
    cond_bool = cond_data.astype(bool)

    if isinstance(true_val, OwnedRasterArray):
        tv = true_val.to_numpy()
    else:
        tv = true_val

    if isinstance(false_val, OwnedRasterArray):
        fv = false_val.to_numpy()
    else:
        fv = false_val

    result_data = np.where(cond_bool, tv, fv)

    # Propagate nodata from condition and raster operands
    nodata = condition.nodata
    if nodata is not None:
        mask = condition.nodata_mask
        result_data = np.where(mask, nodata, result_data)
    if isinstance(true_val, OwnedRasterArray) and true_val.nodata is not None:
        tmask = true_val.nodata_mask
        if nodata is not None:
            result_data = np.where(np.logical_and(cond_bool, tmask), nodata, result_data)
    if isinstance(false_val, OwnedRasterArray) and false_val.nodata is not None:
        fmask = false_val.nodata_mask
        if nodata is not None:
            result_data = np.where(np.logical_and(~cond_bool, fmask), nodata, result_data)

    elapsed = time.perf_counter() - t0

    result = from_numpy(result_data, nodata=nodata, affine=condition.affine, crs=condition.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_where [CPU] shape={condition.shape} elapsed={elapsed:.4f}s",
            residency=Residency.HOST,
        )
    )
    return result


def raster_where(
    condition: OwnedRasterArray,
    true_val: OwnedRasterArray | float | int,
    false_val: OwnedRasterArray | float | int,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Element-wise conditional selection.

    Parameters
    ----------
    condition : OwnedRasterArray
        Boolean-like raster (nonzero = True).
    true_val, false_val : OwnedRasterArray or scalar
        Values to use where condition is True/False.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu_algebra(condition)

    if use_gpu:
        return _raster_where_gpu(condition, true_val, false_val)
    else:
        return _raster_where_cpu(condition, true_val, false_val)


def _raster_classify_gpu(
    raster: OwnedRasterArray,
    bins: list[float],
    labels: list[int | float],
) -> OwnedRasterArray:
    """GPU path: reclassify raster values using CuPy."""
    import cupy as cp

    t0 = time.perf_counter()

    d = _to_device_data(raster)
    bins_d = cp.asarray(bins, dtype=d.dtype)
    labels_d = cp.asarray(labels, dtype=cp.float64)

    indices = cp.digitize(d.ravel(), bins_d).reshape(d.shape)
    result_device = labels_d[indices]

    # Preserve nodata
    if raster.nodata is not None:
        mask = raster.device_nodata_mask()
        if mask is not None:
            result_device = cp.where(mask, raster.nodata, result_device)

    elapsed = time.perf_counter() - t0

    result = from_device(
        result_device.astype(cp.float64, copy=False),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_classify [GPU] shape={raster.shape} bins={len(bins)} elapsed={elapsed:.4f}s",
            residency=Residency.DEVICE,
        )
    )
    return result


def _raster_classify_cpu(
    raster: OwnedRasterArray,
    bins: list[float],
    labels: list[int | float],
) -> OwnedRasterArray:
    """CPU fallback: reclassify raster values using numpy."""
    t0 = time.perf_counter()

    data = raster.to_numpy()
    bins_arr = np.asarray(bins, dtype=data.dtype)
    labels_arr = np.asarray(labels, dtype=np.float64)

    indices = np.digitize(data.ravel(), bins_arr).reshape(data.shape)
    result_data = labels_arr[indices]

    # Preserve nodata
    if raster.nodata is not None:
        mask = raster.nodata_mask
        result_data = np.where(mask, raster.nodata, result_data)

    elapsed = time.perf_counter() - t0

    result = from_numpy(
        result_data.astype(np.float64, copy=False),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"raster_classify [CPU] shape={raster.shape} bins={len(bins)} elapsed={elapsed:.4f}s",
            residency=Residency.HOST,
        )
    )
    return result


def raster_classify(
    raster: OwnedRasterArray,
    bins: list[float],
    labels: list[int | float],
    *,
    use_gpu: bool | None = None,
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
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).
    """
    if len(labels) != len(bins) + 1:
        raise ValueError(
            f"labels must have len(bins)+1={len(bins) + 1} elements, got {len(labels)}"
        )

    if use_gpu is None:
        use_gpu = _should_use_gpu_algebra(raster)

    if use_gpu:
        return _raster_classify_gpu(raster, bins, labels)
    else:
        return _raster_classify_cpu(raster, bins, labels)


# ---------------------------------------------------------------------------
# Fused element-wise expression (NVRTC kernel)
# ---------------------------------------------------------------------------

# Expression validation: allowed tokens are variable names, numbers, operators,
# parentheses, commas, whitespace, and whitelisted function names.
_EXPR_TOKEN_RE = re.compile(
    r"""
    [a-h]\[\d+\]                 # band-indexed variable (a[0], b[3], etc.)
    | [a-h](?![a-zA-Z_0-9\[])   # single-letter variable (a-h), not part of longer identifier or band ref
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

# Regex for detecting band-indexed references: a[0], b[3], etc.
_BAND_INDEX_RE = re.compile(r"\b([a-h])\[(\d+)\]")


def _parse_band_indices(
    expression: str,
    rasters: dict[str, OwnedRasterArray],
) -> dict[str, tuple[str, int]] | None:
    """Detect ``varname[N]`` band-index patterns in *expression*.

    Returns
    -------
    dict[str, tuple[str, int]] or None
        Mapping from replacement variable name (e.g. ``"a_band3"``) to
        ``(original_var, band_index)`` tuple.  Returns ``None`` if there
        are no band-index references (pure single-band expression).

    Raises
    ------
    ValueError
        If a band-indexed variable is not present in *rasters*.
    IndexError
        If a band index exceeds the raster's band count.
    """
    matches = _BAND_INDEX_RE.findall(expression)
    if not matches:
        return None

    band_refs: dict[str, tuple[str, int]] = {}
    for var_name, band_str in matches:
        band_idx = int(band_str)
        if var_name not in rasters:
            raise ValueError(
                f"band-indexed variable {var_name!r} is not defined; "
                f"defined variables: {sorted(rasters.keys())}"
            )
        r = rasters[var_name]
        if band_idx >= r.band_count:
            raise IndexError(
                f"{var_name}[{band_idx}] is out of range for raster with "
                f"{r.band_count} band(s) (0-indexed)"
            )
        key = f"{var_name}_band{band_idx}"
        band_refs[key] = (var_name, band_idx)

    return band_refs


def _rewrite_band_expression(
    expression: str,
    band_refs: dict[str, tuple[str, int]],
) -> str:
    """Rewrite ``a[3]`` -> ``a_band3`` in the expression string."""
    result = expression
    for key, (var_name, band_idx) in band_refs.items():
        result = result.replace(f"{var_name}[{band_idx}]", key)
    return result


def _validate_expression(
    expression: str,
    var_names: tuple[str, ...],
    *,
    has_band_indices: bool = False,
) -> str:
    """Validate and translate expression to CUDA-compatible C code.

    Raises ValueError on invalid tokens or references to undefined variables.
    Returns the C expression string.

    Parameters
    ----------
    has_band_indices : bool
        When True, skip the standalone variable check because band-indexed
        variables (``a[3]``) contain a single letter ``a`` that is not a
        standalone variable reference.
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

    if not has_band_indices:
        # Check that all single-letter identifiers reference defined variables.
        # When band indices are present, the standalone letter check is skipped
        # because e.g. ``a`` in ``a[3]`` is not a standalone variable.
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
            _warn_multiband_squeeze(d, stacklevel=4)
            d = d[0]
        d = d.astype(compute_dtype, copy=False)
        _device_refs.append(d)
        input_ptrs.append(d.data.ptr)

        # Pass mask pointer unconditionally when nodata present (avoid mask.any() sync)
        mask = r.device_nodata_mask()
        if mask is not None:
            m = mask.astype(cp.uint8, copy=False)
            if m.ndim == 3:
                m = m[0]  # follows data squeeze above
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

    # Apply nodata where output mask is set (device-side)
    if nodata is not None:
        d_output = cp.where(d_output_mask.astype(cp.bool_), nodata, d_output)

    elapsed = time.perf_counter() - t0

    result = from_device(
        d_output,
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
            residency=Residency.DEVICE,
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
            _warn_multiband_squeeze(data, stacklevel=4)
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


# ---------------------------------------------------------------------------
# Band-indexed expression paths
# ---------------------------------------------------------------------------


def _raster_band_expression_gpu(
    expression: str,
    rasters: dict[str, OwnedRasterArray],
    band_refs: dict[str, tuple[str, int]],
    nodata: float | int | None,
    out_dtype: np.dtype,
) -> OwnedRasterArray:
    """GPU path for band-indexed expressions using the band-fused kernel."""
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.algebra import (
        band_expression_cache_key,
        generate_band_expression_kernel,
    )

    t0 = time.perf_counter()

    # Determine compute dtype
    if np.issubdtype(out_dtype, np.floating) and out_dtype.itemsize <= 4:
        dtype_name = "float"
        compute_dtype = np.float32
    else:
        dtype_name = "double"
        compute_dtype = np.float64

    # Identify which rasters are band-indexed vs standalone single-band
    band_indexed_vars: set[str] = set()
    standalone_vars: set[str] = set()
    for key, (var_name, _band_idx) in band_refs.items():
        band_indexed_vars.add(var_name)

    # Check for standalone (non-band-indexed) single-letter vars in the
    # rewritten expression.  These are separate single-band rasters used
    # alongside band-indexed multiband rasters.
    rewritten = _rewrite_band_expression(expression, band_refs)
    standalone_in_expr = set(re.findall(r"\b([a-h])\b", rewritten))
    standalone_vars = standalone_in_expr - band_indexed_vars

    # Pick the first raster for spatial metadata
    first_var = next(iter(rasters))
    first_raster = rasters[first_var]
    height = first_raster.height
    width = first_raster.width
    n_pixels = height * width

    nodata_val = float(nodata) if nodata is not None else 0.0

    # For the band-fused kernel, we need exactly ONE multiband raster whose
    # BSQ buffer we pass.  All band_refs must reference the same raster.
    band_src_vars = {var_name for var_name, _ in band_refs.values()}
    if len(band_src_vars) > 1 or standalone_vars:
        # Mixed mode: multiple multiband sources or multiband + single-band.
        # Fall back to extracting individual bands as separate pointers and
        # using the standard multi-raster expression kernel.
        return _raster_mixed_band_expression_gpu(
            rewritten,
            rasters,
            band_refs,
            standalone_vars,
            nodata,
            out_dtype,
            dtype_name,
            compute_dtype,
        )

    # Pure single-source band-fused path
    src_var = next(iter(band_src_vars))
    src_raster = rasters[src_var]

    # Build kernel band_refs: {variable_key: band_index}
    kernel_band_refs: dict[str, int] = {}
    for key, (_var_name, band_idx) in band_refs.items():
        kernel_band_refs[key] = band_idx

    # Translate expression functions to CUDA builtins
    c_expression = _translate_expression(rewritten, dtype_name)

    source = generate_band_expression_kernel(
        band_refs=kernel_band_refs,
        expression=c_expression,
        dtype=dtype_name,
        nodata_val=nodata_val,
    )

    cache_key_str = band_expression_cache_key(
        c_expression,
        len(kernel_band_refs),
        dtype_name,
    )
    cache_key = make_kernel_cache_key(cache_key_str, source)

    runtime = get_cuda_runtime()
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=("band_expression",),
    )

    # Move source raster to device
    src_raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="band_expression requires device-resident data",
    )
    d_data = src_raster.device_data().astype(compute_dtype, copy=False)

    # Build nodata mask for the full BSQ buffer
    d_nodata_mask = src_raster.device_nodata_mask()
    if d_nodata_mask is not None:
        d_nodata_mask = d_nodata_mask.astype(cp.uint8, copy=False)
        mask_ptr = d_nodata_mask.data.ptr
    else:
        mask_ptr = 0  # nullptr

    # Allocate output
    d_output = cp.empty((height, width), dtype=compute_dtype)
    d_output_mask = cp.zeros((height, width), dtype=cp.uint8)

    band_stride = n_pixels

    # Build parameter tuple matching kernel signature:
    # (data, nodata_mask, output, output_nodata_mask, n_pixels, band_stride)
    param_values = (
        d_data.data.ptr,
        mask_ptr,
        d_output.data.ptr,
        d_output_mask.data.ptr,
        n_pixels,
        band_stride,
    )
    param_types = (
        KERNEL_PARAM_PTR,  # data
        KERNEL_PARAM_PTR,  # nodata_mask
        KERNEL_PARAM_PTR,  # output
        KERNEL_PARAM_PTR,  # output_nodata_mask
        KERNEL_PARAM_I32,  # n_pixels
        KERNEL_PARAM_I32,  # band_stride
    )
    params = (param_values, param_types)

    grid, block = runtime.launch_config(kernels["band_expression"], n_pixels)
    runtime.launch(
        kernel=kernels["band_expression"],
        grid=grid,
        block=block,
        params=params,
    )

    # Apply nodata where output mask is set
    if nodata is not None:
        d_output = cp.where(d_output_mask.astype(cp.bool_), nodata, d_output)

    elapsed = time.perf_counter() - t0

    result = from_device(
        d_output,
        nodata=nodata,
        affine=first_raster.affine,
        crs=first_raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_kernel=band_expression pixels={n_pixels} "
                f"bands={len(kernel_band_refs)} expr={expression!r} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.DEVICE,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


def _raster_mixed_band_expression_gpu(
    rewritten_expr: str,
    rasters: dict[str, OwnedRasterArray],
    band_refs: dict[str, tuple[str, int]],
    standalone_vars: set[str],
    nodata: float | int | None,
    out_dtype: np.dtype,
    dtype_name: str,
    compute_dtype: np.dtype,
) -> OwnedRasterArray:
    """GPU path for mixed band-indexed + standalone single-band expressions.

    Extracts individual bands as separate device pointers and compiles a
    standard multi-raster expression kernel where each band reference and
    each standalone variable is treated as a separate input.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.algebra import build_expression_kernel_source

    t0 = time.perf_counter()

    first_raster = rasters[next(iter(rasters))]
    height = first_raster.height
    width = first_raster.width
    n_pixels = height * width
    nodata_val = float(nodata) if nodata is not None else 0.0

    # Build ordered variable names for the kernel:
    # band refs (a_band0, a_band3, ...) + standalone vars (b, c, ...)
    all_var_names = sorted(band_refs.keys()) + sorted(standalone_vars)

    # Move data to device and collect pointers
    input_ptrs: list[int] = []
    mask_ptrs: list[int] = []
    _device_refs: list[object] = []  # prevent GC

    for var_key in all_var_names:
        if var_key in band_refs:
            src_var, band_idx = band_refs[var_key]
            r = rasters[src_var]
            r.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="band_expression (mixed) requires device-resident data",
            )
            d = r.device_band(band_idx).astype(compute_dtype, copy=False)
            _device_refs.append(d)
            input_ptrs.append(d.data.ptr)

            mask = r.device_nodata_mask()
            if mask is not None:
                m = mask[band_idx] if mask.ndim == 3 else mask
                m = m.astype(cp.uint8, copy=False)
                _device_refs.append(m)
                mask_ptrs.append(m.data.ptr)
            else:
                mask_ptrs.append(0)
        else:
            # Standalone single-band variable
            r = rasters[var_key]
            r.move_to(
                Residency.DEVICE,
                trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                reason="raster_expression (mixed) requires device-resident data",
            )
            d = r.device_data()
            if d.ndim == 3:
                d = d[0]
            d = d.astype(compute_dtype, copy=False)
            _device_refs.append(d)
            input_ptrs.append(d.data.ptr)

            mask = r.device_nodata_mask()
            if mask is not None:
                m = mask.astype(cp.uint8, copy=False)
                if m.ndim == 3:
                    m = m[0]
                _device_refs.append(m)
                mask_ptrs.append(m.data.ptr)
            else:
                mask_ptrs.append(0)

    # Allocate output
    d_output = cp.empty((height, width), dtype=compute_dtype)
    d_output_mask = cp.zeros((height, width), dtype=cp.uint8)

    # Translate expression functions to CUDA builtins
    c_expression = _translate_expression(rewritten_expr, dtype_name)

    # Build and compile the standard multi-raster expression kernel
    source = build_expression_kernel_source(
        c_expression,
        tuple(all_var_names),
        dtype_name,
    )
    cache_key = make_kernel_cache_key(f"raster_expr_mixed_{dtype_name}", source)

    runtime = get_cuda_runtime()
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=("raster_expression",),
    )

    # Build parameter tuple
    import ctypes

    from vibespatial.cuda_runtime import KERNEL_PARAM_F64

    if dtype_name == "float":
        nodata_param_type = ctypes.c_float
    else:
        nodata_param_type = KERNEL_PARAM_F64

    param_values = (
        tuple(input_ptrs)
        + tuple(mask_ptrs)
        + (d_output.data.ptr, d_output_mask.data.ptr, nodata_val, n_pixels)
    )
    param_types = (
        tuple(KERNEL_PARAM_PTR for _ in all_var_names)
        + tuple(KERNEL_PARAM_PTR for _ in all_var_names)
        + (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, nodata_param_type, KERNEL_PARAM_I32)
    )
    params = (param_values, param_types)

    grid, block = runtime.launch_config(kernels["raster_expression"], n_pixels)
    runtime.launch(
        kernel=kernels["raster_expression"],
        grid=grid,
        block=block,
        params=params,
    )

    if nodata is not None:
        d_output = cp.where(d_output_mask.astype(cp.bool_), nodata, d_output)

    elapsed = time.perf_counter() - t0

    result = from_device(
        d_output,
        nodata=nodata,
        affine=first_raster.affine,
        crs=first_raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_kernel=raster_expression(mixed_band) pixels={n_pixels} "
                f"inputs={len(all_var_names)} expr={rewritten_expr!r} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.DEVICE,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


def _raster_band_expression_cpu(
    expression: str,
    rasters: dict[str, OwnedRasterArray],
    band_refs: dict[str, tuple[str, int]],
    nodata: float | int | None,
    out_dtype: np.dtype,
) -> OwnedRasterArray:
    """CPU fallback for band-indexed expressions using numpy."""
    t0 = time.perf_counter()

    first_raster = rasters[next(iter(rasters))]
    height = first_raster.height
    width = first_raster.width
    flat_shape = (height, width)

    # Determine compute dtype
    if np.issubdtype(out_dtype, np.floating) and out_dtype.itemsize <= 4:
        compute_dtype = np.float32
    else:
        compute_dtype = np.float64

    # Rewrite expression: a[3] -> a_band3
    rewritten = _rewrite_band_expression(expression, band_refs)

    # Build numpy namespace
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

    eval_ns = dict(numpy_funcs)
    combined_mask = np.zeros(flat_shape, dtype=bool)

    # Populate band-indexed variables
    # Track which rasters we've already loaded to avoid redundant copies
    loaded_data: dict[str, np.ndarray] = {}
    loaded_masks: dict[str, np.ndarray] = {}

    for key, (var_name, band_idx) in band_refs.items():
        if var_name not in loaded_data:
            r = rasters[var_name]
            loaded_data[var_name] = r.to_numpy().astype(compute_dtype)
            if r.nodata is not None:
                loaded_masks[var_name] = r.nodata_mask
        data = loaded_data[var_name]
        if data.ndim == 3:
            eval_ns[key] = data[band_idx]
        else:
            # Single-band raster used with a[0]
            eval_ns[key] = data

        # Nodata mask for this band
        if var_name in loaded_masks:
            mask = loaded_masks[var_name]
            if mask.ndim == 3:
                combined_mask |= mask[band_idx]
            else:
                combined_mask |= mask

    # Populate standalone (non-band-indexed) variables
    standalone_in_expr = set(re.findall(r"\b([a-h])\b", rewritten))
    band_indexed_vars = {var_name for var_name, _ in band_refs.values()}
    standalone_vars = standalone_in_expr - band_indexed_vars

    for name in standalone_vars:
        r = rasters[name]
        data = r.to_numpy().astype(compute_dtype)
        if data.ndim == 3:
            _warn_multiband_squeeze(data, stacklevel=5)
            data = data[0]
        eval_ns[name] = data
        if r.nodata is not None:
            mask = r.nodata_mask
            if mask.ndim == 3:
                mask = mask[0]
            combined_mask |= mask

    # Evaluate
    with np.errstate(divide="ignore", invalid="ignore"):
        result = eval(rewritten, {"__builtins__": {}}, eval_ns)

    result = np.asarray(result, dtype=compute_dtype)
    if result.shape != flat_shape:
        result = np.broadcast_to(result, flat_shape).copy()

    # Replace inf/nan with nodata
    if nodata is not None:
        bad = np.logical_or(np.isinf(result), np.isnan(result))
        combined_mask |= bad
        result[combined_mask] = nodata

    elapsed = time.perf_counter() - t0

    out = from_numpy(
        result,
        nodata=nodata,
        affine=first_raster.affine,
        crs=first_raster.crs,
    )
    out.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"band_expression [CPU] pixels={height * width} "
                f"bands={len(band_refs)} expr={expression!r} "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=Residency.HOST,
        )
    )
    return out


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

        Band indexing is supported for multiband rasters: ``a[0]``,
        ``a[3]``, etc. (0-indexed).  Band references can be mixed
        with standalone single-band raster variables.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).
    **rasters : OwnedRasterArray
        Named input rasters.  Variable names must be single lowercase
        letters a-h.  All rasters must have the same spatial dimensions
        (height, width).

    Returns
    -------
    OwnedRasterArray
        Result raster with the same affine and CRS as the first input.
        Output is single-band.  Output dtype is float64 unless all
        inputs are float32.

    Examples
    --------
    NDVI from separate single-band rasters:

    >>> result = raster_expression("(a - b) / (a + b)", a=nir, b=red)

    NDVI from a multiband raster (band 3 = NIR, band 2 = red, 0-indexed):

    >>> result = raster_expression("(a[3] - a[2]) / (a[3] + a[2])", a=multiband)

    Mixed multiband and single-band:

    >>> result = raster_expression("a[0] * b + a[1]", a=multiband, b=scalar)

    Raises
    ------
    ValueError
        If the expression is invalid, variable names are out of range,
        or raster spatial dimensions do not match.
    IndexError
        If a band index exceeds the referenced raster's band count.
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

    # Detect band-index references (a[0], b[3], etc.)
    band_refs = _parse_band_indices(expression, rasters)
    has_band_indices = band_refs is not None

    # Validate spatial dimensions match (height x width).
    # For band-indexed expressions, rasters may have different band counts
    # but must share height and width.
    var_names = tuple(sorted(rasters.keys()))
    ref_raster = rasters[var_names[0]]
    ref_hw = (ref_raster.height, ref_raster.width)
    for name in var_names[1:]:
        r = rasters[name]
        if (r.height, r.width) != ref_hw:
            raise ValueError(
                f"all rasters must have the same spatial dimensions: "
                f"{var_names[0]} has {ref_hw}, "
                f"{name} has {(r.height, r.width)}"
            )

    # Validate expression tokens
    _validate_expression(expression, var_names, has_band_indices=has_band_indices)

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

    if has_band_indices:
        if use_gpu:
            return _raster_band_expression_gpu(
                expression,
                rasters,
                band_refs,
                nodata,
                out_dtype,
            )
        else:
            return _raster_band_expression_cpu(
                expression,
                rasters,
                band_refs,
                nodata,
                out_dtype,
            )
    else:
        if use_gpu:
            return _raster_expression_gpu(expression, rasters, nodata, out_dtype)
        else:
            return _raster_expression_cpu(expression, rasters, nodata, out_dtype)


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
    # Multiband dispatch: process each band independently
    if raster.band_count > 1:
        return dispatch_per_band_gpu(
            raster,
            lambda r: _gpu_convolve(r, kernel_weights),
            buffers_per_band=2,
        )

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
    d_data = _to_device_data(raster).astype(cp.float64, copy=False)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
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
        d_nodata = raster.device_nodata_mask()
        # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
        if d_nodata.ndim == 3:
            d_nodata = d_nodata[0]
        d_nodata = d_nodata.astype(cp.uint8)
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

    result = from_device(
        d_output,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(f"gpu_convolve {width}x{height} kernel={kw}x{kh} smem={shared_mem_bytes}B"),
            residency=Residency.DEVICE,
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
        Standard deviation of the Gaussian.  Must be positive.
    kernel_size : int or None
        Size of the kernel. Default: 2 * ceil(3*sigma) + 1.

    Raises
    ------
    ValueError
        If *sigma* is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
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
    data = dem.to_numpy().astype(np.float64)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
    if data.ndim == 3:
        data = data[0]
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
    d_data = _to_device_data(dem).astype(cp.float64, copy=False)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    # Allocate output buffers on device
    d_slope = cp.zeros_like(d_data) if compute_slope else cp.empty(1, dtype=cp.float64)
    d_aspect = cp.zeros_like(d_data) if compute_aspect else cp.empty(1, dtype=cp.float64)

    # Nodata mask on device (no host round-trip)
    nodata_val = float(dem.nodata) if dem.nodata is not None else 0.0
    if dem.nodata is not None:
        d_nodata = dem.device_nodata_mask()
        # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
        if d_nodata.ndim == 3:
            d_nodata = d_nodata[0]
        d_nodata = d_nodata.astype(cp.uint8)
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

    # Return device arrays directly (zero-copy, no D→H transfer)
    slope_dev = d_slope if compute_slope else None
    aspect_dev = d_aspect if compute_aspect else None

    return slope_dev, aspect_dev


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
        Auto uses GPU when CuPy and CUDA runtime are available and the
        raster exceeds the pixel-count threshold.
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(dem)

    # Multiband dispatch: process each band independently
    if dem.band_count > 1:
        if use_gpu:
            return dispatch_per_band_gpu(
                dem,
                lambda r: raster_slope(r, use_gpu=True),
                buffers_per_band=2,
            )
        return dispatch_per_band_cpu(
            dem,
            lambda r: raster_slope(r, use_gpu=False),
        )

    orig_dtype = dem.dtype

    if use_gpu:
        slope_dev, _ = _gpu_slope_aspect(dem, compute_slope=True, compute_aspect=False)
        # Restore original dtype for float inputs (float32 in -> float32 out)
        if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
            slope_dev = slope_dev.astype(orig_dtype)
        result = from_device(
            slope_dev,
            nodata=dem.nodata,
            affine=dem.affine,
            crs=dem.crs,
        )
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=f"gpu_slope_fused {slope_dev.shape[1]}x{slope_dev.shape[0]}",
                residency=Residency.DEVICE,
            )
        )
    else:
        slope_host, _ = _cpu_slope_aspect(dem, compute_slope=True, compute_aspect=False)
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
                detail=f"cpu_slope_fused {slope_host.shape[1]}x{slope_host.shape[0]}",
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
        Auto uses GPU when CuPy and CUDA runtime are available and the
        raster exceeds the pixel-count threshold.
    """
    if use_gpu is None:
        use_gpu = _should_use_gpu(dem)

    # Multiband dispatch: process each band independently
    if dem.band_count > 1:
        if use_gpu:
            return dispatch_per_band_gpu(
                dem,
                lambda r: raster_aspect(r, use_gpu=True),
                buffers_per_band=2,
            )
        return dispatch_per_band_cpu(
            dem,
            lambda r: raster_aspect(r, use_gpu=False),
        )

    orig_dtype = dem.dtype

    if use_gpu:
        _, aspect_dev = _gpu_slope_aspect(dem, compute_slope=False, compute_aspect=True)
        # Restore original dtype for float inputs (float32 in -> float32 out)
        if np.issubdtype(orig_dtype, np.floating) and orig_dtype != np.float64:
            aspect_dev = aspect_dev.astype(orig_dtype)
        result = from_device(
            aspect_dev,
            nodata=dem.nodata,
            affine=dem.affine,
            crs=dem.crs,
        )
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=f"gpu_aspect_fused {aspect_dev.shape[1]}x{aspect_dev.shape[0]}",
                residency=Residency.DEVICE,
            )
        )
    else:
        _, aspect_host = _cpu_slope_aspect(dem, compute_slope=False, compute_aspect=True)
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
                detail=f"cpu_aspect_fused {aspect_host.shape[1]}x{aspect_host.shape[0]}",
                residency=Residency.HOST,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Hillshade (fused Horn-method slope/aspect + illumination)
# ---------------------------------------------------------------------------


def _hillshade_cpu(
    dem: OwnedRasterArray,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> OwnedRasterArray:
    """CPU hillshade using Horn method (numpy)."""
    # Multiband dispatch: process each band independently
    if dem.band_count > 1:
        return dispatch_per_band_cpu(
            dem,
            lambda r: _hillshade_cpu(r, azimuth=azimuth, altitude=altitude, z_factor=z_factor),
        )

    data = dem.to_numpy().astype(np.float64)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
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
    # Multiband dispatch: process each band independently
    if dem.band_count > 1:
        return dispatch_per_band_gpu(
            dem,
            lambda r: _hillshade_gpu(r, azimuth=azimuth, altitude=altitude, z_factor=z_factor),
            buffers_per_band=2,
        )

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
        d_data = d_data.astype(cp.float64, copy=False)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
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
        # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
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

    elapsed = _time.perf_counter() - t0

    result = from_device(
        d_output,
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
            residency=Residency.DEVICE,
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
    # Multiband dispatch: process each band independently
    if dem.band_count > 1:
        return dispatch_per_band_gpu(
            dem,
            lambda r: _terrain_derivative_gpu(r, deriv_type),
            buffers_per_band=2,
        )

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
    d_data = dem.device_data().astype(cp.float64, copy=False)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    # Output buffer
    d_output = cp.empty((height, width), dtype=cp.float64)

    # Nodata
    nodata_val = float(dem.nodata) if dem.nodata is not None else -9999.0
    if dem.nodata is not None:
        d_nodata = dem.device_nodata_mask()
        # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
        if d_nodata.ndim == 3:
            d_nodata = d_nodata[0]
        d_nodata = d_nodata.astype(cp.uint8)
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

    elapsed = time.perf_counter() - t0

    deriv_names = {_DERIV_TRI: "TRI", _DERIV_TPI: "TPI", _DERIV_CURV: "curvature"}
    result = from_device(
        d_output,
        nodata=nodata_val,
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
            residency=Residency.DEVICE,
            elapsed_seconds=elapsed,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Focal statistics (min, max, mean, std, range, variety)
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(raster: OwnedRasterArray, threshold: int = 10_000) -> bool:
    """Auto-dispatch heuristic: use GPU when available and image is large enough."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


def _parse_radius(radius) -> tuple[int, int]:
    """Normalize radius argument to (radius_y, radius_x).

    Parameters
    ----------
    radius : int or tuple[int, int]
        If int, symmetric radius. If tuple, (radius_y, radius_x).
    """
    if isinstance(radius, (list, tuple)):
        if len(radius) != 2:
            raise ValueError(f"radius tuple must have 2 elements, got {len(radius)}")
        return int(radius[0]), int(radius[1])
    r = int(radius)
    return r, r


# -- CPU fallback implementations --


def _focal_min_cpu(data: np.ndarray, radius_y: int, radius_x: int, nodata_mask, nodata_val):
    """CPU focal min via scipy generic_filter."""
    from scipy.ndimage import generic_filter

    size = (2 * radius_y + 1, 2 * radius_x + 1)
    work = data.astype(np.float64, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = np.inf

    def _fn(values):
        valid = values[values != np.inf]
        if len(valid) == 0:
            return nodata_val if nodata_val is not None else np.nan
        return valid.min()

    result = generic_filter(work, _fn, size=size, mode="constant", cval=np.inf)
    if nodata_mask is not None and nodata_val is not None:
        result[nodata_mask] = nodata_val
    return result


def _focal_max_cpu(data: np.ndarray, radius_y: int, radius_x: int, nodata_mask, nodata_val):
    """CPU focal max via scipy generic_filter."""
    from scipy.ndimage import generic_filter

    size = (2 * radius_y + 1, 2 * radius_x + 1)
    work = data.astype(np.float64, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = -np.inf

    def _fn(values):
        valid = values[values != -np.inf]
        if len(valid) == 0:
            return nodata_val if nodata_val is not None else np.nan
        return valid.max()

    result = generic_filter(work, _fn, size=size, mode="constant", cval=-np.inf)
    if nodata_mask is not None and nodata_val is not None:
        result[nodata_mask] = nodata_val
    return result


def _focal_mean_cpu(data: np.ndarray, radius_y: int, radius_x: int, nodata_mask, nodata_val):
    """CPU focal mean via scipy generic_filter."""
    from scipy.ndimage import generic_filter

    size = (2 * radius_y + 1, 2 * radius_x + 1)
    work = data.astype(np.float64, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = np.nan

    def _fn(values):
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return nodata_val if nodata_val is not None else np.nan
        return valid.mean()

    result = generic_filter(work, _fn, size=size, mode="constant", cval=np.nan)
    if nodata_mask is not None and nodata_val is not None:
        result[nodata_mask] = nodata_val
    return result


def _focal_std_cpu(data: np.ndarray, radius_y: int, radius_x: int, nodata_mask, nodata_val):
    """CPU focal std (population std, ddof=0) via scipy generic_filter."""
    from scipy.ndimage import generic_filter

    size = (2 * radius_y + 1, 2 * radius_x + 1)
    work = data.astype(np.float64, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = np.nan

    def _fn(values):
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return 0.0
        return valid.std(ddof=0)

    result = generic_filter(work, _fn, size=size, mode="constant", cval=np.nan)
    if nodata_mask is not None and nodata_val is not None:
        result[nodata_mask] = nodata_val
    return result


def _focal_range_cpu(data: np.ndarray, radius_y: int, radius_x: int, nodata_mask, nodata_val):
    """CPU focal range via scipy generic_filter."""
    from scipy.ndimage import generic_filter

    size = (2 * radius_y + 1, 2 * radius_x + 1)
    work = data.astype(np.float64, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = np.nan

    def _fn(values):
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return nodata_val if nodata_val is not None else np.nan
        return valid.max() - valid.min()

    result = generic_filter(work, _fn, size=size, mode="constant", cval=np.nan)
    if nodata_mask is not None and nodata_val is not None:
        result[nodata_mask] = nodata_val
    return result


def _focal_variety_cpu(data: np.ndarray, radius_y: int, radius_x: int, nodata_mask, nodata_val):
    """CPU focal variety (count unique) via scipy generic_filter."""
    from scipy.ndimage import generic_filter

    size = (2 * radius_y + 1, 2 * radius_x + 1)
    work = data.astype(np.float64, copy=True)
    if nodata_mask is not None:
        work[nodata_mask] = np.nan

    def _fn(values):
        valid = values[~np.isnan(values)]
        if len(valid) == 0:
            return 0.0
        return float(len(np.unique(valid)))

    result = generic_filter(work, _fn, size=size, mode="constant", cval=np.nan)
    if nodata_mask is not None and nodata_val is not None:
        result[nodata_mask] = nodata_val
    return result


_CPU_DISPATCH = {
    "min": _focal_min_cpu,
    "max": _focal_max_cpu,
    "mean": _focal_mean_cpu,
    "std": _focal_std_cpu,
    "range": _focal_range_cpu,
    "variety": _focal_variety_cpu,
}


def _focal_stat_cpu(
    raster: OwnedRasterArray, stat_name: str, radius_y: int, radius_x: int
) -> OwnedRasterArray:
    """CPU fallback for a single focal statistic."""
    # Multiband dispatch: process each band independently
    if raster.band_count > 1:
        return dispatch_per_band_cpu(
            raster,
            lambda r: _focal_stat_cpu(r, stat_name, radius_y, radius_x),
        )

    data = raster.to_numpy()
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
    if data.ndim == 3:
        data = data[0]

    nodata_mask = raster.nodata_mask if raster.nodata is not None else None
    if nodata_mask is not None and nodata_mask.ndim == 3:
        nodata_mask = nodata_mask[0]
    nodata_val = float(raster.nodata) if raster.nodata is not None else 0.0

    fn = _CPU_DISPATCH[stat_name]
    t0 = time.monotonic()
    result_data = fn(data, radius_y, radius_x, nodata_mask, nodata_val)
    elapsed = time.monotonic() - t0

    result = from_numpy(
        result_data,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"cpu_focal_{stat_name} {raster.width}x{raster.height} "
                f"radius=({radius_y},{radius_x})"
            ),
            residency=Residency.HOST,
            elapsed_seconds=elapsed,
        )
    )
    return result


# -- Terrain GPU diagnostic (kept from HEAD) --


# -- GPU implementation --

# Maps stat name -> kernel int flag
_STAT_NAME_TO_FLAG = {
    "min": 0,
    "max": 1,
    "mean": 2,
    "std": 3,
    "range": 4,
    "variety": 5,
}


def _focal_stat_gpu(
    raster: OwnedRasterArray, stat_name: str, radius_y: int, radius_x: int
) -> OwnedRasterArray:
    """GPU implementation of focal statistics via NVRTC kernel."""
    # Multiband dispatch: process each band independently
    if raster.band_count > 1:
        return dispatch_per_band_gpu(
            raster,
            lambda r: _focal_stat_gpu(r, stat_name, radius_y, radius_x),
            buffers_per_band=2,
        )

    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_F64,
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.focal import FOCAL_STATS_KERNEL_SOURCE

    t0 = time.monotonic()

    # Move to device, cast to float64 for kernel computation
    d_data = _to_device_data(raster).astype(cp.float64, copy=False)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
    if d_data.ndim == 3:
        d_data = d_data[0]

    height, width = d_data.shape

    d_output = cp.zeros_like(d_data)

    nodata_val = float(raster.nodata) if raster.nodata is not None else 0.0

    if raster.nodata is not None:
        d_nodata = raster.device_nodata_mask()
        # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
        if d_nodata.ndim == 3:
            d_nodata = d_nodata[0]
        d_nodata = d_nodata.astype(cp.uint8)
        nodata_ptr = d_nodata.data.ptr
    else:
        nodata_ptr = 0  # nullptr

    stat_flag = _STAT_NAME_TO_FLAG[stat_name]

    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key("focal_stats", FOCAL_STATS_KERNEL_SOURCE)
    kernels = runtime.compile_kernels(
        cache_key=cache_key,
        source=FOCAL_STATS_KERNEL_SOURCE,
        kernel_names=("focal_stats",),
    )

    kernel = kernels["focal_stats"]

    # Use 2D block layout for stencil kernel
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)

    params = (
        (
            d_data.data.ptr,
            d_output.data.ptr,
            nodata_ptr,
            width,
            height,
            radius_x,
            radius_y,
            stat_flag,
            nodata_val,
        ),
        (
            KERNEL_PARAM_PTR,  # input
            KERNEL_PARAM_PTR,  # output
            KERNEL_PARAM_PTR,  # nodata_mask
            KERNEL_PARAM_I32,  # width
            KERNEL_PARAM_I32,  # height
            KERNEL_PARAM_I32,  # radius_x
            KERNEL_PARAM_I32,  # radius_y
            KERNEL_PARAM_I32,  # stat_type
            KERNEL_PARAM_F64,  # nodata_val
        ),
    )

    runtime.launch(kernel=kernel, grid=grid, block=block, params=params)

    elapsed = time.monotonic() - t0

    result = from_device(
        d_output,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"gpu_focal_{stat_name} {width}x{height} "
                f"radius=({radius_y},{radius_x}) blocks={grid}"
            ),
            residency=Residency.DEVICE,
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
    # Multiband dispatch: process each band independently
    if dem.band_count > 1:
        return dispatch_per_band_cpu(
            dem,
            lambda r: _terrain_derivative_cpu(r, deriv_type),
        )

    data = dem.to_numpy().astype(np.float64)
    # Normalize (1, H, W) -> (H, W) for single-band stored as 3D
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
        nodata=nodata_val,
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


# -- Focal statistics public API --


def raster_focal_min(
    raster: OwnedRasterArray,
    radius: int | tuple[int, int] = 1,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute focal (neighborhood) minimum.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    radius : int or tuple[int, int]
        Window radius. If int, symmetric. If tuple, (radius_y, radius_x).
        Window size = 2*radius + 1 in each dimension.
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-select (None).
    """
    radius_y, radius_x = _parse_radius(radius)
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _focal_stat_gpu(raster, "min", radius_y, radius_x)
    return _focal_stat_cpu(raster, "min", radius_y, radius_x)


def raster_focal_max(
    raster: OwnedRasterArray,
    radius: int | tuple[int, int] = 1,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute focal (neighborhood) maximum.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    radius : int or tuple[int, int]
        Window radius. If int, symmetric. If tuple, (radius_y, radius_x).
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-select (None).
    """
    radius_y, radius_x = _parse_radius(radius)
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _focal_stat_gpu(raster, "max", radius_y, radius_x)
    return _focal_stat_cpu(raster, "max", radius_y, radius_x)


def raster_focal_mean(
    raster: OwnedRasterArray,
    radius: int | tuple[int, int] = 1,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute focal (neighborhood) mean.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    radius : int or tuple[int, int]
        Window radius. If int, symmetric. If tuple, (radius_y, radius_x).
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-select (None).
    """
    radius_y, radius_x = _parse_radius(radius)
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _focal_stat_gpu(raster, "mean", radius_y, radius_x)
    return _focal_stat_cpu(raster, "mean", radius_y, radius_x)


def raster_focal_std(
    raster: OwnedRasterArray,
    radius: int | tuple[int, int] = 1,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute focal (neighborhood) standard deviation (population, ddof=0).

    Uses Welford's online algorithm on GPU to avoid catastrophic cancellation.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    radius : int or tuple[int, int]
        Window radius. If int, symmetric. If tuple, (radius_y, radius_x).
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-select (None).
    """
    radius_y, radius_x = _parse_radius(radius)
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _focal_stat_gpu(raster, "std", radius_y, radius_x)
    return _focal_stat_cpu(raster, "std", radius_y, radius_x)


def raster_focal_range(
    raster: OwnedRasterArray,
    radius: int | tuple[int, int] = 1,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute focal (neighborhood) range (max - min).

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    radius : int or tuple[int, int]
        Window radius. If int, symmetric. If tuple, (radius_y, radius_x).
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-select (None).
    """
    radius_y, radius_x = _parse_radius(radius)
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _focal_stat_gpu(raster, "range", radius_y, radius_x)
    return _focal_stat_cpu(raster, "range", radius_y, radius_x)


def raster_focal_variety(
    raster: OwnedRasterArray,
    radius: int | tuple[int, int] = 1,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Compute focal (neighborhood) variety (count of unique values).

    Uses register-based unique counting on GPU. Practical for windows
    up to ~7x7 (49 unique values max).

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single-band).
    radius : int or tuple[int, int]
        Window radius. If int, symmetric. If tuple, (radius_y, radius_x).
    use_gpu : bool or None
        Force GPU (True), CPU (False), or auto-select (None).
    """
    radius_y, radius_x = _parse_radius(radius)
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _focal_stat_gpu(raster, "variety", radius_y, radius_x)
    return _focal_stat_cpu(raster, "variety", radius_y, radius_x)


# ---------------------------------------------------------------------------
# Spectral index convenience functions
# ---------------------------------------------------------------------------


def raster_ndvi(
    raster: OwnedRasterArray,
    *,
    nir_band: int = 4,
    red_band: int = 3,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Normalized Difference Vegetation Index: (NIR - RED) / (NIR + RED).

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster containing NIR and RED bands.
    nir_band : int
        1-indexed band number for the NIR band (default 4, Landsat/Sentinel-2
        convention).
    red_band : int
        1-indexed band number for the RED band (default 3).
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).

    Returns
    -------
    OwnedRasterArray
        Single-band raster with NDVI values in [-1, 1].  Pixels where
        NIR + RED == 0 are set to nodata.

    Raises
    ------
    ValueError
        If band indices are less than 1.
    IndexError
        If band indices exceed the raster's band count.
    """
    if nir_band < 1:
        raise ValueError(f"nir_band must be >= 1 (1-indexed), got {nir_band}")
    if red_band < 1:
        raise ValueError(f"red_band must be >= 1 (1-indexed), got {red_band}")

    # Convert 1-indexed (rasterio convention) to 0-indexed for expression engine
    nir_idx = nir_band - 1
    red_idx = red_band - 1

    # Validate band indices against raster band count
    for idx, label in ((nir_idx, "nir_band"), (red_idx, "red_band")):
        if idx >= raster.band_count:
            raise IndexError(
                f"{label}={idx + 1} (0-indexed: {idx}) exceeds raster band "
                f"count of {raster.band_count}"
            )

    expr = f"(a[{nir_idx}] - a[{red_idx}]) / (a[{nir_idx}] + a[{red_idx}])"
    return raster_expression(expr, a=raster, use_gpu=use_gpu)


def raster_band_ratio(
    raster: OwnedRasterArray,
    *,
    band_a: int,
    band_b: int,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Generic band ratio: band_a / band_b.

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster.
    band_a : int
        1-indexed band number for the numerator.
    band_b : int
        1-indexed band number for the denominator.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).

    Returns
    -------
    OwnedRasterArray
        Single-band raster with the ratio values.  Pixels where band_b == 0
        are set to nodata.

    Raises
    ------
    ValueError
        If band indices are less than 1.
    IndexError
        If band indices exceed the raster's band count.
    """
    if band_a < 1:
        raise ValueError(f"band_a must be >= 1 (1-indexed), got {band_a}")
    if band_b < 1:
        raise ValueError(f"band_b must be >= 1 (1-indexed), got {band_b}")

    idx_a = band_a - 1
    idx_b = band_b - 1

    for idx, label in ((idx_a, "band_a"), (idx_b, "band_b")):
        if idx >= raster.band_count:
            raise IndexError(
                f"{label}={idx + 1} (0-indexed: {idx}) exceeds raster band "
                f"count of {raster.band_count}"
            )

    expr = f"a[{idx_a}] / a[{idx_b}]"
    return raster_expression(expr, a=raster, use_gpu=use_gpu)


def raster_band_math(
    raster: OwnedRasterArray,
    expression: str,
    *,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Arbitrary band math on a multiband raster.

    The expression uses 0-indexed band references with the variable ``b``:
    ``b[0]``, ``b[1]``, etc.  This is a thin wrapper around
    :func:`raster_expression` that binds the input raster as variable ``b``.

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster.
    expression : str
        Arithmetic expression using ``b[N]`` band references (0-indexed).
        Supported operators: ``+``, ``-``, ``*``, ``/``.
        Supported functions: ``abs``, ``sqrt``, ``pow``, ``min``, ``max``,
        ``clamp``, ``log``, ``log2``, ``log10``, ``exp``, ``sin``, ``cos``,
        ``tan``, ``floor``, ``ceil``.

        Example: ``"(b[3] - b[2]) / (b[3] + b[2] + 0.5 * b[1])"``
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-detect (None).

    Returns
    -------
    OwnedRasterArray
        Single-band raster with the computed result.

    Raises
    ------
    ValueError
        If the expression is empty or contains invalid tokens.
    IndexError
        If a band index exceeds the raster's band count.
    """
    if not expression or not expression.strip():
        raise ValueError("expression must not be empty")
    return raster_expression(expression, b=raster, use_gpu=use_gpu)
