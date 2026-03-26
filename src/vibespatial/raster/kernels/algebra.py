"""NVRTC kernel source templates for fused element-wise raster algebra.

Two kernel families live here:

1. **Multi-raster expression** (``raster_expression``): N separate
   single-band rasters (each with its own device pointer and nodata mask)
   are combined by a user expression.  Parameters grow linearly with the
   number of inputs.

2. **Band-fused expression** (``band_expression``): A single contiguous
   BSQ (B, H, W) buffer holds multiple bands.  The kernel reads from
   compile-time-constant band offsets within that buffer and produces a
   single-band output.  This avoids per-band pointer parameters and is
   ideal for cross-band spectral indices (NDVI, EVI, SAVI, ...).

Both use grid-stride loops with ILP=4 for bandwidth-bound workloads.
"""

from __future__ import annotations

import hashlib

# Template placeholders:
#   {n_inputs}    - number of input rasters (1-8)
#   {dtype}       - C type name: "float" or "double"
#   {expression}  - sanitised C expression string
#   {input_ptrs}  - parameter declarations for input pointers
#   {mask_ptrs}   - parameter declarations for mask pointers
#   {load_inputs} - statements that load input values into local vars
#   {load_masks}  - statements that check per-input nodata masks

EXPRESSION_KERNEL_SOURCE_TEMPLATE = r"""
extern "C" __global__
void raster_expression(
    {input_ptrs}
    {mask_ptrs}
    {dtype}* __restrict__ output,
    unsigned char* __restrict__ output_nodata_mask,
    const {dtype} nodata_val,
    const int n
) {{
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {{

        // Check nodata masks — any input nodata => output nodata
        bool is_nodata = false;
        {load_masks}

        if (is_nodata) {{
            output[idx] = nodata_val;
            output_nodata_mask[idx] = 1;
            continue;
        }}

        // Load input values
        {load_inputs}

        // Evaluate expression
        {dtype} _result = ({dtype})({expression});

        // Handle inf/nan from division by zero etc.
        if (isinf(_result) || isnan(_result)) {{
            output[idx] = nodata_val;
            output_nodata_mask[idx] = 1;
        }} else {{
            output[idx] = _result;
            output_nodata_mask[idx] = 0;
        }}
    }}
}}
"""

# Valid variable names for inputs (a through h, supporting up to 8 inputs)
INPUT_VAR_NAMES = tuple("abcdefgh")

# Allowed function names in expressions — mapped to CUDA math builtins
ALLOWED_FUNCTIONS = {
    "abs": "fabs",
    "sqrt": "sqrt",
    "pow": "pow",
    "min": "fmin",
    "max": "fmax",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "exp": "exp",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "floor": "floor",
    "ceil": "ceil",
}

# For float32, use the f-suffixed versions for correct precision
ALLOWED_FUNCTIONS_F32 = {
    "abs": "fabsf",
    "sqrt": "sqrtf",
    "pow": "powf",
    "min": "fminf",
    "max": "fmaxf",
    "log": "logf",
    "log2": "log2f",
    "log10": "log10f",
    "exp": "expf",
    "sin": "sinf",
    "cos": "cosf",
    "tan": "tanf",
    "floor": "floorf",
    "ceil": "ceilf",
}


def build_expression_kernel_source(
    expression: str,
    var_names: tuple[str, ...],
    dtype_name: str,
) -> str:
    """Build a complete NVRTC kernel source for the given expression.

    Parameters
    ----------
    expression : str
        C-compatible arithmetic expression using var_names as identifiers.
    var_names : tuple[str, ...]
        Ordered variable names (e.g., ("a", "b")).  Must be a subset of
        INPUT_VAR_NAMES.
    dtype_name : str
        C dtype: "float" or "double".

    Returns
    -------
    str
        Complete NVRTC kernel source ready for compilation.
    """
    n = len(var_names)
    if n < 1 or n > len(INPUT_VAR_NAMES):
        raise ValueError(f"expression must use 1-{len(INPUT_VAR_NAMES)} inputs, got {n}")

    # Build parameter declarations
    input_ptrs = "\n    ".join(f"const {dtype_name}* __restrict__ in_{name}," for name in var_names)
    mask_ptrs = "\n    ".join(
        f"const unsigned char* __restrict__ mask_{name}," for name in var_names
    )

    # Build mask checks
    mask_lines = []
    for name in var_names:
        mask_lines.append(f"if (mask_{name} != nullptr && mask_{name}[idx]) is_nodata = true;")
    load_masks = "\n        ".join(mask_lines)

    # Build input loading
    load_lines = []
    for name in var_names:
        load_lines.append(f"{dtype_name} {name} = in_{name}[idx];")
    load_inputs = "\n        ".join(load_lines)

    source = EXPRESSION_KERNEL_SOURCE_TEMPLATE.format(
        dtype=dtype_name,
        expression=expression,
        input_ptrs=input_ptrs,
        mask_ptrs=mask_ptrs,
        load_inputs=load_inputs,
        load_masks=load_masks,
        n_inputs=n,
    )
    return source


# ---------------------------------------------------------------------------
# Band-fused expression kernel — single BSQ buffer, multiple bands
# ---------------------------------------------------------------------------

# Template placeholders:
#   {dtype}         - C type name: "float" or "double"
#   {nodata_checks} - generated per-band nodata checks
#   {band_loads}    - generated variable declarations loading from band offsets
#   {expression}    - sanitised C expression string
#   {nodata_val}    - the nodata sentinel value as a C literal

BAND_EXPRESSION_KERNEL_SOURCE_TEMPLATE = r"""
extern "C" __global__
void band_expression(
    const {dtype}* __restrict__ data,
    const unsigned char* __restrict__ nodata_mask,
    {dtype}* __restrict__ output,
    unsigned char* __restrict__ output_nodata_mask,
    const int n_pixels,
    const int band_stride
) {{
    const int stride = blockDim.x * gridDim.x;
    for (int base = blockIdx.x * blockDim.x + threadIdx.x;
         base < n_pixels;
         base += stride * 4) {{
        #pragma unroll
        for (int j = 0; j < 4; j++) {{
            int i = base + j * stride;
            if (i >= n_pixels) break;

            // Union nodata check for referenced bands
            bool is_nodata = false;
            {nodata_checks}

            if (is_nodata) {{
                output[i] = {nodata_val};
                output_nodata_mask[i] = 1;
                continue;
            }}

            // Load referenced band values
            {band_loads}

            // Compute expression
            {dtype} _result = ({dtype})({expression});

            // Guard inf/nan from division by zero etc.
            if (isinf(_result) || isnan(_result)) {{
                output[i] = {nodata_val};
                output_nodata_mask[i] = 1;
            }} else {{
                output[i] = _result;
                output_nodata_mask[i] = 0;
            }}
        }}
    }}
}}
"""


def generate_band_expression_kernel(
    band_refs: dict[str, int],
    expression: str,
    dtype: str,
    nodata_val: float | None,
) -> str:
    """Generate a compilable NVRTC kernel source for a cross-band expression.

    The kernel reads from a single contiguous BSQ (B, H, W) buffer and
    produces a single-band (H, W) output.  Band indices are baked into the
    source as compile-time integer literals, so the only runtime parameters
    are the buffer pointer, nodata mask pointer, output pointer, pixel count,
    and band stride (= H * W).

    Parameters
    ----------
    band_refs : dict[str, int]
        Mapping of variable names to 0-indexed band indices within the BSQ
        buffer.  For example ``{"nir": 3, "red": 2}`` means the variable
        ``nir`` reads from ``data[3 * band_stride + i]``.
    expression : str
        C-compatible arithmetic expression using the variable names from
        *band_refs* as identifiers.  For NDVI:
        ``"(nir - red) / (nir + red)"``.
    dtype : str
        C type name: ``"float"`` or ``"double"``.
    nodata_val : float or None
        The nodata sentinel value.  When ``None``, the nodata value
        defaults to ``0.0`` (the kernel still writes output_nodata_mask
        for downstream consumers to use).

    Returns
    -------
    str
        Complete NVRTC kernel source ready for ``runtime.compile_kernels()``.

    Raises
    ------
    ValueError
        If *band_refs* is empty or *dtype* is not ``"float"`` or
        ``"double"``.
    """
    if not band_refs:
        raise ValueError("band_refs must contain at least one band reference")
    if dtype not in ("float", "double"):
        raise ValueError(f"dtype must be 'float' or 'double', got {dtype!r}")

    # Format the nodata sentinel as a C literal
    nd = nodata_val if nodata_val is not None else 0.0
    if dtype == "float":
        nodata_literal = f"{nd!r}f"
    else:
        nodata_literal = repr(nd)

    # Build per-band nodata checks.
    # Each referenced band contributes one nullable check against the BSQ
    # nodata_mask at that band's offset.
    nodata_lines: list[str] = []
    for band_idx in band_refs.values():
        nodata_lines.append(
            f"if (nodata_mask != nullptr && nodata_mask[{band_idx} * band_stride + i]) "
            f"is_nodata = true;"
        )
    nodata_checks = "\n            ".join(nodata_lines)

    # Build per-band variable loads.
    # Each variable loads from its band's offset within the BSQ buffer.
    load_lines: list[str] = []
    for name, band_idx in band_refs.items():
        load_lines.append(f"{dtype} {name} = data[{band_idx} * band_stride + i];")
    band_loads = "\n            ".join(load_lines)

    source = BAND_EXPRESSION_KERNEL_SOURCE_TEMPLATE.format(
        dtype=dtype,
        nodata_checks=nodata_checks,
        band_loads=band_loads,
        expression=expression,
        nodata_val=nodata_literal,
    )
    return source


def band_expression_cache_key(
    expression: str,
    band_count: int,
    dtype: str,
) -> str:
    """Compute a unique cache key for a band-fused expression kernel.

    The key includes a SHA-1 hash of the expression, the number of referenced
    bands, and the dtype so that different expressions, band counts, or dtypes
    produce different compiled kernels.

    Parameters
    ----------
    expression : str
        The user expression string (pre-translation to CUDA builtins).
    band_count : int
        Number of bands referenced in the expression.
    dtype : str
        C type name: ``"float"`` or ``"double"``.

    Returns
    -------
    str
        A cache key string suitable for ``runtime.compile_kernels()``.
    """
    payload = f"{expression}|{band_count}|{dtype}"
    digest = hashlib.sha1(payload.encode()).hexdigest()
    return f"band_expr_{dtype}_{band_count}b-{digest}"
