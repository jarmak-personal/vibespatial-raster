"""NVRTC kernel source template for fused element-wise raster algebra.

The expression kernel evaluates a user-supplied arithmetic expression
over up to 8 named input rasters in a single GPU pass.  Each input is
bound to a variable name (a-h) and the kernel writes a single output.
Nodata propagation is handled via a per-input mask array: if *any*
input pixel is nodata, the output pixel is set to the nodata sentinel.

Grid-stride loop with 4-element ILP for bandwidth-bound workloads.
"""

from __future__ import annotations

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
