"""NVRTC kernel sources for raster resampling/warping.

Each output pixel independently computes its source coordinate via
inverse affine transform and samples the source raster. This is
embarrassingly parallel -- one thread per output pixel with a
grid-stride loop for large rasters.

Three interpolation methods:
  - Nearest-neighbor: snaps to closest source pixel
  - Bilinear: 2x2 weighted linear interpolation
  - Bicubic: 4x4 Catmull-Rom spline interpolation

All kernels propagate nodata: if any contributing source pixel is nodata
(or falls outside bounds), the output pixel is set to nodata.

Kernel source strings are templated on dtype via Python .format() before
compilation. The placeholder {dtype} is replaced with the C type name
(e.g., "float", "double", "short", "unsigned char").
"""

from __future__ import annotations


def _resample_kernel_source(dtype_c: str) -> str:
    """Return NVRTC source for all three resample kernels, templated on dtype.

    Parameters
    ----------
    dtype_c : str
        C type name for the raster data (e.g., "double", "float",
        "short", "unsigned char").
    """
    return r"""
// Catmull-Rom cubic basis function
__device__ double cubic_weight(double t) {{
    double at = (t < 0.0) ? -t : t;
    if (at <= 1.0) {{
        return (1.5 * at * at * at) - (2.5 * at * at) + 1.0;
    }} else if (at < 2.0) {{
        return (-0.5 * at * at * at) + (2.5 * at * at) - (4.0 * at) + 2.0;
    }}
    return 0.0;
}}

// ---------------------------------------------------------------------------
// Nearest-neighbor resampling
// ---------------------------------------------------------------------------
extern "C" __global__
void resample_nearest(
    const {dtype}* __restrict__ src,
    {dtype}* __restrict__ dst,
    const unsigned char* __restrict__ src_nodata_mask,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const double inv_a,
    const double inv_b,
    const double inv_c,
    const double inv_d,
    const double inv_e,
    const double inv_f,
    const {dtype} nodata_val,
    const int has_nodata
) {{
    const int total = dst_width * dst_height;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {{

        int dst_row = idx / dst_width;
        int dst_col = idx % dst_width;

        // Output pixel center in world coordinates (target affine applied
        // by the caller -- we receive the COMPOSED inverse from target
        // pixel coords directly to source pixel coords).
        double pc = (double)dst_col + 0.5;
        double pr = (double)dst_row + 0.5;

        double src_col_f = inv_a * pc + inv_b * pr + inv_c;
        double src_row_f = inv_d * pc + inv_e * pr + inv_f;

        int sc = (int)floor(src_col_f);
        int sr = (int)floor(src_row_f);

        if (sc < 0 || sc >= src_width || sr < 0 || sr >= src_height) {{
            dst[idx] = nodata_val;
        }} else {{
            long long si = (long long)sr * src_width + sc;
            if (has_nodata && src_nodata_mask != nullptr && src_nodata_mask[si]) {{
                dst[idx] = nodata_val;
            }} else {{
                dst[idx] = src[si];
            }}
        }}
    }}
}}

// ---------------------------------------------------------------------------
// Bilinear resampling (2x2 interpolation)
// ---------------------------------------------------------------------------
extern "C" __global__
void resample_bilinear(
    const {dtype}* __restrict__ src,
    {dtype}* __restrict__ dst,
    const unsigned char* __restrict__ src_nodata_mask,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const double inv_a,
    const double inv_b,
    const double inv_c,
    const double inv_d,
    const double inv_e,
    const double inv_f,
    const {dtype} nodata_val,
    const int has_nodata
) {{
    const int total = dst_width * dst_height;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {{

        int dst_row = idx / dst_width;
        int dst_col = idx % dst_width;

        double pc = (double)dst_col + 0.5;
        double pr = (double)dst_row + 0.5;

        double src_col_f = inv_a * pc + inv_b * pr + inv_c;
        double src_row_f = inv_d * pc + inv_e * pr + inv_f;

        // Pixel centers are at integer + 0.5; shift to get fractional index
        double sx = src_col_f - 0.5;
        double sy = src_row_f - 0.5;

        int x0 = (int)floor(sx);
        int y0 = (int)floor(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        double fx = sx - (double)x0;
        double fy = sy - (double)y0;

        // Bounds check: all 4 source pixels must be in bounds
        if (x0 < 0 || x1 >= src_width || y0 < 0 || y1 >= src_height) {{
            dst[idx] = nodata_val;
            continue;
        }}

        long long i00 = (long long)y0 * src_width + x0;
        long long i01 = (long long)y0 * src_width + x1;
        long long i10 = (long long)y1 * src_width + x0;
        long long i11 = (long long)y1 * src_width + x1;

        // Nodata check: if any contributing pixel is nodata, output nodata
        if (has_nodata && src_nodata_mask != nullptr) {{
            if (src_nodata_mask[i00] || src_nodata_mask[i01] ||
                src_nodata_mask[i10] || src_nodata_mask[i11]) {{
                dst[idx] = nodata_val;
                continue;
            }}
        }}

        double v00 = (double)src[i00];
        double v01 = (double)src[i01];
        double v10 = (double)src[i10];
        double v11 = (double)src[i11];

        double val = v00 * (1.0 - fx) * (1.0 - fy)
                   + v01 * fx         * (1.0 - fy)
                   + v10 * (1.0 - fx) * fy
                   + v11 * fx         * fy;

        dst[idx] = ({dtype})val;
    }}
}}

// ---------------------------------------------------------------------------
// Bicubic resampling (4x4 Catmull-Rom)
// ---------------------------------------------------------------------------
extern "C" __global__
void resample_bicubic(
    const {dtype}* __restrict__ src,
    {dtype}* __restrict__ dst,
    const unsigned char* __restrict__ src_nodata_mask,
    const int src_width,
    const int src_height,
    const int dst_width,
    const int dst_height,
    const double inv_a,
    const double inv_b,
    const double inv_c,
    const double inv_d,
    const double inv_e,
    const double inv_f,
    const {dtype} nodata_val,
    const int has_nodata
) {{
    const int total = dst_width * dst_height;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride) {{

        int dst_row = idx / dst_width;
        int dst_col = idx % dst_width;

        double pc = (double)dst_col + 0.5;
        double pr = (double)dst_row + 0.5;

        double src_col_f = inv_a * pc + inv_b * pr + inv_c;
        double src_row_f = inv_d * pc + inv_e * pr + inv_f;

        double sx = src_col_f - 0.5;
        double sy = src_row_f - 0.5;

        int x0 = (int)floor(sx) - 1;  // leftmost of 4x4 neighborhood
        int y0 = (int)floor(sy) - 1;

        double fx = sx - floor(sx);
        double fy = sy - floor(sy);

        // 4x4 neighborhood bounds check
        if (x0 < 0 || (x0 + 3) >= src_width ||
            y0 < 0 || (y0 + 3) >= src_height) {{
            dst[idx] = nodata_val;
            continue;
        }}

        // Nodata check across all 16 contributing pixels
        if (has_nodata && src_nodata_mask != nullptr) {{
            bool any_nodata = false;
            for (int jj = 0; jj < 4 && !any_nodata; jj++) {{
                for (int ii = 0; ii < 4 && !any_nodata; ii++) {{
                    long long si = (long long)(y0 + jj) * src_width + (x0 + ii);
                    if (src_nodata_mask[si]) any_nodata = true;
                }}
            }}
            if (any_nodata) {{
                dst[idx] = nodata_val;
                continue;
            }}
        }}

        // Compute bicubic weights
        double wx[4], wy[4];
        wx[0] = cubic_weight(fx + 1.0);
        wx[1] = cubic_weight(fx);
        wx[2] = cubic_weight(fx - 1.0);
        wx[3] = cubic_weight(fx - 2.0);

        wy[0] = cubic_weight(fy + 1.0);
        wy[1] = cubic_weight(fy);
        wy[2] = cubic_weight(fy - 1.0);
        wy[3] = cubic_weight(fy - 2.0);

        double val = 0.0;
        for (int jj = 0; jj < 4; jj++) {{
            for (int ii = 0; ii < 4; ii++) {{
                long long si = (long long)(y0 + jj) * src_width + (x0 + ii);
                val += (double)src[si] * wx[ii] * wy[jj];
            }}
        }}

        dst[idx] = ({dtype})val;
    }}
}}
""".format(dtype=dtype_c)  # noqa: UP032


# Pre-generate sources for commonly used types.
# Additional types can be generated on-the-fly via _resample_kernel_source().
RESAMPLE_F64_SOURCE = _resample_kernel_source("double")
RESAMPLE_F32_SOURCE = _resample_kernel_source("float")
RESAMPLE_I16_SOURCE = _resample_kernel_source("short")
RESAMPLE_U8_SOURCE = _resample_kernel_source("unsigned char")

# Map numpy dtype to (kernel source, C type name) for dispatch.
_DTYPE_KERNEL_MAP: dict[str, tuple[str, str]] = {
    "float64": (RESAMPLE_F64_SOURCE, "double"),
    "float32": (RESAMPLE_F32_SOURCE, "float"),
    "int16": (RESAMPLE_I16_SOURCE, "short"),
    "uint8": (RESAMPLE_U8_SOURCE, "unsigned char"),
}


def get_resample_source(dtype_name: str) -> tuple[str, str]:
    """Return (kernel_source, c_type_name) for the given numpy dtype name.

    Falls back to float64 for unsupported types, which requires the
    caller to cast input data to float64 before launching the kernel.
    """
    if dtype_name in _DTYPE_KERNEL_MAP:
        return _DTYPE_KERNEL_MAP[dtype_name]
    # Fallback: generate at runtime (rare path)
    return RESAMPLE_F64_SOURCE, "double"
