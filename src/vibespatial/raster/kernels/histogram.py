"""NVRTC kernel source for histogram equalization remap.

The remap kernel reads a per-pixel value, looks up its CDF-based
equalized output in a precomputed lookup table, and writes the result.
Used by raster_histogram_equalize to redistribute pixel values to a
roughly uniform distribution in [0, 255].
"""

from __future__ import annotations

HISTOGRAM_REMAP_KERNEL_SOURCE = r"""
extern "C" __global__
void histogram_remap(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ lut,
    const unsigned char* __restrict__ nodata_mask,
    const int n,
    const int nodata_val
) {
    const unsigned char nd = (unsigned char)nodata_val;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        if (nodata_mask != nullptr && nodata_mask[idx]) {
            output[idx] = nd;
        } else {
            output[idx] = lut[input[idx]];
        }
    }
}
"""
