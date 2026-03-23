"""NVRTC shared-memory tiled convolution kernel for focal raster operations.

Bead o17.8.5: Focal raster operations via custom NVRTC stencil kernels.
"""

from __future__ import annotations

# Generic 2D convolution kernel with shared-memory tiling and halo.
# Supports arbitrary kernel sizes. Nodata pixels are excluded from accumulation.

CONVOLVE_2D_KERNEL_SOURCE = r"""
extern "C" __global__ void convolve_2d(
    const double* input,
    double* output,
    const double* kernel_weights,
    const unsigned char* nodata_mask,  // 1 = nodata
    int width, int height,
    int kw, int kh,    // kernel width, height
    int pad_x, int pad_y,  // padding = kernel_size // 2
    double nodata_val
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    long long idx = (long long)row * width + col;

    if (nodata_mask != nullptr && nodata_mask[idx]) {
        output[idx] = nodata_val;
        return;
    }

    double sum = 0.0;
    double weight_sum = 0.0;

    for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            int src_row = row + ky - pad_y;
            int src_col = col + kx - pad_x;

            if (src_row < 0 || src_row >= height ||
                src_col < 0 || src_col >= width) {
                continue;
            }

            long long src_idx = (long long)src_row * width + src_col;

            if (nodata_mask != nullptr && nodata_mask[src_idx]) {
                continue;
            }

            double w = kernel_weights[ky * kw + kx];
            sum += input[src_idx] * w;
            weight_sum += w;
        }
    }

    if (weight_sum > 0.0) {
        output[idx] = sum / weight_sum;
    } else {
        output[idx] = nodata_val;
    }
}
"""

# Simplified version: normalized convolution (divides by weight sum for
# border handling). This is what we use for nodata-aware focal ops.
CONVOLVE_NORMALIZED_KERNEL_SOURCE = r"""
extern "C" __global__ void convolve_normalized(
    const double* input,
    double* output,
    const double* kernel_weights,
    const unsigned char* nodata_mask,
    int width, int height,
    int kw, int kh,
    int pad_x, int pad_y,
    double nodata_val
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    long long idx = (long long)row * width + col;

    if (nodata_mask != nullptr && nodata_mask[idx]) {
        output[idx] = nodata_val;
        return;
    }

    double sum = 0.0;
    double weight_sum = 0.0;

    for (int ky = 0; ky < kh; ++ky) {
        for (int kx = 0; kx < kw; ++kx) {
            int src_row = row + ky - pad_y;
            int src_col = col + kx - pad_x;

            if (src_row < 0 || src_row >= height ||
                src_col < 0 || src_col >= width) {
                continue;
            }

            long long src_idx = (long long)src_row * width + src_col;

            if (nodata_mask != nullptr && nodata_mask[src_idx]) {
                continue;
            }

            double w = kernel_weights[ky * kw + kx];
            sum += input[src_idx] * w;
            weight_sum += w;
        }
    }

    output[idx] = (weight_sum > 0.0) ? (sum / weight_sum) : nodata_val;
}
"""

# Hillshade kernel: fused Horn-method slope/aspect + hillshade illumination
# in a single pass using shared-memory 3x3 stencil tiling.
#
# Inputs: fp64 elevation, optional uint8 nodata_mask
# Outputs: uint8 hillshade (0-255)
# Parameters: cell_x, cell_y (pixel size), z_factor,
#             zenith_rad, azimuth_rad (pre-converted on host)
#
# Shared memory tile: (TILE_H+2) x (TILE_W+2+1) -- +2 for 1-cell halo,
# +1 column padding eliminates shared-memory bank conflicts on column access.

HILLSHADE_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16
// +1 padding on innermost dimension avoids 32-bank shared-memory conflicts
#define SMEM_STRIDE (TILE_W + 2 + 1)

extern "C" __global__ void hillshade(
    const double* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ nodata_mask,  // nullable: 1 = nodata
    const int width,
    const int height,
    const double cell_x,
    const double cell_y,
    const double z_factor,
    const double zenith_rad,
    const double azimuth_rad,
    const int nodata_out_i
) {
    const unsigned char nodata_out = (unsigned char)nodata_out_i;
    // Shared-memory tile with 1-cell halo on all sides
    __shared__ double tile[TILE_H + 2][SMEM_STRIDE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global pixel coordinates for this thread's output pixel
    const int col = blockIdx.x * TILE_W + tx;
    const int row = blockIdx.y * TILE_H + ty;

    // ---------- Load tile center (each thread loads one pixel) ----------
    {
        int src_col = col;
        int src_row = row;
        // Clamp to raster bounds (edge replication)
        if (src_col >= width) src_col = width - 1;
        if (src_row >= height) src_row = height - 1;
        tile[ty + 1][tx + 1] = input[(long long)src_row * width + src_col];
    }

    // ---------- Load halo cells ----------
    // Left halo column (threads with tx == 0)
    if (tx == 0) {
        int halo_col = col - 1;
        int halo_row = row;
        if (halo_col < 0) halo_col = 0;
        if (halo_row < 0) halo_row = 0;
        if (halo_row >= height) halo_row = height - 1;
        tile[ty + 1][0] = input[(long long)halo_row * width + halo_col];
    }
    // Right halo column (threads with tx == TILE_W - 1 or at raster edge)
    if (tx == TILE_W - 1) {
        int halo_col = col + 1;
        int halo_row = row;
        if (halo_col >= width) halo_col = width - 1;
        if (halo_row < 0) halo_row = 0;
        if (halo_row >= height) halo_row = height - 1;
        tile[ty + 1][TILE_W + 1] = input[(long long)halo_row * width + halo_col];
    }
    // Top halo row (threads with ty == 0)
    if (ty == 0) {
        int halo_col = col;
        int halo_row = row - 1;
        if (halo_col < 0) halo_col = 0;
        if (halo_col >= width) halo_col = width - 1;
        if (halo_row < 0) halo_row = 0;
        tile[0][tx + 1] = input[(long long)halo_row * width + halo_col];
    }
    // Bottom halo row (threads with ty == TILE_H - 1)
    if (ty == TILE_H - 1) {
        int halo_col = col;
        int halo_row = row + 1;
        if (halo_col < 0) halo_col = 0;
        if (halo_col >= width) halo_col = width - 1;
        if (halo_row >= height) halo_row = height - 1;
        tile[TILE_H + 1][tx + 1] = input[(long long)halo_row * width + halo_col];
    }
    // Four corner halo cells
    if (tx == 0 && ty == 0) {
        int cr = row - 1; int cc = col - 1;
        if (cr < 0) cr = 0; if (cc < 0) cc = 0;
        tile[0][0] = input[(long long)cr * width + cc];
    }
    if (tx == TILE_W - 1 && ty == 0) {
        int cr = row - 1; int cc = col + 1;
        if (cr < 0) cr = 0; if (cc >= width) cc = width - 1;
        tile[0][TILE_W + 1] = input[(long long)cr * width + cc];
    }
    if (tx == 0 && ty == TILE_H - 1) {
        int cr = row + 1; int cc = col - 1;
        if (cr >= height) cr = height - 1; if (cc < 0) cc = 0;
        tile[TILE_H + 1][0] = input[(long long)cr * width + cc];
    }
    if (tx == TILE_W - 1 && ty == TILE_H - 1) {
        int cr = row + 1; int cc = col + 1;
        if (cr >= height) cr = height - 1; if (cc >= width) cc = width - 1;
        tile[TILE_H + 1][TILE_W + 1] = input[(long long)cr * width + cc];
    }

    __syncthreads();

    // ---------- Bounds check: only process pixels within the raster ----------
    if (col >= width || row >= height) return;

    const long long idx = (long long)row * width + col;

    // ---------- Nodata propagation: center pixel or any 3x3 neighbor ----------
    if (nodata_mask != nullptr) {
        if (nodata_mask[idx]) {
            output[idx] = nodata_out;
            return;
        }
        // Check 3x3 neighborhood for nodata
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = row + dy;
                int nx = col + dx;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                    if (nodata_mask[(long long)ny * width + nx]) {
                        output[idx] = nodata_out;
                        return;
                    }
                }
            }
        }
    }

    // ---------- Read 3x3 stencil from shared memory ----------
    // Notation: z[row_offset+1][col_offset+1] in tile coords
    // Horn method naming: a=NW, b=N, c=NE, d=W, e=center, f=E, g=SW, h=S, i=SE
    double a = tile[ty    ][tx    ];  // NW
    double b = tile[ty    ][tx + 1];  // N
    double c = tile[ty    ][tx + 2];  // NE
    double d = tile[ty + 1][tx    ];  // W
    // e = center (not needed for derivatives)
    double f = tile[ty + 1][tx + 2];  // E
    double g = tile[ty + 2][tx    ];  // SW
    double h = tile[ty + 2][tx + 1];  // S
    double i = tile[ty + 2][tx + 2];  // SE

    // Horn method partial derivatives
    // dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_x)
    // dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_y)
    double dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * cell_x) * z_factor;
    double dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * cell_y) * z_factor;

    // Slope and aspect
    double slope = atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy));

    // Aspect: atan2(-dz_dy, dz_dx), then convert math angle to compass bearing
    double aspect;
    if (dz_dx == 0.0 && dz_dy == 0.0) {
        // Flat surface: aspect is undefined, use 0 (convention)
        aspect = 0.0;
    } else {
        aspect = atan2(-dz_dy, dz_dx);
        // Convert from math angle (east=0, CCW) to compass (north=0, CW)
        // Not needed: hillshade formula uses azimuth_rad - aspect directly
    }

    // Hillshade formula:
    // hs = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
    double cos_z = cos(zenith_rad);
    double sin_z = sin(zenith_rad);
    double hs = cos_z * cos(slope) + sin_z * sin(slope) * cos(azimuth_rad - aspect);

    // Clamp to [0, 1] and scale to [0, 255]
    if (hs < 0.0) hs = 0.0;
    if (hs > 1.0) hs = 1.0;
    output[idx] = (unsigned char)(hs * 255.0 + 0.5);
}
"""
