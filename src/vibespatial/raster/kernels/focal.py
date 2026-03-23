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


# ---------------------------------------------------------------------------
# Terrain derivatives kernel: TRI, TPI, and curvature.
#
# Uses a shared-memory 3x3 tile with 1-cell halo (TILE+2 x TILE+2) for
# coalesced neighbor access. The derivative type is selected via an integer
# flag passed as a kernel parameter:
#   0 = TRI  (Terrain Ruggedness Index — mean |center - neighbor|)
#   1 = TPI  (Topographic Position Index — center - mean(neighbors))
#   2 = Profile curvature (Zevenbergen & Thorne 1987)
#
# Nodata propagation: if the center pixel or ANY of its 8 neighbors is nodata,
# the output pixel is nodata.
#
# Shared memory layout uses +1 column padding to avoid bank conflicts.
# ---------------------------------------------------------------------------

TERRAIN_DERIVATIVES_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16
#define DERIV_TRI  0
#define DERIV_TPI  1
#define DERIV_CURV 2

extern "C" __global__ void terrain_derivatives(
    const double* __restrict__ input,
    double* __restrict__ output,
    const unsigned char* __restrict__ nodata_mask,
    const int width,
    const int height,
    const double cellsize_x,
    const double cellsize_y,
    const double nodata_val,
    const int deriv_type
) {
    // +1 padding on columns to eliminate shared-memory bank conflicts
    __shared__ double tile[TILE_H + 2][TILE_W + 3];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * TILE_W + tx;
    const int row = blockIdx.y * TILE_H + ty;

    // ------------------------------------------------------------------
    // Load center of tile
    // ------------------------------------------------------------------
    double center_val = 0.0;
    if (col < width && row < height) {
        center_val = input[(long long)row * width + col];
    }
    tile[ty + 1][tx + 1] = center_val;

    // ------------------------------------------------------------------
    // Load halo cells (border threads load the extra ring)
    // ------------------------------------------------------------------
    // Left halo
    if (tx == 0) {
        int hcol = col - 1;
        int hrow = row;
        tile[ty + 1][0] = (hcol >= 0 && hrow >= 0 && hrow < height)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    // Right halo
    if (tx == TILE_W - 1 || col == width - 1) {
        int hcol = col + 1;
        int hrow = row;
        tile[ty + 1][tx + 2] = (hcol < width && hrow >= 0 && hrow < height)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    // Top halo
    if (ty == 0) {
        int hcol = col;
        int hrow = row - 1;
        tile[0][tx + 1] = (hrow >= 0 && hcol >= 0 && hcol < width)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    // Bottom halo
    if (ty == TILE_H - 1 || row == height - 1) {
        int hcol = col;
        int hrow = row + 1;
        tile[ty + 2][tx + 1] = (hrow < height && hcol >= 0 && hcol < width)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    // Four corners
    if (tx == 0 && ty == 0) {
        int hcol = col - 1; int hrow = row - 1;
        tile[0][0] = (hcol >= 0 && hrow >= 0)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    if ((tx == TILE_W - 1 || col == width - 1) && ty == 0) {
        int hcol = col + 1; int hrow = row - 1;
        tile[0][tx + 2] = (hcol < width && hrow >= 0)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    if (tx == 0 && (ty == TILE_H - 1 || row == height - 1)) {
        int hcol = col - 1; int hrow = row + 1;
        tile[ty + 2][0] = (hcol >= 0 && hrow < height)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }
    if ((tx == TILE_W - 1 || col == width - 1) &&
        (ty == TILE_H - 1 || row == height - 1)) {
        int hcol = col + 1; int hrow = row + 1;
        tile[ty + 2][tx + 2] = (hcol < width && hrow < height)
            ? input[(long long)hrow * width + hcol] : 0.0;
    }

    __syncthreads();

    // ------------------------------------------------------------------
    // Bounds check — only interior pixels produce output
    // ------------------------------------------------------------------
    if (col >= width || row >= height) return;

    const long long idx = (long long)row * width + col;

    // Nodata check: center pixel
    if (nodata_mask != nullptr && nodata_mask[idx]) {
        output[idx] = nodata_val;
        return;
    }

    // Nodata check: any of the 8 neighbors. If center is on the border,
    // we also mark as nodata (no valid 3x3 window).
    if (row == 0 || row == height - 1 || col == 0 || col == width - 1) {
        output[idx] = nodata_val;
        return;
    }

    if (nodata_mask != nullptr) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dy == 0 && dx == 0) continue;
                long long nidx = (long long)(row + dy) * width + (col + dx);
                if (nodata_mask[nidx]) {
                    output[idx] = nodata_val;
                    return;
                }
            }
        }
    }

    // Read 3x3 neighborhood from shared memory
    //   z0 z1 z2
    //   z3 z4 z5
    //   z6 z7 z8
    const double z0 = tile[ty    ][tx    ];
    const double z1 = tile[ty    ][tx + 1];
    const double z2 = tile[ty    ][tx + 2];
    const double z3 = tile[ty + 1][tx    ];
    const double z4 = tile[ty + 1][tx + 1];  // center
    const double z5 = tile[ty + 1][tx + 2];
    const double z6 = tile[ty + 2][tx    ];
    const double z7 = tile[ty + 2][tx + 1];
    const double z8 = tile[ty + 2][tx + 2];

    double result;

    if (deriv_type == DERIV_TRI) {
        // TRI: mean absolute difference between center and 8 neighbors
        // Riley et al. (1999)
        result = (fabs(z0 - z4) + fabs(z1 - z4) + fabs(z2 - z4) +
                  fabs(z3 - z4) +                  fabs(z5 - z4) +
                  fabs(z6 - z4) + fabs(z7 - z4) + fabs(z8 - z4)) / 8.0;
    }
    else if (deriv_type == DERIV_TPI) {
        // TPI: center elevation minus mean of 8 neighbors
        double neighbor_mean = (z0 + z1 + z2 + z3 + z5 + z6 + z7 + z8) / 8.0;
        result = z4 - neighbor_mean;
    }
    else {
        // Profile curvature — Zevenbergen & Thorne (1987)
        // Second-order finite differences from the 3x3 window
        double D = ((z3 + z5) / 2.0 - z4) / (cellsize_x * cellsize_x);
        double E = ((z1 + z7) / 2.0 - z4) / (cellsize_y * cellsize_y);
        // Profile curvature = -2(D + E) * 100  (standard scaling)
        result = -2.0 * (D + E) * 100.0;
    }

    output[idx] = result;
}
"""
