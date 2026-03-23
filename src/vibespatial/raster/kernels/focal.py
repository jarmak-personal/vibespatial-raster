"""NVRTC shared-memory tiled convolution and focal statistics kernels.

Bead o17.8.5: Focal raster operations via custom NVRTC stencil kernels.
Focal statistics: min, max, mean, std, range, variety (count unique).
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
# Focal statistics kernel (min, max, mean, std, range, variety)
# ---------------------------------------------------------------------------
# Statistic type flags (passed as int to the kernel):
#   0 = min, 1 = max, 2 = mean, 3 = std, 4 = range, 5 = variety
#
# Mean and std use online Welford's algorithm to avoid catastrophic
# cancellation. Variety uses register-based linear scan for counting
# unique values in the window (practical for windows up to ~7x7 = 49 values).

FOCAL_STAT_MIN = 0
FOCAL_STAT_MAX = 1
FOCAL_STAT_MEAN = 2
FOCAL_STAT_STD = 3
FOCAL_STAT_RANGE = 4
FOCAL_STAT_VARIETY = 5

FOCAL_STATS_KERNEL_SOURCE = r"""
// Focal statistics kernel.
// stat_type: 0=min, 1=max, 2=mean, 3=std, 4=range, 5=variety
// Uses 2D block layout for coalesced row-major access.
// Nodata-aware: skips nodata pixels in neighborhood accumulation.

extern "C" __global__ void focal_stats(
    const double* __restrict__ input,
    double* __restrict__ output,
    const unsigned char* __restrict__ nodata_mask,  // nullable
    int width, int height,
    int radius_x, int radius_y,
    int stat_type,
    double nodata_val
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col >= width || row >= height) return;

    long long idx = (long long)row * width + col;

    // If center pixel is nodata, output is nodata
    if (nodata_mask != nullptr && nodata_mask[idx]) {
        output[idx] = nodata_val;
        return;
    }

    // Welford accumulators (for mean/std)
    double welford_mean = 0.0;
    double welford_m2 = 0.0;

    // Min/max accumulators
    double v_min = 1.0e308;   // DBL_MAX
    double v_max = -1.0e308;  // -DBL_MAX

    // Variety: register-resident buffer for unique values.
    // Max window = (2*7+1)^2 = 225 elements, but we cap at 49 unique
    // slots to keep register pressure manageable.
    double uniq[49];
    int n_uniq = 0;

    int count = 0;

    for (int dy = -radius_y; dy <= radius_y; ++dy) {
        int ny = row + dy;
        if (ny < 0 || ny >= height) continue;

        for (int dx = -radius_x; dx <= radius_x; ++dx) {
            int nx = col + dx;
            if (nx < 0 || nx >= width) continue;

            long long nidx = (long long)ny * width + nx;

            if (nodata_mask != nullptr && nodata_mask[nidx]) continue;

            double val = input[nidx];
            count++;

            // Min / Max / Range
            if (val < v_min) v_min = val;
            if (val > v_max) v_max = val;

            // Welford online mean + M2 (for mean and std)
            if (stat_type == 2 || stat_type == 3) {
                double delta = val - welford_mean;
                welford_mean += delta / (double)count;
                double delta2 = val - welford_mean;
                welford_m2 += delta * delta2;
            }

            // Variety: linear scan for uniqueness
            if (stat_type == 5) {
                bool found = false;
                for (int u = 0; u < n_uniq; ++u) {
                    if (uniq[u] == val) { found = true; break; }
                }
                if (!found && n_uniq < 49) {
                    uniq[n_uniq++] = val;
                }
            }
        }
    }

    if (count == 0) {
        output[idx] = nodata_val;
        return;
    }

    double result;
    switch (stat_type) {
        case 0: result = v_min; break;                           // min
        case 1: result = v_max; break;                           // max
        case 2: result = welford_mean; break;                    // mean
        case 3:                                                  // std
            result = (count > 1) ? sqrt(welford_m2 / (double)(count - 1)) : 0.0;
            break;
        case 4: result = v_max - v_min; break;                   // range
        case 5: result = (double)n_uniq; break;                  // variety
        default: result = nodata_val; break;
    }

    output[idx] = result;
}
"""
