"""NVRTC kernel sources for GPU Euclidean Distance Transform via Jump Flooding.

Algorithm: Jump Flooding Algorithm (JFA) for Euclidean Distance Transform.
1. jfa_init: Foreground pixels store their own coordinates as seeds,
   background pixels store sentinel (-1, -1).
2. jfa_step: For step size k, each pixel checks 8 neighbors at distance k
   and keeps the nearest seed (by squared Euclidean distance).
   Steps: k = N/2, N/4, ..., 2, 1 where N = next power of 2 >= max(H, W).
3. distance_compute: Final pass converts seed coordinates to Euclidean
   distances (sqrt of squared distance to nearest seed).

O(log N) iterations, each is embarrassingly parallel. No convergence
flag needed -- the number of iterations is fixed.

Seed coordinates are stored as two separate int arrays (SoA layout) for
coalesced memory access: seed_x[i] and seed_y[i].
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 1: Initialize seed buffer
# ---------------------------------------------------------------------------

JFA_INIT_SOURCE = r"""
extern "C" __global__
void jfa_init(
    const unsigned char* __restrict__ foreground,
    int* __restrict__ seed_x,
    int* __restrict__ seed_y,
    const int width,
    const int height
) {
    const int n = width * height;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride)
    {
        if (foreground[idx]) {
            int col = idx % width;
            int row = idx / width;
            seed_x[idx] = col;
            seed_y[idx] = row;
        } else {
            seed_x[idx] = -1;
            seed_y[idx] = -1;
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 2: JFA step -- check 8 neighbors at distance k
# ---------------------------------------------------------------------------

JFA_STEP_SOURCE = r"""
extern "C" __global__
void jfa_step(
    const int* __restrict__ seed_x_in,
    const int* __restrict__ seed_y_in,
    int* __restrict__ seed_x_out,
    int* __restrict__ seed_y_out,
    const int width,
    const int height,
    const int step_k
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;

    /* Current best seed */
    int best_sx = seed_x_in[idx];
    int best_sy = seed_y_in[idx];
    long long best_dist2;
    if (best_sx < 0) {
        best_dist2 = (long long)width * width + (long long)height * height + 1;
    } else {
        long long ddx = (long long)(col - best_sx);
        long long ddy = (long long)(row - best_sy);
        best_dist2 = ddx * ddx + ddy * ddy;
    }

    /* Check 8 neighbors at distance step_k (plus self for robustness) */
    #pragma unroll
    for (int dy = -1; dy <= 1; dy++) {
        #pragma unroll
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int ny = row + dy * step_k;
            int nx = col + dx * step_k;
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

            int nidx = ny * width + nx;
            int nsx = seed_x_in[nidx];
            int nsy = seed_y_in[nidx];
            if (nsx < 0) continue;  /* neighbor has no seed */

            long long ndx = (long long)(col - nsx);
            long long ndy = (long long)(row - nsy);
            long long nd2 = ndx * ndx + ndy * ndy;
            if (nd2 < best_dist2) {
                best_dist2 = nd2;
                best_sx = nsx;
                best_sy = nsy;
            }
        }
    }

    seed_x_out[idx] = best_sx;
    seed_y_out[idx] = best_sy;
}
"""

# ---------------------------------------------------------------------------
# Phase 3: Compute final Euclidean distance from seed coordinates
# ---------------------------------------------------------------------------

DISTANCE_COMPUTE_SOURCE = r"""
extern "C" __global__
void distance_compute(
    const int* __restrict__ seed_x,
    const int* __restrict__ seed_y,
    double* __restrict__ distance_out,
    const unsigned char* __restrict__ nodata_mask,
    const double nodata_value,
    const int width,
    const int height
) {
    const int n = width * height;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride)
    {
        /* Nodata pixels get nodata sentinel */
        if (nodata_mask != nullptr && nodata_mask[idx]) {
            distance_out[idx] = nodata_value;
            continue;
        }

        int sx = seed_x[idx];
        int sy = seed_y[idx];
        if (sx < 0) {
            /* No seed found -- entire raster is background */
            distance_out[idx] = nodata_value;
        } else {
            int col = idx % width;
            int row = idx / width;
            double ddx = (double)(col - sx);
            double ddy = (double)(row - sy);
            distance_out[idx] = sqrt(ddx * ddx + ddy * ddy);
        }
    }
}
"""

# Kernel names tuple for compilation
JFA_INIT_NAMES = ("jfa_init",)
JFA_STEP_NAMES = ("jfa_step",)
DISTANCE_COMPUTE_NAMES = ("distance_compute",)
