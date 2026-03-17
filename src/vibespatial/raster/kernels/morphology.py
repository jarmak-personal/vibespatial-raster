"""NVRTC shared-memory tiled morphology kernels for binary erode/dilate.

Bead: GPU morphology stencil kernels (erode/dilate) with shared memory halo.
Uses 3x3 structuring elements with (16,16) thread blocks and 1-pixel halo.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Binary erosion kernel -- shared memory tile with 1-pixel halo
# ---------------------------------------------------------------------------

BINARY_ERODE_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16

extern "C" __global__
void binary_erode(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ selem,  /* 9 elements, row-major 3x3 */
    const int width,
    const int height
) {
    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    /* Load center of tile */
    unsigned char center_val = (gx < width && gy < height)
                                   ? input[gy * width + gx] : 0;
    tile[ty + 1][tx + 1] = center_val;

    /* Load left halo */
    if (tx == 0) {
        tile[ty + 1][0] = (gx > 0 && gy < height)
                              ? input[gy * width + (gx - 1)] : 0;
    }
    /* Load right halo */
    if (tx == TILE_W - 1 || gx == width - 1) {
        tile[ty + 1][tx + 2] = (gx + 1 < width && gy < height)
                                    ? input[gy * width + (gx + 1)] : 0;
    }
    /* Load top halo */
    if (ty == 0) {
        tile[0][tx + 1] = (gy > 0 && gx < width)
                              ? input[(gy - 1) * width + gx] : 0;
    }
    /* Load bottom halo */
    if (ty == TILE_H - 1 || gy == height - 1) {
        tile[ty + 2][tx + 1] = (gy + 1 < height && gx < width)
                                    ? input[(gy + 1) * width + gx] : 0;
    }
    /* Load corner halos */
    if (tx == 0 && ty == 0) {
        tile[0][0] = (gx > 0 && gy > 0)
                         ? input[(gy - 1) * width + (gx - 1)] : 0;
    }
    if ((tx == TILE_W - 1 || gx == width - 1) && ty == 0) {
        tile[0][tx + 2] = (gx + 1 < width && gy > 0)
                              ? input[(gy - 1) * width + (gx + 1)] : 0;
    }
    if (tx == 0 && (ty == TILE_H - 1 || gy == height - 1)) {
        tile[ty + 2][0] = (gx > 0 && gy + 1 < height)
                              ? input[(gy + 1) * width + (gx - 1)] : 0;
    }
    if ((tx == TILE_W - 1 || gx == width - 1) &&
        (ty == TILE_H - 1 || gy == height - 1)) {
        tile[ty + 2][tx + 2] = (gx + 1 < width && gy + 1 < height)
                                    ? input[(gy + 1) * width + (gx + 1)] : 0;
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    /* Erode: output 1 only if ALL structuring-element neighbors are 1 */
    unsigned char result = 1;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (selem[(dy + 1) * 3 + (dx + 1)]) {
                if (!tile[ty + 1 + dy][tx + 1 + dx]) {
                    result = 0;
                }
            }
        }
    }
    output[gy * width + gx] = result;
}
"""

# ---------------------------------------------------------------------------
# Binary dilation kernel -- shared memory tile with 1-pixel halo
# ---------------------------------------------------------------------------

BINARY_DILATE_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16

extern "C" __global__
void binary_dilate(
    const unsigned char* __restrict__ input,
    unsigned char* __restrict__ output,
    const unsigned char* __restrict__ selem,  /* 9 elements, row-major 3x3 */
    const int width,
    const int height
) {
    __shared__ unsigned char tile[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    /* Load center of tile */
    unsigned char center_val = (gx < width && gy < height)
                                   ? input[gy * width + gx] : 0;
    tile[ty + 1][tx + 1] = center_val;

    /* Load left halo */
    if (tx == 0) {
        tile[ty + 1][0] = (gx > 0 && gy < height)
                              ? input[gy * width + (gx - 1)] : 0;
    }
    /* Load right halo */
    if (tx == TILE_W - 1 || gx == width - 1) {
        tile[ty + 1][tx + 2] = (gx + 1 < width && gy < height)
                                    ? input[gy * width + (gx + 1)] : 0;
    }
    /* Load top halo */
    if (ty == 0) {
        tile[0][tx + 1] = (gy > 0 && gx < width)
                              ? input[(gy - 1) * width + gx] : 0;
    }
    /* Load bottom halo */
    if (ty == TILE_H - 1 || gy == height - 1) {
        tile[ty + 2][tx + 1] = (gy + 1 < height && gx < width)
                                    ? input[(gy + 1) * width + gx] : 0;
    }
    /* Load corner halos */
    if (tx == 0 && ty == 0) {
        tile[0][0] = (gx > 0 && gy > 0)
                         ? input[(gy - 1) * width + (gx - 1)] : 0;
    }
    if ((tx == TILE_W - 1 || gx == width - 1) && ty == 0) {
        tile[0][tx + 2] = (gx + 1 < width && gy > 0)
                              ? input[(gy - 1) * width + (gx + 1)] : 0;
    }
    if (tx == 0 && (ty == TILE_H - 1 || gy == height - 1)) {
        tile[ty + 2][0] = (gx > 0 && gy + 1 < height)
                              ? input[(gy + 1) * width + (gx - 1)] : 0;
    }
    if ((tx == TILE_W - 1 || gx == width - 1) &&
        (ty == TILE_H - 1 || gy == height - 1)) {
        tile[ty + 2][tx + 2] = (gx + 1 < width && gy + 1 < height)
                                    ? input[(gy + 1) * width + (gx + 1)] : 0;
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    /* Dilate: output 1 if ANY structuring-element neighbor is 1 */
    unsigned char result = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (selem[(dy + 1) * 3 + (dx + 1)]) {
                if (tile[ty + 1 + dy][tx + 1 + dx]) {
                    result = 1;
                }
            }
        }
    }
    output[gy * width + gx] = result;
}
"""
