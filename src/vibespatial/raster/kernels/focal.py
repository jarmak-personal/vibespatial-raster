"""NVRTC shared-memory tiled convolution and focal statistics kernels.

Bead o17.8.5: Focal raster operations via custom NVRTC stencil kernels.

Kernels use shared-memory tiling with halo cells for data reuse.
Kernel weights are loaded into shared memory once per block.
Nodata masking uses predicated writes (no early-return divergence).
All read-only pointers use const __restrict__ for compiler optimization.
Focal statistics: min, max, mean, std, range, variety (count unique).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Shared-memory tiled 2D normalized convolution (nodata-aware)
# ---------------------------------------------------------------------------
# This kernel handles arbitrary kernel sizes up to 25x25. Each thread block
# loads a tile of input data plus halo cells into shared memory. Kernel
# weights and the nodata mask tile are also loaded into shared memory.
#
# Design decisions:
#   - Tile = 16x16 threads, halo = pad_y rows top/bottom, pad_x cols left/right
#   - Shared memory uses +1 column padding to avoid bank conflicts
#   - Nodata mask is loaded into a second shared memory region alongside the
#     data tile (uint8, 1/8 the cost of the double data tile) so that the
#     inner KxK loop reads nodata from shared memory, not global memory
#   - Nodata is handled with predicated accumulation (no branch divergence)
#   - Border pixels use clamped (edge-replicated) access instead of skip-if-OOB
#   - const __restrict__ on all read-only pointer parameters
#   - Double precision throughout (rasters are upcast to float64 before launch)

CONVOLVE_NORMALIZED_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16

extern "C" __global__
void convolve_normalized(
    const double* __restrict__ input,
    double* __restrict__ output,
    const double* __restrict__ kernel_weights,
    const unsigned char* __restrict__ nodata_mask,
    const int width,
    const int height,
    const int kw,
    const int kh,
    const int pad_x,
    const int pad_y,
    const double nodata_val
) {
    /* --- Shared memory layout ---
     *   [0                       .. data_tile_end)        : double data tile
     *   [data_tile_end           .. kweights_end)         : double kernel weights
     *   [kweights_end            .. nodata_tile_end)      : uint8  nodata tile
     *
     * The nodata tile uses the raw (non-padded) tile dimensions since it
     * does not need bank-conflict padding (uint8 elements, 1 byte each).
     */
    extern __shared__ char _smem[];
    const int smem_cols = TILE_W + 2 * pad_x + 1;  /* +1 bank padding */
    const int smem_tile_size = (TILE_H + 2 * pad_y) * smem_cols;
    double* tile = (double*)_smem;
    double* kw_shared = tile + smem_tile_size;

    const int ksize = kw * kh;
    /* nodata tile starts after kernel weights, aligned to 8 bytes */
    const int tile_rows = TILE_H + 2 * pad_y;
    const int tile_cols = TILE_W + 2 * pad_x;  /* no +1 padding for uint8 */
    const int tile_total = tile_rows * tile_cols;
    unsigned char* nodata_tile = (unsigned char*)(kw_shared + ksize);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TILE_W + tx;
    const int block_threads = TILE_W * TILE_H;

    /* --- Load kernel weights into shared memory (cooperative) --- */
    for (int i = tid; i < ksize; i += block_threads) {
        kw_shared[i] = kernel_weights[i];
    }

    /* --- Global coordinates of this thread's output pixel --- */
    const int gx = blockIdx.x * TILE_W + tx;
    const int gy = blockIdx.y * TILE_H + ty;

    /* --- Load data tile + nodata tile into shared memory --- */
    for (int i = tid; i < tile_total; i += block_threads) {
        int lr = i / tile_cols;
        int lc = i % tile_cols;

        int src_row = (int)blockIdx.y * TILE_H + lr - pad_y;
        int src_col = (int)blockIdx.x * TILE_W + lc - pad_x;

        src_row = max(0, min(src_row, height - 1));
        src_col = max(0, min(src_col, width - 1));

        long long src_gidx = (long long)src_row * width + src_col;
        tile[lr * smem_cols + lc] = input[src_gidx];
        nodata_tile[lr * tile_cols + lc] =
            (nodata_mask != nullptr) ? nodata_mask[src_gidx] : 0;
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    const long long out_idx = (long long)gy * width + gx;

    /* Center pixel nodata check via shared memory */
    if (nodata_tile[(ty + pad_y) * tile_cols + (tx + pad_x)]) {
        output[out_idx] = nodata_val;
        return;
    }

    double sum = 0.0;
    double wsum = 0.0;

    const int tile_base_row = ty;
    const int tile_base_col = tx;

    for (int ky_i = 0; ky_i < kh; ++ky_i) {
        for (int kx_i = 0; kx_i < kw; ++kx_i) {
            int src_lr = tile_base_row + ky_i;
            int src_lc = tile_base_col + kx_i;

            int orig_row = (int)blockIdx.y * TILE_H + src_lr - pad_y;
            int orig_col = (int)blockIdx.x * TILE_W + src_lc - pad_x;

            if (orig_row < 0 || orig_row >= height ||
                orig_col < 0 || orig_col >= width) {
                continue;
            }

            /* Read nodata flag from shared memory (not global) */
            int is_nodata = (int)nodata_tile[src_lr * tile_cols + src_lc];

            double w = kw_shared[ky_i * kw + kx_i];
            double val = tile[src_lr * smem_cols + src_lc];

            double valid = (double)(1 - is_nodata);
            sum += val * w * valid;
            wsum += w * valid;
        }
    }

    output[out_idx] = (wsum > 0.0) ? (sum / wsum) : nodata_val;
}
"""


# ---------------------------------------------------------------------------
# Fused slope+aspect kernel (Horn method, 3x3 stencil)
# ---------------------------------------------------------------------------
# Computes both slope (degrees) and aspect (compass degrees, north=0 CW)
# in a single pass over the DEM. Uses shared-memory tiling with 1-pixel halo.
# This avoids the double allocation + double pass of separate slope/aspect.

SLOPE_ASPECT_KERNEL_SOURCE = r"""
#define TILE_W 16
#define TILE_H 16

/* Safe global memory fetch: returns 0.0 for out-of-bounds coordinates. */
__device__ __forceinline__
double safe_fetch_d(
    const double* __restrict__ data,
    int x, int y, int width, int height
) {
    if (x >= 0 && x < width && y >= 0 && y < height)
        return data[(long long)y * width + x];
    return 0.0;
}

extern "C" __global__
void slope_aspect(
    const double* __restrict__ dem,
    double* __restrict__ slope_out,
    double* __restrict__ aspect_out,
    const unsigned char* __restrict__ nodata_mask,
    const int width,
    const int height,
    const double cell_x,
    const double cell_y,
    const double nodata_val,
    const int compute_slope,
    const int compute_aspect
) {
    /* Shared memory tile with 1-pixel halo + bank-conflict padding */
    __shared__ double tile[TILE_H + 2][TILE_W + 2 + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    /* Tile origin: top-left corner of the center (non-halo) region */
    const int tile_ox = blockIdx.x * TILE_W;
    const int tile_oy = blockIdx.y * TILE_H;
    const int gx = tile_ox + tx;
    const int gy = tile_oy + ty;

    /* ---- Load center cell ---- */
    tile[ty + 1][tx + 1] = safe_fetch_d(dem, gx, gy, width, height);

    /* ---- Load halo edges (4 sides) at FIXED tile-edge positions ---- */
    /* Left halo column (shared col 0): loaded by tx == 0 */
    if (tx == 0) {
        tile[ty + 1][0] = safe_fetch_d(dem, tile_ox - 1, gy, width, height);
    }
    /* Right halo column (shared col TILE_W + 1): loaded by tx == TILE_W - 1 */
    if (tx == TILE_W - 1) {
        tile[ty + 1][TILE_W + 1] = safe_fetch_d(
            dem, tile_ox + TILE_W, gy, width, height);
    }
    /* Top halo row (shared row 0): loaded by ty == 0 */
    if (ty == 0) {
        tile[0][tx + 1] = safe_fetch_d(dem, gx, tile_oy - 1, width, height);
    }
    /* Bottom halo row (shared row TILE_H + 1): loaded by ty == TILE_H - 1 */
    if (ty == TILE_H - 1) {
        tile[TILE_H + 1][tx + 1] = safe_fetch_d(
            dem, gx, tile_oy + TILE_H, width, height);
    }

    /* ---- Load 4 corner halo cells ---- */
    if (tx == 0 && ty == 0) {
        tile[0][0] = safe_fetch_d(
            dem, tile_ox - 1, tile_oy - 1, width, height);
    }
    if (tx == TILE_W - 1 && ty == 0) {
        tile[0][TILE_W + 1] = safe_fetch_d(
            dem, tile_ox + TILE_W, tile_oy - 1, width, height);
    }
    if (tx == 0 && ty == TILE_H - 1) {
        tile[TILE_H + 1][0] = safe_fetch_d(
            dem, tile_ox - 1, tile_oy + TILE_H, width, height);
    }
    if (tx == TILE_W - 1 && ty == TILE_H - 1) {
        tile[TILE_H + 1][TILE_W + 1] = safe_fetch_d(
            dem, tile_ox + TILE_W, tile_oy + TILE_H, width, height);
    }

    __syncthreads();

    if (gx >= width || gy >= height) return;

    const long long idx = (long long)gy * width + gx;

    /* Nodata check */
    if (nodata_mask != nullptr && nodata_mask[idx]) {
        if (compute_slope) slope_out[idx] = nodata_val;
        if (compute_aspect) aspect_out[idx] = nodata_val;
        return;
    }

    /* Read 3x3 neighborhood from shared memory */
    /* Naming: z_RC where R=row(t=top,m=mid,b=bot), C=col(l=left,m=mid,r=right) */
    double z_tl = tile[ty    ][tx    ];
    double z_tm = tile[ty    ][tx + 1];
    double z_tr = tile[ty    ][tx + 2];
    double z_ml = tile[ty + 1][tx    ];
    /* z_mm = tile[ty+1][tx+1] = center_val (not needed for gradient) */
    double z_mr = tile[ty + 1][tx + 2];
    double z_bl = tile[ty + 2][tx    ];
    double z_bm = tile[ty + 2][tx + 1];
    double z_br = tile[ty + 2][tx + 2];

    /* Horn method partial derivatives */
    double dz_dx = ((z_tr + 2.0 * z_mr + z_br) - (z_tl + 2.0 * z_ml + z_bl))
                   / (8.0 * cell_x);
    double dz_dy = ((z_bl + 2.0 * z_bm + z_br) - (z_tl + 2.0 * z_tm + z_tr))
                   / (8.0 * cell_y);

    if (compute_slope) {
        double slope_rad = atan(sqrt(dz_dx * dz_dx + dz_dy * dz_dy));
        slope_out[idx] = slope_rad * (180.0 / 3.14159265358979323846);
    }

    if (compute_aspect) {
        double aspect_rad = atan2(-dz_dy, dz_dx);
        double aspect_deg = aspect_rad * (180.0 / 3.14159265358979323846);
        /* Convert from math angle (east=0, CCW) to compass (north=0, CW) */
        aspect_deg = fmod(90.0 - aspect_deg, 360.0);
        if (aspect_deg < 0.0) aspect_deg += 360.0;
        aspect_out[idx] = aspect_deg;
    }
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
