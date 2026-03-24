"""NVRTC kernel sources for marching-squares GPU polygonize.

Kernels:
  classify_cells — binary-classify each 2x2 cell for a given target value
  edge_count     — count edges per non-trivial cell (for prefix-sum offsets)
  emit_edges     — emit directed edge segments for non-trivial cells

The 16 cell classification indices encode which of the 4 corners (TL, TR, BR, BL)
match the target value.  Index 0 = no corners match, index 15 = all corners match.
Non-trivial cells (1..14) produce 1 or 2 boundary edge segments.

Edge segments are directed so that the *interior* (matching region) is always
on the RIGHT side of the edge.  This winding convention allows the host-side
ring chainer to assemble correctly-oriented polygon rings.

Multi-value support: the dispatch code iterates over each unique raster value,
running classify_cells once per value with that value as the target.  This
produces a complete closed ring for every distinct region, regardless of how
many different values adjoin a given cell.

Optimizations (o18.3.22):
  - Grid-stride loops with ILP=4 on all kernels
  - const __restrict__ on all read-only pointer parameters
  - __constant__ edge lookup tables (emit_edges, edge_count)
  - Precomputed affine transform scalars passed as kernel params
  - Branchless nodata classification via bitwise ops
  - Float literal precision qualifiers (not applicable: raster data is f64)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# classify_cells kernel (binary, per-target-value)
# ---------------------------------------------------------------------------
# Grid-stride loop with ILP=4.  One logical work item per 2x2 cell.
# Cell (cx, cy) covers raster pixels:
#   TL = (cy, cx)      TR = (cy, cx+1)
#   BL = (cy+1, cx)    BR = (cy+1, cx+1)
#
# cell_width  = raster_width  - 1
# cell_height = raster_height - 1
# total_cells = cell_width * cell_height
#
# Parameters:
#   target_value — the raster value to treat as "inside" for this pass
#
# Outputs:
#   d_cell_class[cell_idx]  — 4-bit classification (0..15)
#
# The classification bit-pattern is:
#   bit 3 (8)  = TL matches target
#   bit 2 (4)  = TR matches target
#   bit 1 (2)  = BR matches target
#   bit 0 (1)  = BL matches target
#
# "Matches" means the pixel value equals target_value AND is not nodata.

CLASSIFY_CELLS_KERNEL_SOURCE = r"""
extern "C" __global__ void classify_cells(
    const double* __restrict__ raster,
    const unsigned char* __restrict__ nodata_mask,
    int* __restrict__ cell_class,
    const int raster_width,
    const int cell_width,
    const int cell_height,
    const double target_value
) {
    const long long total_cells = (long long)cell_width * cell_height;
    const long long stride = blockDim.x * gridDim.x;
    const int ILP = 4;

    for (long long base = blockIdx.x * blockDim.x + threadIdx.x;
         base < total_cells;
         base += stride * ILP) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            const long long idx = base + (long long)j * stride;
            if (idx >= total_cells) break;

            const long long cy = idx / cell_width;
            const long long cx = idx - cy * cell_width;

            // Pixel indices for the 4 corners
            const long long row0 = cy * raster_width;
            const long long row1 = row0 + raster_width;
            const long long tl = row0 + cx;
            const long long tr = row0 + cx + 1;
            const long long bl = row1 + cx;
            const long long br = row1 + cx + 1;

            const double v_tl = raster[tl];
            const double v_tr = raster[tr];
            const double v_bl = raster[bl];
            const double v_br = raster[br];

            // Nodata flags: branchless load (0 when mask is null)
            const int has_mask = (nodata_mask != nullptr);
            const int nd_tl = has_mask ? (int)nodata_mask[tl] : 0;
            const int nd_tr = has_mask ? (int)nodata_mask[tr] : 0;
            const int nd_bl = has_mask ? (int)nodata_mask[bl] : 0;
            const int nd_br = has_mask ? (int)nodata_mask[br] : 0;

            // Binary classification: corner matches iff value == target AND not nodata
            const int match_tl = (!nd_tl) & (v_tl == target_value);
            const int match_tr = (!nd_tr) & (v_tr == target_value);
            const int match_br = (!nd_br) & (v_br == target_value);
            const int match_bl = (!nd_bl) & (v_bl == target_value);

            const int cls = (match_tl << 3) | (match_tr << 2) | (match_br << 1) | match_bl;

            cell_class[idx] = cls;
        }
    }
}
"""

# ---------------------------------------------------------------------------
# emit_edges kernel
# ---------------------------------------------------------------------------
# Grid-stride loop with ILP=4.  One logical work item per non-trivial cell.
#
# Each non-trivial cell emits 1 or 2 directed edge segments.
# Edges connect midpoints of the 4 cell sides:
#   Top midpoint    = (cx + 0.5, cy)
#   Right midpoint  = (cx + 1,   cy + 0.5)
#   Bottom midpoint = (cx + 0.5, cy + 1)
#   Left midpoint   = (cx,       cy + 0.5)
#
# Midpoints are converted to world coordinates using precomputed affine
# half-step offsets passed as kernel params (avoids per-thread recomputation).
#
# Winding: interior (matching region) on the RIGHT of the directed edge.
#
# Outputs:
#   d_edge_x0, d_edge_y0 — start vertex world coords
#   d_edge_x1, d_edge_y1 — end vertex world coords
#   d_edge_value          — raster value for this edge (for grouping)
#
# The edge lookup tables are declared __constant__ to reduce register
# pressure and exploit the constant cache (broadcast to all threads in a
# warp accessing the same index).

EMIT_EDGES_KERNEL_SOURCE = r"""
// Constant-memory lookup tables shared by all threads.
// n_edges_table: number of edges emitted per marching-squares class.
__constant__ int c_n_edges[16] = {
    0, 1, 1, 1, 1, 2, 1, 1,
    1, 1, 2, 1, 1, 1, 1, 0
};

// Edge 0: start and end sides per class.  Side encoding: 0=T, 1=R, 2=B, 3=L.
__constant__ int c_e0_start[16] = {
    -1,  3,  2,  3,  1,  1,  2,  3,
     0,  0,  0,  0,  1,  1,  2, -1
};
__constant__ int c_e0_end[16] = {
    -1,  2,  1,  1,  0,  2,  0,  0,
     3,  2,  1,  1,  3,  2,  3, -1
};

// Edge 1: only populated for saddle cases 5 and 10.
__constant__ int c_e1_start[16] = {
    -1, -1, -1, -1, -1,  3, -1, -1,
    -1, -1,  2, -1, -1, -1, -1, -1
};
__constant__ int c_e1_end[16] = {
    -1, -1, -1, -1, -1,  0, -1, -1,
    -1, -1,  3, -1, -1, -1, -1, -1
};

extern "C" __global__ void emit_edges(
    const long long* __restrict__ compact_cell_idx,
    const int* __restrict__ compact_cell_class,
    const int* __restrict__ edge_offsets,
    double* __restrict__ edge_x0,
    double* __restrict__ edge_y0,
    double* __restrict__ edge_x1,
    double* __restrict__ edge_y1,
    double* __restrict__ edge_value,
    const int cell_width,
    const int n_cells,
    const double target_value,
    const double aff_a, const double aff_b, const double aff_c,
    const double aff_d, const double aff_e, const double aff_f,
    const double half_dx_x, const double half_dx_y,
    const double half_dy_x, const double half_dy_y
) {
    // Precomputed half-step offsets (passed as kernel params):
    //   half_dx_x = 0.5 * aff_a   (x-shift for half-step in column direction)
    //   half_dx_y = 0.5 * aff_d   (y-shift for half-step in column direction)
    //   half_dy_x = 0.5 * aff_b   (x-shift for half-step in row direction)
    //   half_dy_y = 0.5 * aff_e   (y-shift for half-step in row direction)

    const int stride = blockDim.x * gridDim.x;
    const int ILP = 4;

    for (int base = blockIdx.x * blockDim.x + threadIdx.x;
         base < n_cells;
         base += stride * ILP) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            const int tid = base + j * stride;
            if (tid >= n_cells) break;

            const long long orig_idx = compact_cell_idx[tid];
            const int cls = compact_cell_class[tid];

            const long long cy = orig_idx / cell_width;
            const long long cx = orig_idx - cy * cell_width;

            // World coordinates of cell origin (cx, cy)
            const double ox = aff_a * cx + aff_b * cy + aff_c;
            const double oy = aff_d * cx + aff_e * cy + aff_f;

            // Four midpoints in world coordinates, computed from origin + offsets
            // Top:    origin + half_dx  (midpoint of top edge)
            // Right:  origin + full_dx + half_dy  (midpoint of right edge)
            // Bottom: origin + half_dx + full_dy  (midpoint of bottom edge)
            // Left:   origin + half_dy  (midpoint of left edge)
            const double wx_top    = ox + half_dx_x;
            const double wy_top    = oy + half_dx_y;
            const double wx_right  = ox + aff_a + half_dy_x;
            const double wy_right  = oy + aff_d + half_dy_y;
            const double wx_bottom = ox + half_dx_x + aff_b;
            const double wy_bottom = oy + half_dx_y + aff_e;
            const double wx_left   = ox + half_dy_x;
            const double wy_left   = oy + half_dy_y;

            // Pack midpoints into arrays for table-driven lookup
            const double mx[4] = {wx_top, wx_right, wx_bottom, wx_left};
            const double my[4] = {wy_top, wy_right, wy_bottom, wy_left};

            const int n_edges = c_n_edges[cls];
            const int write_pos = edge_offsets[tid];

            if (n_edges >= 1) {
                const int s0 = c_e0_start[cls];
                const int e0 = c_e0_end[cls];

                edge_x0[write_pos] = mx[s0];
                edge_y0[write_pos] = my[s0];
                edge_x1[write_pos] = mx[e0];
                edge_y1[write_pos] = my[e0];
                edge_value[write_pos] = target_value;
            }

            if (n_edges >= 2) {
                const int s1 = c_e1_start[cls];
                const int e1 = c_e1_end[cls];

                edge_x0[write_pos + 1] = mx[s1];
                edge_y0[write_pos + 1] = my[s1];
                edge_x1[write_pos + 1] = mx[e1];
                edge_y1[write_pos + 1] = my[e1];
                edge_value[write_pos + 1] = target_value;
            }
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Edge count kernel (helper to get per-cell edge counts for prefix sum)
# ---------------------------------------------------------------------------
# Grid-stride loop with ILP=4.
# Uses __constant__ lookup table shared with emit_edges.

EDGE_COUNT_KERNEL_SOURCE = r"""
__constant__ int c_edge_count[16] = {
    0, 1, 1, 1, 1, 2, 1, 1,
    1, 1, 2, 1, 1, 1, 1, 0
};

extern "C" __global__ void edge_count(
    const int* __restrict__ compact_cell_class,
    int* __restrict__ counts,
    const int n_cells
) {
    const int stride = blockDim.x * gridDim.x;
    const int ILP = 4;

    for (int base = blockIdx.x * blockDim.x + threadIdx.x;
         base < n_cells;
         base += stride * ILP) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            const int tid = base + j * stride;
            if (tid >= n_cells) break;

            counts[tid] = c_edge_count[compact_cell_class[tid]];
        }
    }
}
"""
