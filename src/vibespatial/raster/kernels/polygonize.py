"""NVRTC kernel sources for marching-squares GPU polygonize.

Kernels:
  classify_cells — classify each 2x2 cell into one of 16 marching-squares cases
  emit_edges     — emit directed edge segments for non-trivial cells

The 16 cell classification indices encode which of the 4 corners (TL, TR, BR, BL)
match the target value.  Index 0 = no corners match, index 15 = all corners match.
Non-trivial cells (1..14) produce 1 or 2 boundary edge segments.

Edge segments are directed so that the *interior* (matching region) is always
on the RIGHT side of the edge.  This winding convention allows the host-side
ring chainer to assemble correctly-oriented polygon rings.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# classify_cells kernel
# ---------------------------------------------------------------------------
# One thread per 2x2 cell.  Cell (cx, cy) covers raster pixels:
#   TL = (cy, cx)      TR = (cy, cx+1)
#   BL = (cy+1, cx)    BR = (cy+1, cx+1)
#
# cell_width  = raster_width  - 1
# cell_height = raster_height - 1
# total_cells = cell_width * cell_height
#
# Outputs:
#   d_cell_class[cell_idx]  — 4-bit classification (0..15)
#   d_cell_value[cell_idx]  — raster value of the TL corner (used for grouping)
#
# The classification bit-pattern is:
#   bit 3 (8)  = TL matches
#   bit 2 (4)  = TR matches
#   bit 1 (2)  = BR matches
#   bit 0 (1)  = BL matches
#
# "Matches" means the pixel value equals the TL corner value AND is not nodata.

CLASSIFY_CELLS_KERNEL_SOURCE = r"""
extern "C" __global__ void classify_cells(
    const double* __restrict__ raster,
    const unsigned char* __restrict__ nodata_mask,  // 1=nodata, nullable
    int* __restrict__ cell_class,
    double* __restrict__ cell_value,
    int raster_width,
    int raster_height,
    int cell_width,
    int cell_height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = cell_width * cell_height;
    if (idx >= total_cells) return;

    int cy = idx / cell_width;
    int cx = idx % cell_width;

    // Pixel indices for the 4 corners
    int tl = cy * raster_width + cx;
    int tr = cy * raster_width + (cx + 1);
    int bl = (cy + 1) * raster_width + cx;
    int br = (cy + 1) * raster_width + (cx + 1);

    double v_tl = raster[tl];
    double v_tr = raster[tr];
    double v_bl = raster[bl];
    double v_br = raster[br];

    // Check nodata
    int nd_tl = (nodata_mask != nullptr) ? nodata_mask[tl] : 0;
    int nd_tr = (nodata_mask != nullptr) ? nodata_mask[tr] : 0;
    int nd_bl = (nodata_mask != nullptr) ? nodata_mask[bl] : 0;
    int nd_br = (nodata_mask != nullptr) ? nodata_mask[br] : 0;

    // If TL is nodata, this cell is trivial (class 0, no edges)
    if (nd_tl) {
        cell_class[idx] = 0;
        cell_value[idx] = 0.0;
        return;
    }

    // Reference value is TL corner
    double ref = v_tl;

    // Classify corners: does each corner match the reference value?
    // Nodata corners do NOT match.
    int match_tl = 1;  // always matches itself
    int match_tr = (!nd_tr && v_tr == ref) ? 1 : 0;
    int match_br = (!nd_br && v_br == ref) ? 1 : 0;
    int match_bl = (!nd_bl && v_bl == ref) ? 1 : 0;

    int cls = (match_tl << 3) | (match_tr << 2) | (match_br << 1) | match_bl;

    cell_class[idx] = cls;
    cell_value[idx] = ref;
}
"""

# ---------------------------------------------------------------------------
# emit_edges kernel
# ---------------------------------------------------------------------------
# One thread per non-trivial cell (class != 0 and class != 15).
#
# Each non-trivial cell emits 1 or 2 directed edge segments.
# Edges connect midpoints of the 4 cell sides:
#   Top midpoint    = (cx + 0.5, cy)
#   Right midpoint  = (cx + 1,   cy + 0.5)
#   Bottom midpoint = (cx + 0.5, cy + 1)
#   Left midpoint   = (cx,       cy + 0.5)
#
# These are in pixel coordinates; the affine transform converts to world coords
# inside the kernel.
#
# The edge table encodes, for each of the 16 classes, up to 2 edges.
# Each edge is (start_side, end_side) where sides are:
#   0 = Top, 1 = Right, 2 = Bottom, 3 = Left
# -1 means "no edge".
#
# Winding: interior (matching region) on the RIGHT of the directed edge.
#
# Outputs:
#   d_edge_x0, d_edge_y0 — start vertex world coords
#   d_edge_x1, d_edge_y1 — end vertex world coords
#   d_edge_value          — raster value for this edge (for grouping)
#   d_edge_count_per_cell — number of edges emitted by each cell (1 or 2)

EMIT_EDGES_KERNEL_SOURCE = r"""
extern "C" __global__ void emit_edges(
    const int* __restrict__ compact_cell_idx,    // original cell indices (compacted)
    const int* __restrict__ compact_cell_class,  // classification for each compacted cell
    const double* __restrict__ compact_cell_value,
    const int* __restrict__ edge_offsets,         // prefix sum: where each cell writes
    double* __restrict__ edge_x0,
    double* __restrict__ edge_y0,
    double* __restrict__ edge_x1,
    double* __restrict__ edge_y1,
    double* __restrict__ edge_value,
    int cell_width,
    int n_cells,        // number of non-trivial cells
    double aff_a, double aff_b, double aff_c,
    double aff_d, double aff_e, double aff_f
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cells) return;

    int orig_idx = compact_cell_idx[tid];
    int cls = compact_cell_class[tid];
    double val = compact_cell_value[tid];

    int cy = orig_idx / cell_width;
    int cx = orig_idx % cell_width;

    // Midpoints in pixel coordinates (fractional)
    // Top:    (cx + 0.5, cy)
    // Right:  (cx + 1,   cy + 0.5)
    // Bottom: (cx + 0.5, cy + 1)
    // Left:   (cx,       cy + 0.5)
    double mx[4], my[4];
    mx[0] = cx + 0.5; my[0] = cy;        // Top
    mx[1] = cx + 1.0; my[1] = cy + 0.5;  // Right
    mx[2] = cx + 0.5; my[2] = cy + 1.0;  // Bottom
    mx[3] = cx;        my[3] = cy + 0.5;  // Left

    // Edge table: for each class (0-15), up to 2 edges.
    // Each edge is encoded as (start_side, end_side).
    // -1 means no edge.
    // Winding: interior on RIGHT of directed edge.
    //
    // The 16 marching-squares cases with TL=bit3, TR=bit2, BR=bit1, BL=bit0:
    //  0 (0000): no edges
    //  1 (0001): BL only      -> Left to Bottom    (interior=BL, on right)
    //  2 (0010): BR only      -> Bottom to Right
    //  3 (0011): BL+BR        -> Left to Right
    //  4 (0100): TR only      -> Right to Top
    //  5 (0101): TR+BL saddle -> Right to Bottom, Left to Top  (ambiguous, choose one)
    //  6 (0110): TR+BR        -> Bottom to Top
    //  7 (0111): TR+BR+BL     -> Left to Top
    //  8 (1000): TL only      -> Top to Left
    //  9 (1001): TL+BL        -> Top to Bottom
    // 10 (1010): TL+BR saddle -> Top to Right, Bottom to Left  (ambiguous, choose one)
    // 11 (1011): TL+BL+BR     -> Top to Right
    // 12 (1100): TL+TR        -> Right to Left
    // 13 (1101): TL+TR+BL     -> Right to Bottom
    // 14 (1110): TL+TR+BR     -> Bottom to Left
    // 15 (1111): no edges

    // Edge table: [class][edge_idx] = (start_side, end_side) packed as start*4+end
    // Using -1 for no-edge.
    // Encoding: side 0=T, 1=R, 2=B, 3=L
    // start_side * 4 + end_side, or -1

    // Lookup: number of edges per class
    const int n_edges_table[16] = {
        0, 1, 1, 1, 1, 2, 1, 1,
        1, 1, 2, 1, 1, 1, 1, 0
    };

    // Edge start sides and end sides for each class, up to 2 edges
    // Edge 0:
    const int e0_start[16] = {
        -1,  3,  2,  3,  1,  1,  2,  3,
         0,  0,  0,  0,  1,  1,  2, -1
    };
    const int e0_end[16] = {
        -1,  2,  1,  1,  0,  2,  0,  0,
         3,  2,  1,  1,  3,  2,  3, -1
    };
    // Edge 1 (only for saddle cases 5 and 10):
    const int e1_start[16] = {
        -1, -1, -1, -1, -1,  3, -1, -1,
        -1, -1,  2, -1, -1, -1, -1, -1
    };
    const int e1_end[16] = {
        -1, -1, -1, -1, -1,  0, -1, -1,
        -1, -1,  3, -1, -1, -1, -1, -1
    };

    int n_edges = n_edges_table[cls];
    int write_pos = edge_offsets[tid];

    // Helper: convert pixel coords to world coords via affine
    // world_x = aff_a * px + aff_b * py + aff_c
    // world_y = aff_d * px + aff_e * py + aff_f

    if (n_edges >= 1) {
        int s0 = e0_start[cls];
        int e0 = e0_end[cls];
        double px0 = mx[s0], py0 = my[s0];
        double px1 = mx[e0], py1 = my[e0];

        edge_x0[write_pos] = aff_a * px0 + aff_b * py0 + aff_c;
        edge_y0[write_pos] = aff_d * px0 + aff_e * py0 + aff_f;
        edge_x1[write_pos] = aff_a * px1 + aff_b * py1 + aff_c;
        edge_y1[write_pos] = aff_d * px1 + aff_e * py1 + aff_f;
        edge_value[write_pos] = val;
    }

    if (n_edges >= 2) {
        int s1 = e1_start[cls];
        int e1_s = e1_end[cls];
        double px0 = mx[s1], py0 = my[s1];
        double px1 = mx[e1_s], py1 = my[e1_s];

        edge_x0[write_pos + 1] = aff_a * px0 + aff_b * py0 + aff_c;
        edge_y0[write_pos + 1] = aff_d * px0 + aff_e * py0 + aff_f;
        edge_x1[write_pos + 1] = aff_a * px1 + aff_b * py1 + aff_c;
        edge_y1[write_pos + 1] = aff_d * px1 + aff_e * py1 + aff_f;
        edge_value[write_pos + 1] = val;
    }
}
"""

# ---------------------------------------------------------------------------
# Edge count kernel (helper to get per-cell edge counts for prefix sum)
# ---------------------------------------------------------------------------

EDGE_COUNT_KERNEL_SOURCE = r"""
extern "C" __global__ void edge_count(
    const int* __restrict__ compact_cell_class,
    int* __restrict__ counts,
    int n_cells
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_cells) return;

    int cls = compact_cell_class[tid];

    const int n_edges_table[16] = {
        0, 1, 1, 1, 1, 2, 1, 1,
        1, 1, 2, 1, 1, 1, 1, 0
    };

    counts[tid] = n_edges_table[cls];
}
"""
