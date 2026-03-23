"""NVRTC kernel sources for GPU connected component labeling (union-find).

Algorithm: iterative union-find with atomicMin merging and pointer jumping.
1. init_labels: foreground pixels get label = flat_index, background = -1
2. local_merge: 4/8-connected neighbor merge using atomicMin on root labels
   - Path-compressing find_root with path-splitting
   - Grid-stride loop for full GPU utilization
3. pointer_jump: path compression until full convergence (run to fixpoint)
4. relabel: compact labels via direct lookup table (no binary search)

ADR-0040: CCCL Connected Component Labeling
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 1: Initialize labels (grid-stride loop)
# ---------------------------------------------------------------------------

INIT_LABELS_SOURCE = r"""
extern "C" __global__
void init_labels(
    int* __restrict__ labels,
    const unsigned char* __restrict__ foreground,
    const int n
) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        labels[idx] = foreground[idx] ? idx : -1;
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 2: Local merge (4-connected) -- path-splitting find_root
# ---------------------------------------------------------------------------

LOCAL_MERGE_4C_SOURCE = r"""
__device__ inline int find_root(int* __restrict__ labels, int idx) {
    int root = idx;
    while (true) {
        int parent = labels[root];
        if (parent == root) break;
        // Path splitting: make root point to grandparent
        int grandparent = labels[parent];
        // Use atomicCAS for safe path splitting (non-destructive)
        atomicCAS(&labels[root], parent, grandparent);
        root = parent;
    }
    return root;
}

__device__ inline void union_roots(int* __restrict__ labels, int a, int b) {
    // Lock-free union: always make the larger root point to the smaller
    while (a != b) {
        if (a < b) {
            // Try to make b point to a
            int old = atomicCAS(&labels[b], b, a);
            if (old == b) return;  // Success
            // Someone else changed labels[b]; re-find
            b = find_root(labels, b);
            a = find_root(labels, a);
        } else {
            // Swap so a < b
            int tmp = a; a = b; b = tmp;
        }
    }
}

extern "C" __global__
void local_merge_4c(
    int* __restrict__ labels,
    const unsigned char* __restrict__ foreground,
    const int width,
    const int height,
    int* __restrict__ changed
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;
    if (!foreground[idx]) return;

    int my_root = find_root(labels, idx);

    // 4-connected neighbors: right and down only (avoids redundant work)
    // Each edge is processed once instead of twice
    if (col + 1 < width) {
        int nidx = idx + 1;
        if (foreground[nidx]) {
            int n_root = find_root(labels, nidx);
            if (n_root != my_root) {
                union_roots(labels, my_root, n_root);
                *changed = 1;
                my_root = find_root(labels, idx);
            }
        }
    }
    if (row + 1 < height) {
        int nidx = idx + width;
        if (foreground[nidx]) {
            int n_root = find_root(labels, nidx);
            if (n_root != my_root) {
                union_roots(labels, my_root, n_root);
                *changed = 1;
                my_root = find_root(labels, idx);
            }
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 2 (alt): Local merge (8-connected) -- path-splitting find_root
# ---------------------------------------------------------------------------

LOCAL_MERGE_8C_SOURCE = r"""
__device__ inline int find_root(int* __restrict__ labels, int idx) {
    int root = idx;
    while (true) {
        int parent = labels[root];
        if (parent == root) break;
        int grandparent = labels[parent];
        atomicCAS(&labels[root], parent, grandparent);
        root = parent;
    }
    return root;
}

__device__ inline void union_roots(int* __restrict__ labels, int a, int b) {
    while (a != b) {
        if (a < b) {
            int old = atomicCAS(&labels[b], b, a);
            if (old == b) return;
            b = find_root(labels, b);
            a = find_root(labels, a);
        } else {
            int tmp = a; a = b; b = tmp;
        }
    }
}

extern "C" __global__
void local_merge_8c(
    int* __restrict__ labels,
    const unsigned char* __restrict__ foreground,
    const int width,
    const int height,
    int* __restrict__ changed
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;
    if (!foreground[idx]) return;

    int my_root = find_root(labels, idx);

    // 8-connected: only check right, bottom-left, bottom, bottom-right
    // This processes each edge exactly once (asymmetric neighborhood)

    // Right neighbor
    if (col + 1 < width) {
        int nidx = idx + 1;
        if (foreground[nidx]) {
            int n_root = find_root(labels, nidx);
            if (n_root != my_root) {
                union_roots(labels, my_root, n_root);
                *changed = 1;
                my_root = find_root(labels, idx);
            }
        }
    }
    // Bottom-left neighbor
    if (row + 1 < height && col > 0) {
        int nidx = idx + width - 1;
        if (foreground[nidx]) {
            int n_root = find_root(labels, nidx);
            if (n_root != my_root) {
                union_roots(labels, my_root, n_root);
                *changed = 1;
                my_root = find_root(labels, idx);
            }
        }
    }
    // Bottom neighbor
    if (row + 1 < height) {
        int nidx = idx + width;
        if (foreground[nidx]) {
            int n_root = find_root(labels, nidx);
            if (n_root != my_root) {
                union_roots(labels, my_root, n_root);
                *changed = 1;
                my_root = find_root(labels, idx);
            }
        }
    }
    // Bottom-right neighbor
    if (row + 1 < height && col + 1 < width) {
        int nidx = idx + width + 1;
        if (foreground[nidx]) {
            int n_root = find_root(labels, nidx);
            if (n_root != my_root) {
                union_roots(labels, my_root, n_root);
                *changed = 1;
                my_root = find_root(labels, idx);
            }
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 3: Pointer jumping (path compression) -- grid-stride with ILP
# ---------------------------------------------------------------------------

POINTER_JUMP_SOURCE = r"""
extern "C" __global__
void pointer_jump(
    int* __restrict__ labels,
    const int n,
    int* __restrict__ changed
) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        int label = labels[idx];
        if (label < 0) continue;

        // Bounded path compression: chase toward root with hop limit.
        // 64 hops is sufficient for all practical tree depths; for
        // pathological inputs (long thin components), remaining depth
        // is flattened in subsequent pointer_jump iterations.
        int root = label;
        for (int hop = 0; hop < 64; hop++) {
            int parent = labels[root];
            if (parent == root) break;
            root = parent;
        }
        if (root != label) {
            labels[idx] = root;
            *changed = 1;
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 4: Relabel via direct lookup table (no binary search)
# ---------------------------------------------------------------------------

RELABEL_SOURCE = r"""
extern "C" __global__
void relabel(
    int* __restrict__ labels,
    const int* __restrict__ compact_lut,
    const int lut_size,
    const int n
) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        int label = labels[idx];
        if (label < 0) {
            labels[idx] = 0;
        } else {
            // Direct lookup: compact_lut[label] gives compact ID
            labels[idx] = (label < lut_size) ? compact_lut[label] : 0;
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Sieve filter: zero out labels with component size below threshold
# ---------------------------------------------------------------------------

SIEVE_THRESHOLD_SOURCE = r"""
extern "C" __global__
void sieve_threshold(
    const int* __restrict__ labels_in,
    int* __restrict__ labels_out,
    const int* __restrict__ component_sizes,
    const int num_bins,
    const int min_size,
    const int replace_value,
    const int nodata_label,
    const int n
) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        int label = labels_in[idx];
        if (label == nodata_label) {
            labels_out[idx] = replace_value;
        } else if (label >= 0 && label < num_bins &&
                   component_sizes[label] < min_size) {
            labels_out[idx] = replace_value;
        } else {
            labels_out[idx] = label;
        }
    }
}
"""
