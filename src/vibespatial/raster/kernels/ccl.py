"""NVRTC kernel sources for GPU connected component labeling (union-find).

Algorithm: iterative union-find with atomicMin merging and pointer jumping.
1. init_labels: foreground pixels get label = flat_index, background = -1
2. local_merge: 4/8-connected neighbor merge using atomicMin on root labels
3. pointer_jump: path compression until convergence
4. relabel: compact labels to sequential 1..N

ADR-0040: CCCL Connected Component Labeling
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Phase 1: Initialize labels
# ---------------------------------------------------------------------------

INIT_LABELS_SOURCE = r"""
extern "C" __global__
void init_labels(
    int* __restrict__ labels,
    const unsigned char* __restrict__ foreground,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    labels[idx] = foreground[idx] ? idx : -1;
}
"""

# ---------------------------------------------------------------------------
# Phase 2: Local merge (4-connected)
# ---------------------------------------------------------------------------

LOCAL_MERGE_4C_SOURCE = r"""
__device__ inline int find_root(int* labels, int idx) {
    int root = idx;
    while (labels[root] != root) {
        root = labels[root];
    }
    return root;
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

    // 4-connected neighbors: up, down, left, right
    int dx[4] = {0, 0, -1, 1};
    int dy[4] = {-1, 1, 0, 0};

    for (int i = 0; i < 4; i++) {
        int nc = col + dx[i];
        int nr = row + dy[i];
        if (nc < 0 || nc >= width || nr < 0 || nr >= height) continue;

        int nidx = nr * width + nc;
        if (!foreground[nidx]) continue;

        int n_root = find_root(labels, nidx);
        if (n_root == my_root) continue;

        // Union: smaller root becomes parent of larger root
        int lo = min(my_root, n_root);
        int hi = max(my_root, n_root);
        int old = atomicMin(&labels[hi], lo);
        if (old != lo) {
            *changed = 1;
            my_root = lo;  // update for subsequent neighbors
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 2 (alt): Local merge (8-connected)
# ---------------------------------------------------------------------------

LOCAL_MERGE_8C_SOURCE = r"""
__device__ inline int find_root(int* labels, int idx) {
    int root = idx;
    while (labels[root] != root) {
        root = labels[root];
    }
    return root;
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

    // 8-connected neighbors
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nc = col + dx;
            int nr = row + dy;
            if (nc < 0 || nc >= width || nr < 0 || nr >= height) continue;

            int nidx = nr * width + nc;
            if (!foreground[nidx]) continue;

            int n_root = find_root(labels, nidx);
            if (n_root == my_root) continue;

            int lo = min(my_root, n_root);
            int hi = max(my_root, n_root);
            int old = atomicMin(&labels[hi], lo);
            if (old != lo) {
                *changed = 1;
                my_root = lo;
            }
        }
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 3: Pointer jumping (path compression)
# ---------------------------------------------------------------------------

POINTER_JUMP_SOURCE = r"""
extern "C" __global__
void pointer_jump(
    int* __restrict__ labels,
    const int n,
    int* __restrict__ changed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int label = labels[idx];
    if (label < 0) return;

    int root = labels[label];
    if (root != label) {
        labels[idx] = root;
        *changed = 1;
    }
}
"""

# ---------------------------------------------------------------------------
# Phase 4: Relabel roots to compact sequential labels (1..N)
# ---------------------------------------------------------------------------

RELABEL_SOURCE = r"""
extern "C" __global__
void relabel(
    int* __restrict__ labels,
    const int* __restrict__ root_to_compact,
    const int* __restrict__ unique_roots,
    const int num_roots,
    const int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int label = labels[idx];
    if (label < 0) {
        labels[idx] = 0;
        return;
    }

    // Binary search for root in unique_roots array
    int lo = 0, hi = num_roots - 1;
    int compact = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (unique_roots[mid] == label) {
            compact = root_to_compact[mid];
            break;
        } else if (unique_roots[mid] < label) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    labels[idx] = compact;
}
"""
