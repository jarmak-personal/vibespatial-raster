# GPU Kernels

All GPU operations use custom NVRTC kernel source strings in `kernels/`.
No cuCIM, no CuPy ndimage.

## Kernel source files

| File | Kernels | Purpose |
|---|---|---|
| `kernels/ccl.py` | INIT_LABELS, LOCAL_MERGE_4C, LOCAL_MERGE_8C, POINTER_JUMP, RELABEL | Connected component labeling |
| `kernels/focal.py` | CONVOLVE_2D, CONVOLVE_NORMALIZED | 2D convolution with shared-memory tiling |
| `kernels/morphology.py` | BINARY_ERODE, BINARY_DILATE | Erosion/dilation with 1-pixel halo |
| `kernels/polygonize.py` | CLASSIFY_CELLS, EMIT_EDGES | Marching-squares edge extraction |

## CCL (Connected Component Labeling)

Union-find algorithm on GPU:

1. **Init** — each pixel gets label = flat_index (or -1 for background)
2. **Local merge** — NVRTC kernel, each thread checks 4/8 neighbors, `atomicMin`
   to merge labels
3. **Pointer jumping** — NVRTC kernel, compress label chains to roots
4. **Iterate** — repeat merge+jump until convergence (typically 3-5 passes)
5. **Relabel** — CCCL sort + unique_by_key for dense sequential labels

Thread block: one thread per pixel. Convergence detected by a global changed flag.

## Focal convolution

Shared-memory tiled convolution:

- 16x16 thread blocks
- Halo region loaded into shared memory (kernel_radius pixels on each side)
- Normalized variant divides by weight sum (for nodata handling)
- Output written to global memory

## Morphology

Binary erosion and dilation:

- 16x16 thread blocks with 1-pixel halo
- 3x3 structuring element
- Shared-memory tiling matches focal convolution pattern
- Multiple iterations supported via ping-pong between two buffers

## Polygonize (Marching Squares)

Two-kernel pipeline:

1. **CLASSIFY_CELLS** — each thread processes a 2x2 cell, produces a 4-bit
   case index from the 16 marching-squares cases
2. **EMIT_EDGES** — each thread emits 0-2 directed boundary edge segments
   based on the case index (interior always on right side)

Edge table encodes topology. Winding convention ensures correct polygon ring
orientation after host-side ring chaining.

## Kernel conventions

- All kernels are raw CUDA C strings compiled via NVRTC at runtime
- Thread block sizes: 16x16 for 2D kernels, 256 for 1D kernels
- Shared memory used for tiled access patterns (focal, morphology)
- `atomicMin` / `atomicCAS` for concurrent label updates (CCL)
- Bounds checking on every thread (grid may not be block-aligned)
