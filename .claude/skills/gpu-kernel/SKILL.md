---
name: gpu-kernel
description: "PROACTIVELY USE THIS SKILL when writing, modifying, or reviewing GPU kernels, CUDA/NVRTC kernel source, CCCL primitive usage, device memory management, stream-based pipelining, or any GPU dispatch logic in src/vibespatial/raster/. Covers kernel lifecycle, CCCL primitives, shared-memory stencil tiling, CCL union-find patterns, cooperative primitives, GPU saturation techniques, and raster-specific dispatch patterns."
user-invocable: true
argument-hint: <optional kernel-file-path or topic>
---

# GPU Kernel Development Guide — vibespatial-raster

You are writing GPU kernels for vibespatial-raster. Follow these rules strictly.

## 1. Architecture Constraints (Non-Negotiable)

- **NO cuCIM, NO CuPy ndimage.** All GPU raster ops must use custom CCCL primitives + NVRTC kernels.
- **Zero-copy design:** Once data is on-device, ALL processing stays on-device. No host/device transfers mid-pipeline. The only transfers allowed are: initial input H->D, final result D->H.
- **OwnedRasterArray stores native dtype** (uint8, int16, float32, float64) — NOT always fp64. Kernel templates must handle the actual dtype.
- **Rasterio is optional.** Check `has_rasterio_support()` and raise clear errors. Tests skip gracefully.
- **scipy is a core dependency** (CPU baseline for CCL and morphology).

### Tier Decision

Before writing any GPU code, classify your operation:

```
Is the inner loop raster-specific (stencil neighborhood, CCL merge, marching-squares, PIP winding)?
  -> Yes: Custom NVRTC kernel
  -> No: Is it segmented (per-zone reduction, sort, scan)?
    -> Yes: CCCL segmented_* or cuda.compute
    -> No: Is it sort/unique/search/partition/histogram?
      -> Yes: CCCL / cuda.compute
      -> No: Is it element-wise/gather/scatter/concat?
        -> Yes: CuPy (cp.where, fancy indexing, cp.sum, etc.)
        -> No: Custom NVRTC kernel
```

---

## 2. Kernel Launch Lifecycle

Every NVRTC kernel follows this exact sequence:

```python
from vibespatial.cuda_runtime import (
    get_cuda_runtime,
    make_kernel_cache_key,
    KERNEL_PARAM_PTR,
    KERNEL_PARAM_I32,
    KERNEL_PARAM_F64,
)
import cupy as cp

# 1. Get runtime singleton
runtime = get_cuda_runtime()

# 2. Compile (cached via SHA1 of source)
cache_key = make_kernel_cache_key("my_kernel", KERNEL_SOURCE)
kernels = runtime.compile_kernels(
    cache_key=cache_key,
    source=KERNEL_SOURCE,
    kernel_names=("my_kernel",),
)

# 3. Construct parameters as (values_tuple, types_tuple)
params = (
    (d_input.data.ptr, d_output.data.ptr, width, height, some_float),
    (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_I32, KERNEL_PARAM_F64),
)

# 4. Occupancy-based launch config (NEVER hardcode block=(256,1,1))
grid, block = runtime.launch_config(kernels["my_kernel"], width * height)
# OR for 2D stencil:
block = (16, 16, 1)
grid = ((width + 15) // 16, (height + 15) // 16, 1)

# 5. Launch
runtime.launch(kernel=kernels["my_kernel"], grid=grid, block=block, params=params)
```

### Parameter Type Constants
```python
KERNEL_PARAM_PTR = ctypes.c_void_p    # Device memory pointers (d_array.data.ptr)
KERNEL_PARAM_I32 = ctypes.c_int       # 32-bit integers
KERNEL_PARAM_F64 = ctypes.c_double    # 64-bit floats
```

**CRITICAL:** Plain `ctypes.c_int` will fail. Always use the named constants from `vibespatial.cuda_runtime`.

### Occupancy-Based Block Sizing

**NEVER** hardcode `block=(256, 1, 1)`. For 1D kernels, use:

```python
grid, block = runtime.launch_config(kernel, item_count)
```

For 2D stencil kernels, use occupancy-aware tile sizes:

```python
block_size = runtime.optimal_block_size(kernel, shared_mem_bytes=smem)
# Choose 2D tile that matches block_size (e.g., 16x16=256)
```

---

## 3. Device Memory Management

### Transfers (CuPy)
```python
# Host -> Device (only at start)
d_data = cp.asarray(np.ascontiguousarray(host_array))

# Allocate on device directly
d_output = cp.zeros((height, width), dtype=np.int32)
d_output = cp.full(n, fill_value, dtype=np.float64)

# Extract device pointer for NVRTC kernel params
ptr = d_data.data.ptr  # int — the raw device pointer

# Device -> Host (only at end)
host_result = cp.asnumpy(d_output)
```

### OwnedRasterArray Integration
```python
from vibespatial.raster.buffers import Residency, TransferTrigger

# Move to device
raster.move_to(
    Residency.DEVICE,
    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
    reason="my_operation requires device-resident data",
)
d_data = raster.device_data()       # CuPy array
d_mask = raster.device_nodata_mask() # CuPy bool array or None

# Build result (always return HOST-resident for public API)
host_result = cp.asnumpy(d_output)
result = from_numpy(host_result, nodata=raster.nodata, affine=raster.affine, crs=raster.crs)
```

---

## 4. CCCL Primitives (cuda.compute — Device-Wide Algorithms)

Use these instead of writing custom kernels when a CCCL primitive exists.

### Available in cccl_primitives.py (pre-wrapped):
```python
from vibespatial.cccl_primitives import (
    segmented_reduce_sum,    # Per-segment sum
    segmented_reduce_min,    # Per-segment min
    segmented_reduce_max,    # Per-segment max
    segmented_sort,          # Sort within segments
    exclusive_sum,           # Prefix sum
    three_way_partition,     # 3-way partition with predicates
    counting_iterator,       # Zero-allocation lazy [0,1,2,...]
    transform_iterator,      # Fused lazy transforms
)
```

### Direct cuda.compute API (when wrappers don't exist):
```python
import cuda.compute
from cuda.compute import OpKind, CountingIterator, TransformIterator

# Segmented reduce — one result per zone
cuda.compute.segmented_reduce(
    d_in=d_values, d_out=d_zone_results,
    start_offsets_in=d_starts, end_offsets_in=d_ends,
    op=OpKind.PLUS, h_init=np.array([0], dtype=np.float64),
    num_segments=n_zones,
)

# Radix sort (for building segment offsets)
cuda.compute.radix_sort(
    d_in_keys=d_zone_ids, d_out_keys=d_sorted_zones,
    d_in_values=d_pixel_values, d_out_values=d_sorted_values,
    order=cuda.compute.SortOrder.ASCENDING, num_items=n_pixels,
)

# Histogram (for zone pixel counts)
cuda.compute.histogram_even(
    d_samples=d_zone_ids, d_histogram=d_counts,
    num_output_levels=max_zone + 2,
    lower_level=0, upper_level=max_zone + 1, num_samples=n_pixels,
)
```

### Object-based API (reuse temp buffers across calls):
```python
reducer = cuda.compute.make_segmented_reduce(d_in, d_out, starts, ends, op, h_init)
temp_bytes = reducer(None, d_in, d_out, starts, ends, op, h_init, n_segs)
temp = cp.empty(temp_bytes, dtype=np.uint8)
reducer(temp, d_in, d_out, starts, ends, op, h_init, n_segs)
```

### Iterators (zero-copy transforms):
```python
# CountingIterator — lazy [0,1,2,...] with no allocation
idx_iter = CountingIterator(np.int32(0))

# TransformIterator — fused lazy element-wise transform
squared = TransformIterator(d_values, lambda x: x * x)
```

---

## 5. Cooperative Primitives (cuda.coop — Block/Warp Level)

Use inside Numba CUDA kernels for intra-block coordination:

```python
import cuda.coop as coop
from numba import cuda as numba_cuda

# Block-level reduction
block_reduce = coop.block.make_reduce(numba.int32, threads_per_block=256, binary_op=max_op)

# Block-level scan (prefix sum)
block_scan = coop.block.make_exclusive_sum(numba.int32, threads_per_block=256, items_per_thread=1)

# Warp-level reduction
warp_reduce = coop.warp.make_reduce(numba.int32, binary_op=max_op)
```

---

## 6. GPU Saturation Techniques

### Thread Occupancy
```python
# Use runtime's occupancy API for optimal block size
grid, block = runtime.launch_config(kernel, item_count, shared_mem_bytes=0)
```

### Memory Coalescing
- Access consecutive memory addresses from consecutive threads
- For 2D rasters: row-major layout means adjacent threads process adjacent columns
- Avoid strided access patterns — transpose data if needed

### Shared Memory for Stencil Operations
```c
// In NVRTC kernel source — shared memory with halo for neighborhood access
extern "C" __global__ void stencil_kernel(...) {
    __shared__ float tile[TILE_H + 2][TILE_W + 2];  // +2 for halo
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Load tile center
    tile[ty + 1][tx + 1] = (gx < width && gy < height) ? input[gy * width + gx] : 0.0f;

    // Load halo cells (border threads load extra)
    if (tx == 0)          tile[ty + 1][0]          = (gx > 0) ? input[gy * width + gx - 1] : 0.0f;
    if (tx == TILE_W - 1) tile[ty + 1][TILE_W + 1] = (gx + 1 < width) ? input[gy * width + gx + 1] : 0.0f;
    // ... similar for ty boundaries

    __syncthreads();

    // All neighbors are in shared memory — no global memory stalls
    float result = tile[ty][tx + 1] + tile[ty + 2][tx + 1] + tile[ty + 1][tx] + tile[ty + 1][tx + 2];
}
```

### Bank Conflict Avoidance

Shared memory has 32 banks, each 4 bytes wide. The **+1 padding trick**
eliminates bank conflicts on column access:

```c
// BAD: 32-way bank conflict when reading columns
__shared__ float tile[32][32];

// GOOD: +1 padding shifts bank mapping — no conflicts
__shared__ float tile[32][33];
```

### Minimize Divergence
- Avoid warp-divergent branches in inner loops
- Process nodata pixels with masking rather than early-return
- Use predicated writes: `output[idx] = valid ? result : nodata;`

### Kernel Fusion
- Combine sequential operations into a single kernel launch when possible
- Use TransformIterator to fuse element-wise ops with reductions
- Avoid launching many tiny kernels — each launch has ~3-5us overhead

### Grid-Stride Loops with ILP

Prefer grid-stride loops over one-thread-per-element. Process multiple
elements per thread for instruction-level parallelism:

```c
const int stride = blockDim.x * gridDim.x;
for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < n;
     idx += stride * 4) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int elem = idx + j * stride;
        if (elem < n) output[elem] = compute(input[elem]);
    }
}
```

### Async Operations
```python
# Use CuPy streams for overlapping compute and transfer
stream = cp.cuda.Stream(non_blocking=True)
with stream:
    d_data = cp.asarray(host_data)
    # ... kernel launches on this stream
stream.synchronize()
```

---

## 7. NVRTC Kernel Source Templates

### 1D Per-Pixel Kernel
```c
extern "C" __global__
void per_pixel_op(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    const int width,
    const int height
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    output[idx] = /* operation on input[idx] */;
}}
```

### 2D Stencil Kernel (Focal/Morphology)
```c
extern "C" __global__
void stencil_3x3(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    const unsigned char* __restrict__ nodata_mask,  // nullable
    const int width,
    const int height
) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;

    // Skip nodata
    if (nodata_mask != nullptr && nodata_mask[idx]) {{
        output[idx] = /* nodata_value */;
        return;
    }}

    // 3x3 neighborhood access
    for (int dy = -1; dy <= 1; dy++) {{
        for (int dx = -1; dx <= 1; dx++) {{
            int ny = row + dy, nx = col + dx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {{
                int nidx = ny * width + nx;
                // accumulate neighbor values
            }}
        }}
    }}

    output[idx] = /* result */;
}}
```

### Iterative Convergence Kernel (CCL Union-Find)
```c
// Path-splitting find_root: compresses tree in-place via atomicCAS
__device__ int find_root(int* __restrict__ labels, int idx) {{
    int root = idx;
    while (true) {{
        int parent = labels[root];
        if (parent == root) return root;
        int grandparent = labels[parent];
        if (grandparent != parent)
            atomicCAS(&labels[root], parent, grandparent);  // path split
        root = parent;
    }}
}}

// Lock-free union via atomicCAS (replaces atomicMin)
__device__ void union_roots(int* __restrict__ labels, int a, int b) {{
    while (a != b) {{
        if (a > b) {{ int t = a; a = b; b = t; }}
        int old = atomicCAS(&labels[b], b, a);
        if (old == b) return;  // success
        a = find_root(labels, a);
        b = find_root(labels, old);
    }}
}}

// Phase 1: Initialize labels (grid-stride loop)
extern "C" __global__
void init_labels(int* __restrict__ labels,
                 const unsigned char* __restrict__ foreground, int n) {{
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride)
        labels[idx] = foreground[idx] ? idx : -1;
}}

// Phase 2: Local merge — asymmetric scan (right+down only for 4C)
extern "C" __global__
void local_merge_4c(int* __restrict__ labels,
                    const unsigned char* __restrict__ fg,
                    int width, int height, int* __restrict__ changed) {{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;
    if (!fg[idx]) return;
    int r = find_root(labels, idx);
    // Right neighbor
    if (col + 1 < width && fg[idx + 1]) {{
        int nr = find_root(labels, idx + 1);
        if (r != nr) {{ union_roots(labels, r, nr); *changed = 1; }}
    }}
    // Down neighbor
    if (row + 1 < height && fg[idx + width]) {{
        int nr = find_root(labels, idx + width);
        if (r != nr) {{ union_roots(labels, r, nr); *changed = 1; }}
    }}
}}

// Phase 3: Bounded pointer jumping (multi-hop with safety limit)
extern "C" __global__
void pointer_jump(int* __restrict__ labels, int n,
                  int* __restrict__ changed) {{
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += stride) {{
        int label = labels[idx];
        if (label < 0) continue;
        int root = label;
        for (int hop = 0; hop < 64; hop++) {{
            int parent = labels[root];
            if (parent == root) break;
            root = parent;
        }}
        if (root != label) {{
            labels[idx] = root;
            *changed = 1;
        }}
    }}
}}
```

---

## 8. Dispatcher Pattern (GPU/CPU Auto-Selection)

```python
def my_operation(raster, *, use_gpu: bool | None = None, **kwargs):
    """Public API — auto-dispatches to GPU or CPU."""
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return _my_operation_gpu(raster, **kwargs)
    else:
        return _my_operation_cpu(raster, **kwargs)

def _should_use_gpu(raster):
    """Auto-dispatch heuristic."""
    try:
        import cupy as cp
        from vibespatial.cuda_runtime import get_cuda_runtime
        runtime = get_cuda_runtime()
        return raster.pixel_count >= 100_000  # threshold for GPU advantage
    except (ImportError, RuntimeError):
        return False
```

---

## 9. Testing Pattern

```python
import pytest

requires_gpu = pytest.mark.skipif(
    not _has_cupy(), reason="CuPy not available"
)

@requires_gpu
def test_my_op_gpu():
    """Test GPU path produces same results as CPU."""
    raster = from_numpy(test_data, nodata=0)
    cpu_result = my_op_cpu(raster)
    gpu_result = my_op_gpu(raster)
    np.testing.assert_array_equal(cpu_result.data, gpu_result.data)

def test_my_op_auto_dispatch():
    """Test auto-dispatch falls back to CPU gracefully."""
    raster = from_numpy(test_data, nodata=0)
    result = my_op(raster)  # Should work regardless of GPU availability
    assert result is not None
```

---

## 10. Diagnostic Events

All GPU operations must log diagnostics:

```python
from vibespatial.raster.buffers import RasterDiagnosticEvent, RasterDiagnosticKind

result.diagnostics.append(
    RasterDiagnosticEvent(
        kind=RasterDiagnosticKind.RUNTIME,
        detail=f"gpu_kernel={kernel_name} pixels={total_pixels} blocks={grid_size}",
        residency=result.residency,
        visible_to_user=True,
        elapsed_seconds=elapsed,
    )
)
```

---

## 11. Performance Rules

### Memory Access
- **Coalesced reads**: Adjacent threads read adjacent addresses. Row-major
  layout is already coalesced for row processing.
- **`const __restrict__`**: Always annotate read-only pointer parameters.
- **Vectorized loads** for bandwidth-bound bulk I/O (double2/float4).

### Kernel Launch
- Never launch with <32 threads of real work (wastes a warp).
- Avoid many tiny kernels — each launch has ~3-5us overhead.
- Remove unnecessary syncs between same-stream operations.

### Synchronization Rules
- `runtime.synchronize()` syncs the CUDA context (all streams). Use sparingly.
- CUDA guarantees execution order within a single stream. Do NOT sync
  between consecutive launches on the same stream.
- Keep sync calls before: `cp.asnumpy()`, `.get()`, `copy_device_to_host`.

---

## 12. Dtype-Templated Kernel Sources via `.format()`

Rasters store native dtypes (uint8, int16, float32, float64). Kernel source
strings use `{dtype}` placeholders that are filled at compile time via
Python `.format()`. This avoids writing separate kernels per type.

**Pattern** (see `kernels/hydrology.py`, `kernels/resample.py`):

```python
KERNEL_TEMPLATE = r"""
extern "C" __global__
void my_kernel(
    const {dtype}* __restrict__ input,
    {dtype}* __restrict__ output,
    const int n
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = input[idx] * ({dtype})2.0;
}}
"""

def get_kernel_source(dtype_name: str) -> str:
    """Return kernel source for 'float' or 'double'."""
    return KERNEL_TEMPLATE.format(dtype=dtype_name)
```

**Rules:**
- Use `{{` and `}}` for literal braces in CUDA code (Python `.format()` escaping).
- Map numpy dtypes to C type names: `float32 -> "float"`, `float64 -> "double"`.
- **Integer input promotion:** Narrow integers (uint8, int16, itemsize < 4) are promoted to float32 (`"float"` kernel). Wide integers (int32, int64, itemsize >= 4) are promoted to float64 (`"double"` kernel) to preserve precision beyond 2^24. See the dtype dispatch in `_binary_op_gpu` and `_raster_expression_gpu` in `algebra.py`.
- For float32, use `f`-suffixed constants (`3.14f`) and math functions
  (`sqrtf`, `fabsf`) to avoid implicit double promotion.

---

## 13. Runtime Expression Compilation

The fused expression kernel (see `kernels/algebra.py`) compiles a
user-supplied arithmetic expression into a single GPU pass at runtime.
This avoids multiple kernel launches for chained element-wise ops.

**Pattern:**

```python
from vibespatial.raster.kernels.algebra import build_expression_kernel_source

# User writes: "a * 2.0 + sqrt(b)"
# build_expression_kernel_source produces a complete NVRTC source with:
#   - N input pointer parameters (const {dtype}* __restrict__ in_a, ...)
#   - N nodata mask parameters (const unsigned char* __restrict__ mask_a, ...)
#   - Grid-stride loop evaluating the expression per pixel
#   - Nodata propagation: if ANY input is nodata, output is nodata
#   - inf/nan guard: division-by-zero results become nodata

source = build_expression_kernel_source(
    expression="a * 2.0 + sqrt(b)",
    var_names=("a", "b"),
    dtype_name="double",  # or "float" for float32 rasters
)
# source is a complete NVRTC kernel string ready for compile_kernels()
```

**Rules:**
- Allowed functions are mapped to CUDA math builtins (see `ALLOWED_FUNCTIONS`
  and `ALLOWED_FUNCTIONS_F32` in `kernels/algebra.py`).
- Up to 8 input rasters (variables `a` through `h`).
- The expression string is sanitized before embedding in CUDA source.
- Each unique expression produces a unique kernel cache key via SHA1.

---

## 14. Compile-Time `-D` Defines for Parameterized Kernels

For kernels where parameters affect shared memory layout or loop bounds,
use NVRTC compile-time defines (`-D`) instead of runtime parameters. This
enables the compiler to optimize tile sizes and unroll loops.

**Pattern** (see `label.py` NxN morphology dispatch):

```python
# Structuring element dimensions determine shared memory halo size
defines = (
    f"-DSE_RADIUS_X={se_rx}",
    f"-DSE_RADIUS_Y={se_ry}",
    f"-DSE_W={se_w}",
    f"-DSE_H={se_h}",
    f"-DTILE_W={tile_w}",
    f"-DTILE_H={tile_h}",
)

kernels = runtime.compile_kernels(
    cache_key=cache_key,
    source=BINARY_ERODE_NXN_KERNEL_SOURCE,
    kernel_names=("binary_erode_nxn",),
    options=defines,  # passed to nvrtcCompileProgram
)
```

**When to use `-D` defines vs. kernel parameters:**
- Use `-D` when the value affects `__shared__` array dimensions or loop
  unroll counts (the compiler needs them at compile time).
- Use kernel parameters for values that change per-launch but do not
  affect memory layout (e.g., image width/height, nodata sentinel).
- Each unique set of defines produces a separate cached compilation.

---

## 15. Separable Pass Decomposition

For rectangular (full-row/column) structuring elements, morphology decomposes
2D operations into two 1D passes (horizontal then vertical). This reduces
per-pixel cost from O(W*H) to O(W + H).

**Pattern** (see `label.py` separable morphology, `kernels/morphology.py`
`BINARY_ERODE_SEP_H/V_KERNEL_SOURCE`):

```python
# Dispatch checks if the SE is a full rectangle
if se.all():  # All ones => separable
    # Horizontal pass: each thread processes one row segment
    # Vertical pass: each thread processes one column segment
    _execute_separable_morphology(d_fg, ops, se, width, height, ...)
else:
    # General NxN: 2D shared-memory tiled kernel
    _execute_nxn_morphology(d_fg, ops, se, width, height, ...)
```

**Kernel design for separable passes:**
- Each pass uses a 1D shared-memory tile with halo = SE radius.
- Horizontal kernel: block = (TILE_SIZE, 1, 1), grid covers rows.
- Vertical kernel: block = (1, TILE_SIZE, 1), grid covers columns.
- `-DRADIUS` and `-DTILE_SIZE` set via compile-time defines.
- For compound operations (open = erode + dilate), chain passes
  without synchronization between same-stream launches.

---

## 16. Ping-Pong Buffer Pattern

Iterative GPU algorithms use two buffer sets and alternate (ping-pong)
between them each iteration: read from buffer A, write to buffer B,
then swap. This avoids race conditions without global synchronization.

### Fixed-iteration ping-pong (JFA distance transform)

The Jump Flooding Algorithm runs exactly `log2(N)` iterations. No
convergence flag is needed. See `kernels/distance.py` and `distance.py`.

```python
# Allocate two seed buffers (SoA layout for coalesced access)
d_seed_x_a = cp.empty(n, dtype=np.int32)
d_seed_y_a = cp.empty(n, dtype=np.int32)
d_seed_x_b = cp.empty(n, dtype=np.int32)
d_seed_y_b = cp.empty(n, dtype=np.int32)

cur_sx, cur_sy = d_seed_x_a, d_seed_y_a
out_sx, out_sy = d_seed_x_b, d_seed_y_b

step_k = N // 2
while step_k >= 1:
    # Launch: read from cur, write to out
    runtime.launch(kernel=jfa_step, ...,
        params=((cur_sx.data.ptr, cur_sy.data.ptr,
                 out_sx.data.ptr, out_sy.data.ptr, ...),
                (KERNEL_PARAM_PTR, ...) ))
    # Swap (no sync needed -- same stream guarantees ordering)
    cur_sx, out_sx = out_sx, cur_sx
    cur_sy, out_sy = out_sy, cur_sy
    step_k //= 2
```

### Convergence-driven iteration (hydrology sink fill)

The priority-flood algorithm iterates until no pixel changes. A device-side
`changed` flag avoids per-iteration D2H of the entire buffer. See
`kernels/hydrology.py` and `hydrology.py`.

```python
d_changed = cp.zeros(1, dtype=np.int32)

for iteration in range(max_iterations):
    d_changed.fill(0)  # Reset flag on device (no D2H)
    runtime.launch(kernel=propagate, ...,
        params=((d_elevation.data.ptr, d_fill.data.ptr, ...,
                 d_changed.data.ptr),
                (KERNEL_PARAM_PTR, ...) ))
    # Single scalar D2H to check convergence
    if int(d_changed.get()) == 0:
        break
```

**Rules:**
- Allocate both buffers once before the loop (no per-iteration allocation).
- No `synchronize()` between iterations on the same stream.
- For convergence-driven loops, the only D2H per iteration is the
  single-int `changed` flag (unavoidable for the host-side break decision).

---

## 17. CCCL `histogram_even` Usage

For binned histograms of device arrays, use `cuda.compute.algorithms.histogram_even`
instead of custom kernels or `cp.histogram` (which transfers to host).

**Pattern** (see `histogram.py`):

```python
from cuda.compute import algorithms as cccl_algorithms

# CCCL histogram_even uses half-open intervals [lower, upper).
# Nudge upper past actual max so the last bin captures it.
hi_for_cccl = float(np.nextafter(np.float64(hi), np.float64(np.inf)))

# Samples must be float64 for histogram_even
d_samples = d_valid.astype(cp.float64, copy=False)

# NOTE: histogram_even requires int32 counters (atomicAdd_block limitation).
d_histogram = cp.zeros(bins, dtype=cp.int32)

cccl_algorithms.histogram_even(
    d_samples=d_samples,
    d_histogram=d_histogram,
    num_output_levels=bins + 1,
    lower_level=float(lo),
    upper_level=hi_for_cccl,
    num_samples=len(d_samples),
)
# d_histogram is now on device -- feed directly into CDF via exclusive_sum
```

**Rules:**
- Counter array must be `int32` (CCCL limitation). Cast to int64 after
  if needed for downstream arithmetic.
- `num_output_levels = bins + 1` (edges, not bin count).
- Combine with `exclusive_sum` for CDF computation (stays on device).
- For histogram equalization, build the LUT on device and apply via a
  custom NVRTC remap kernel (see `kernels/histogram.py`).
