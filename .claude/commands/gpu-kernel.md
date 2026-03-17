# GPU Kernel Development Guide — vibespatial-raster

You are writing GPU kernels for vibespatial-raster. Follow these rules strictly.

---

## 1. Architecture Constraints (Non-Negotiable)

- **NO cuCIM, NO CuPy ndimage.** All GPU raster ops must use custom CCCL primitives + NVRTC kernels.
- **Zero-copy design:** Once data is on-device, ALL processing stays on-device. No host/device transfers mid-pipeline. The only transfers allowed are: initial input H→D, final result D→H.
- **OwnedRasterArray stores native dtype** (uint8, int16, float32, float64) — NOT always fp64. Kernel templates must handle the actual dtype.
- **Rasterio is optional.** Check `has_rasterio_support()` and raise clear errors. Tests skip gracefully.
- **scipy is a core dependency** (CPU baseline for CCL and morphology).

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

# 4. Calculate grid/block dimensions
block = (256, 1, 1)  # 1D
grid = ((total_items + 255) // 256, 1, 1)
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

---

## 3. Device Memory Management

### Transfers (CuPy)
```python
# Host → Device (only at start)
d_data = cp.asarray(np.ascontiguousarray(host_array))

# Allocate on device directly
d_output = cp.zeros((height, width), dtype=np.int32)
d_output = cp.full(n, fill_value, dtype=np.float64)

# Extract device pointer for NVRTC kernel params
ptr = d_data.data.ptr  # int — the raw device pointer

# Device → Host (only at end)
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

# Build result (always return HOST-resident)
host_result = cp.asnumpy(d_output)
result = from_numpy(host_result, nodata=raster.nodata, affine=raster.affine, crs=raster.crs)
```

---

## 4. CCCL Primitives (cuda.compute — Device-Wide Algorithms)

Use these instead of writing custom kernels when a CCCL primitive exists. They are already wrapped in vibespatial core's `cccl_primitives.py`.

### Available in cccl_primitives.py (pre-wrapped):
```python
from vibespatial.cccl_primitives import (
    segmented_reduce_sum,    # Per-segment sum
    segmented_reduce_min,    # Per-segment min  (if available)
    segmented_reduce_max,    # Per-segment max  (if available)
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
from cuda.compute import OpKind, CountingIterator, TransformIterator, ZipIterator

# Segmented reduce — one result per zone
cuda.compute.segmented_reduce(
    d_in=d_values,
    d_out=d_zone_results,
    start_offsets_in=d_starts,
    end_offsets_in=d_ends,
    op=OpKind.PLUS,         # or custom: lambda a, b: a + b
    h_init=np.array([0], dtype=np.float64),
    num_segments=n_zones,
)

# Radix sort (for building segment offsets)
cuda.compute.radix_sort(
    d_in_keys=d_zone_ids,
    d_out_keys=d_sorted_zones,
    d_in_values=d_pixel_values,
    d_out_values=d_sorted_values,
    order=cuda.compute.SortOrder.ASCENDING,
    num_items=n_pixels,
)

# Unique by key (for extracting unique zone IDs)
cuda.compute.unique_by_key(
    d_in_keys=d_sorted_zones,
    d_in_items=d_indices,
    d_out_keys=d_unique_zones,
    d_out_items=d_unique_indices,
    d_out_num_selected=d_count,
    op=OpKind.EQUAL_TO,
    num_items=n_pixels,
)

# Histogram (for zone pixel counts)
cuda.compute.histogram_even(
    d_samples=d_zone_ids,
    d_histogram=d_counts,
    num_output_levels=max_zone + 2,
    lower_level=0,
    upper_level=max_zone + 1,
    num_samples=n_pixels,
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

# ZipIterator — combine arrays without copying
pairs = ZipIterator(idx_iter, d_values)
```

---

## 5. Cooperative Primitives (cuda.coop — Block/Warp Level)

Use inside Numba CUDA kernels for intra-block coordination:

```python
import cuda.coop as coop
from numba import cuda as numba_cuda
import numba

# Block-level reduction
block_reduce = coop.block.make_reduce(numba.int32, threads_per_block=256, binary_op=max_op)

# Block-level scan (prefix sum)
block_scan = coop.block.make_exclusive_sum(numba.int32, threads_per_block=256, items_per_thread=1)

# Block-level sort
block_sort = coop.block.make_radix_sort_keys(numba.int32, threads_per_block=256, items_per_thread=4)

# Warp-level reduction
warp_reduce = coop.warp.make_reduce(numba.int32, binary_op=max_op)

# Link primitives to kernel
@numba_cuda.jit(link=block_reduce.files)
def my_kernel(input_arr, output_arr):
    result = block_reduce(input_arr[numba_cuda.threadIdx.x])
    if numba_cuda.threadIdx.x == 0:
        output_arr[0] = result
```

---

## 6. GPU Saturation Techniques

### Thread Occupancy
```python
# Use runtime's occupancy API for optimal block size
block_size = runtime.optimal_block_size(kernel, shared_mem_bytes=0)
grid_x = max(1, (item_count + block_size - 1) // block_size)

# Or use launch_config helper
grid, block = runtime.launch_config(kernel, item_count, shared_mem_bytes=0)
```

### Memory Coalescing
- Access consecutive memory addresses from consecutive threads
- For 2D rasters: row-major layout means adjacent threads should process adjacent columns
- Avoid strided access patterns — transpose data if needed

### Shared Memory for Stencil Operations
```c
// In NVRTC kernel source — use shared memory for neighborhood access
extern "C" __global__ void stencil_kernel(...) {
    __shared__ float tile[TILE_H + 2][TILE_W + 2];  // +2 for halo
    int tx = threadIdx.x, ty = threadIdx.y;
    int gx = blockIdx.x * TILE_W + tx;
    int gy = blockIdx.y * TILE_H + ty;

    // Load tile with halo into shared memory (one load per thread)
    tile[ty + 1][tx + 1] = (gx < width && gy < height) ? input[gy * width + gx] : 0.0f;

    // Load halo cells (border threads load extra)
    if (tx == 0)          tile[ty + 1][0]          = (gx > 0) ? input[gy * width + gx - 1] : 0.0f;
    if (tx == TILE_W - 1) tile[ty + 1][TILE_W + 1] = (gx + 1 < width) ? input[gy * width + gx + 1] : 0.0f;
    // ... similar for ty boundaries

    __syncthreads();

    // Now all neighbors are in shared memory — no global memory stalls
    float result = tile[ty][tx + 1] + tile[ty + 2][tx + 1] + tile[ty + 1][tx] + tile[ty + 1][tx + 2];
}
```

### Minimize Divergence
- Avoid warp-divergent branches in inner loops
- Process nodata pixels with masking rather than early-return
- Use predicated writes: `output[idx] = valid ? result : nodata;`

### Kernel Fusion
- Combine sequential operations into a single kernel launch when possible
- Use TransformIterator to fuse element-wise ops with reductions
- Avoid launching many tiny kernels — each launch has ~5μs overhead

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
    const double* __restrict__ input,
    double* __restrict__ output,
    const int width,
    const int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx >= total) return;

    output[idx] = /* operation on input[idx] */;
}
```

### 2D Stencil Kernel (Focal/Morphology)
```c
extern "C" __global__
void stencil_3x3(
    const double* __restrict__ input,
    double* __restrict__ output,
    const unsigned char* __restrict__ nodata_mask,  // nullable
    const int width,
    const int height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;

    int idx = row * width + col;

    // Skip nodata
    if (nodata_mask != nullptr && nodata_mask[idx]) {
        output[idx] = /* nodata_value */;
        return;
    }

    // 3x3 neighborhood access
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int ny = row + dy, nx = col + dx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                int nidx = ny * width + nx;
                // accumulate neighbor values
            }
        }
    }

    output[idx] = /* result */;
}
```

### Iterative Convergence Kernel (CCL Union-Find)
```c
// Phase 1: Initialize labels
extern "C" __global__
void init_labels(int* labels, const unsigned char* foreground, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    labels[idx] = foreground[idx] ? idx : -1;
}

// Phase 2: Local merge with atomicMin
extern "C" __global__
void local_merge(int* labels, const unsigned char* foreground,
                 int width, int height, int* changed) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;
    if (!foreground[idx]) return;

    int my_label = labels[idx];
    // Check 4-connected neighbors
    int neighbors[4] = {/* up, down, left, right */};
    for (int i = 0; i < 4; i++) {
        int nidx = neighbors[i];
        if (valid(nidx) && foreground[nidx]) {
            int nlabel = labels[nidx];
            if (nlabel < my_label) {
                atomicMin(&labels[my_label], nlabel);
                *changed = 1;
            }
        }
    }
}

// Phase 3: Pointer jumping (path compression)
extern "C" __global__
void pointer_jump(int* labels, int n, int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (labels[idx] < 0) return;

    int label = labels[idx];
    int root = labels[label];
    if (root != label) {
        labels[idx] = root;
        *changed = 1;
    }
}
```

---

## 8. Dispatcher Pattern (GPU/CPU Auto-Selection)

Follow the pattern from `rasterize.py`:

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

# Mark GPU tests
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
