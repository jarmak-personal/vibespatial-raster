---
name: gpu-code-review
description: "PROACTIVELY USE THIS SKILL when reviewing GPU kernel code, CUDA/NVRTC source, CuPy operations, CCCL primitive usage, device memory management, stream-based pipelining, or any GPU dispatch logic. This skill contains quantitative thresholds, anti-pattern detection rules, and architecture-specific guidance for A100, H100, RTX 3090, and RTX 4090 GPUs. Use it to catch performance regressions, memory management issues, synchronization bugs, and access pattern problems before they land."
user-invocable: true
argument-hint: <file-path or PR diff to review>
---

# GPU Code Review Reference — vibespatial-raster

You are reviewing GPU code for vibespatial-raster. Use this reference to
identify performance issues, anti-patterns, and correctness risks. Every
finding should cite a specific rule from this document.

This library uses cuda-python, CuPy, CCCL (CUDA C++ Core Libraries with
Python bindings), and NVRTC for runtime compilation. Target hardware: A100,
H100 (datacenter) and RTX 3090, RTX 4090 (consumer).

---

## 0. Target Hardware Reference Card

| Spec | A100 (CC 8.0) | H100 (CC 9.0) | RTX 3090 (CC 8.6) | RTX 4090 (CC 8.9) |
|------|---------------|---------------|-------------------|-------------------|
| SMs | 108 | 132 | 82 | 128 |
| Registers per SM | 65,536 | 65,536 | 65,536 | 65,536 |
| Max threads per SM | 2,048 | 2,048 | 1,536 | 1,536 |
| Max blocks per SM | 32 | 32 | 16 | 24 |
| Shared mem per SM | 164 KB | 228 KB | 100 KB | 100 KB |
| Max shared mem/block | 163 KB | 227 KB | 99 KB | 99 KB |
| L2 cache | 40 MB | 50 MB | 6 MB | 72 MB |
| Memory bandwidth | 1,555 GB/s | 3,350 GB/s | 936 GB/s | 1,008 GB/s |
| FP64:FP32 ratio | 1:1 | 1:2 | 1:64 | 1:64 |
| Warp size | 32 | 32 | 32 | 32 |
| VRAM | 40/80 GB | 80 GB | 24 GB | 24 GB |

---

## 1. Memory Management

### 1.1 Pool Allocation
**Rule**: Always use CuPy memory pool. Never call raw `cudaMalloc`/`cudaFree`
in hot paths. Pool suballocations cost <1us; raw cudaMalloc costs ~100-1000us.

### 1.2 Fragmentation Avoidance
- [ ] Allocations freed in LIFO order where possible
- [ ] No interleaving of long-lived and short-lived allocations
- [ ] Temporary buffers freed before output allocation when possible

---

## 2. CUDA Stream Best Practices

### 2.1 When Streams Help
1. Independent H2D uploads (different arrays, no dependency)
2. Independent kernel launches on separate data
3. Overlapping D2H transfer with compute on different buffers

### 2.2 When NOT to Use
1. Sequential data dependencies — same-stream ordering handles this
2. Tiny transfers — stream creation costs ~1-2us
3. Single-kernel operations — nothing to overlap

### 2.3 Over-Synchronization (most common issue)
```python
# BAD: unnecessary sync between same-stream operations
runtime.launch(kernel_a, ...)
runtime.synchronize()           # UNNECESSARY
runtime.launch(kernel_b, ...)

# GOOD: sync only before host-side reads
runtime.launch(kernel_a, ...)
runtime.launch(kernel_b, ...)
runtime.synchronize()  # only when host needs the result
host_data = cp.asnumpy(d_output)
```

### 2.4 Implicit Synchronization Triggers
| Operation | Sync Type |
|-----------|-----------|
| `cudaMalloc()` | Device-wide |
| `cudaFree()` | Device-wide |
| CuPy `.get()` / `.item()` | Implicit sync + D2H |
| `cp.asnumpy()` | Implicit sync + D2H |
| `print()` of CuPy array | Implicit sync + D2H |
| `int(cupy_scalar)` | Implicit sync |

---

## 3. Kernel Launch Optimization

### 3.1 Occupancy-Based Block Sizing
**Rule**: NEVER hardcode `block=(256, 1, 1)`. Use the occupancy API.

```python
grid, block = runtime.launch_config(kernel, item_count)
```

### 3.2 Grid-Stride Loops
Preferred for most kernels. Grid size independent of problem size.

### 3.3 Launch Overhead
Kernel launch overhead: 2.5-5us. Minimum work per launch:
- Trivial kernel: ~10,000 threads
- Complex kernel (shared mem, loops): ~256 threads (1 block/SM min)
- If kernel < 25us execution: consider batching or fusing

### 3.4 Wave Quantization
On RTX 4090 (128 SMs), launching 129 blocks = ~50% GPU waste on final wave.
Size grids to exactly fill GPU with grid-stride loops.

---

## 4. Memory Access Patterns

### 4.1 Coalesced Access
**Rule**: Adjacent threads must access adjacent addresses. For 2D rasters,
row-major layout means adjacent threads process adjacent columns.

### 4.2 Shared Memory
- 32 banks, 4 bytes each
- **+1 padding** eliminates bank conflicts on column access:
  `__shared__ float tile[32][33];`
- Max per block: A100=163KB, H100=227KB, RTX 3090/4090=99KB

### 4.3 Read-Only Cache
```c
// Use const __restrict__ for automatic __ldg cache path
__global__ void kernel(const float* __restrict__ input, ...) {
```

---

## 5. Raster-Specific Patterns

### 5.1 Stencil/Focal Operations
- Must use shared memory with halo cells
- Tile size should maximize occupancy given shared memory needs
- `__syncthreads()` required after loading halo before computation
- Handle boundary pixels (clamp, mirror, or zero-pad)

### 5.2 Iterative Convergence (CCL)
- Minimize convergence iterations with path compression
- Use `atomicMin` for parallel merge
- Check convergence via device-side flag (not host round-trip per iteration)
- Avoid `synchronize()` between phases on same stream

### 5.3 Nodata Handling
- Use nullable nodata_mask pointer (nullptr when no nodata)
- Predicated writes over early-return to reduce divergence:
  `output[idx] = has_nodata ? nodata_val : computed_val;`
- Nodata propagation through neighborhood ops

### 5.4 Dtype Awareness
- Rasters store native dtype (uint8, int16, float32, float64)
- Kernel source must be templated on actual dtype
- Don't upcast to float64 unnecessarily
- Cache key must include dtype

---

## 6. Anti-Pattern Detection Checklist

### 6.1 Host-Device Ping-Pong (CRITICAL)
```python
# BAD: D->H->D round-trip
for i in range(n_zones):
    count = d_counts[i].get()        # D2H sync per iteration
    if count > 0: ...

# GOOD: keep everything on device
offsets = exclusive_sum(d_counts)
total = int((offsets[-1] + d_counts[-1]).get())  # single sync
```

### 6.2 Implicit Synchronization (HIGH)
| Pattern | Fix |
|---------|-----|
| `print(cupy_array)` | Remove or gate behind DEBUG |
| `len(cupy_array)` via `.get()` | Use `.shape[0]` |
| `int(cupy_scalar)` | Keep as device scalar until needed |
| `cp.asnumpy()` mid-pipeline | Defer to end |

### 6.3 Small Kernel Launches (MEDIUM)
Minimum grid sizes to saturate GPU:
| GPU | SMs | Min blocks for 50% | Min threads (256/block) |
|-----|-----|--------------------|-----------------------|
| A100 | 108 | 54 | 13,824 |
| H100 | 132 | 66 | 16,896 |
| RTX 3090 | 82 | 41 | 10,496 |
| RTX 4090 | 128 | 64 | 16,384 |

### 6.4 Register Pressure (MEDIUM)
Occupancy impact (A100, 65,536 regs/SM, 2,048 max threads/SM):
- 32 regs/thread -> 100% occupancy
- 64 regs/thread -> 50% occupancy
- 128 regs/thread -> 25% occupancy

On CC 8.6/8.9 (RTX 3090/4090, 1,536 max threads/SM):
- 43 regs/thread -> 87.5% occupancy
- 85 regs/thread -> ~48% occupancy

### 6.5 Missing Vectorized Loads (MEDIUM)
For memory-bound kernels, 128-bit loads give 1.3-1.5x bandwidth:
```c
double2 vals = reinterpret_cast<const double2*>(input)[idx];
```

---

## 7. Review Procedure

For each file with GPU code changes, perform these passes:

1. **Host-Device Boundary** — Find every transfer and sync point
2. **Synchronization** — Remove unnecessary syncs, flag implicit ones
3. **Kernel Efficiency** — Block size, grid size, divergence, access
4. **Memory Access** — Coalescing, shared memory, stencil tiling
5. **Memory Management** — Pool usage, pre-allocation, dtype
6. **NVRTC/Compilation** — Cache keys, parameterization, no compile in hot path

Every finding is **BLOCKING** unless purely cosmetic with zero impact.

---

## 8. Quick Reference Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| Block size | Hardcoded | Use occupancy API |
| Grid size < SM_count | GPU under-utilized | Increase parallelism |
| .get() in loop | D2H per iteration | Bulk device operation |
| synchronize() between same-stream ops | Unnecessary stall | Remove |
| Registers > 64/thread | Occupancy < 50% (datacenter) | Reduce or cap |
| Registers > 43/thread | Occupancy < 100% (consumer) | Monitor |
| Kernel < 25us | Launch-overhead dominated | Fuse or batch |
| Shared mem without +1 pad | Bank conflicts | Add padding |
| Missing `const __restrict__` | Missed cache optimization | Add annotation |
