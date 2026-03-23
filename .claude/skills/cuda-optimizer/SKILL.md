---
name: cuda-optimizer
description: "Use this skill to optimize existing CUDA/NVRTC kernel code, CuPy operations, CCCL primitive usage, or GPU dispatch logic in src/vibespatial/raster/. Unlike gpu-code-review (which flags issues) and gpu-kernel (which guides new code), this skill reads existing code and produces concrete rewrites with measured justification. Invoke on pre-existing kernel files to bring them up to NVIDIA best-practice performance standards."
user-invocable: true
argument-hint: <file-path or module-name to optimize>
---

# CUDA Optimizer — vibespatial-raster

You are optimizing GPU code in vibespatial-raster. Your job is to read the
target file, identify every concrete optimization opportunity, and produce
ready-to-apply rewrites ranked by expected impact.

**Target:** `$ARGUMENTS`

---

## Procedure

### Step 0: Read the Code

Read the target file completely. Also read any files it imports from
`vibespatial.*` that contain kernel source strings or GPU dispatch logic.
Identify:

- All NVRTC kernel source strings (look for `_KERNEL_SOURCE` or multi-line
  strings containing `__global__`)
- All CuPy operations (`cp.` calls)
- All CCCL primitive usage (`cccl_primitives.*`, `cuda.compute.*`)
- All `runtime.launch()` calls and their parameters
- All host-device transfers (`.get()`, `cp.asnumpy()`, `cp.asarray()`)
- All synchronization points (`runtime.synchronize()`, `stream.synchronize()`)

### Step 1: Host-Device Boundary (CRITICAL — highest impact)

Scan for these patterns and produce rewrites:

**1a. D->H->D ping-pong in loops**
```python
# BEFORE: ping-pong per pixel/region
for i in range(n):
    count = d_counts[i].get()        # D2H sync per iteration
    if count > 0:
        d_out = cp.empty((count,), ...)

# AFTER: single bulk operation
offsets = exclusive_sum(d_counts, synchronize=False)
total = int(offsets[-1].get() + d_counts[-1].get())
d_out = cp.empty((total,), ...)
```

**1b. Python loops over device arrays**
```python
# BEFORE: element-wise Python loop
for i in range(height):
    row_data = d_array[i].get()  # N syncs

# AFTER: bulk kernel or CuPy vectorized op
result = bulk_process_gpu(d_array)  # 1 launch
```

**1c. Mid-pipeline `.get()` / `cp.asnumpy()`**

Any `.get()` or `cp.asnumpy()` that is NOT at the final return of a
pipeline is suspect. Check if the host value is used to make a decision
that feeds back into GPU work. If yes, rewrite to keep the decision on
device.

### Step 2: Synchronization Elimination (HIGH impact)

**2a. Unnecessary `runtime.synchronize()` between same-stream ops**

CUDA guarantees execution order within a single stream. Remove syncs
between consecutive kernel launches or CCCL calls on the same stream.

**2b. Implicit syncs from scalar reads**

Flag: `int(cupy_scalar)`, `float(cupy_scalar)`, `print(cupy_array)`,
`len()` that triggers `.get()`. Replace with `.shape[0]` for length.

### Step 3: Kernel Source Optimization (HIGH impact)

Read every NVRTC kernel source string and check:

**3a. `const __restrict__` on read-only pointers**

Every pointer parameter that the kernel only reads from should have
`const ... __restrict__`.

**3b. Grid-stride loop with ILP**

If the kernel uses a simple grid-stride loop processing one element per
iteration, rewrite with multi-element ILP (4 elements/thread).

**3c. Shared memory bank conflicts**

If shared memory arrays are 32-wide and accessed in columns, add +1
padding.

**3d. Float constant precision**

In fp32 kernels, check for unqualified float constants (`0.0`, `1.0`)
which compile as doubles and force conversion. Use `0.0f` or explicit cast.

**3e. Stencil halo optimization**

For stencil kernels, verify:
- Shared memory tile includes proper halo cells
- Halo loading doesn't cause bank conflicts
- `__syncthreads()` placed correctly after all loads complete
- Tile size maximizes occupancy given shared memory constraints

### Step 4: Launch Configuration (MEDIUM impact)

**4a. Hardcoded block sizes**

Any `block=(256, 1, 1)` or similar literal should use the occupancy API:
```python
grid, block = runtime.launch_config(kernel, item_count)
```

**4b. Wave quantization**

Check if `grid_size` is just barely over `SM_count * max_blocks_per_SM`.
Use grid-stride loops and cap grid size.

**4c. Tiny kernel launches**

Flag any kernel launched with < 32 threads of real work, or launched
per-pixel in a Python loop.

### Step 5: Memory Access Patterns (MEDIUM impact)

**5a. Non-coalesced access**

For 2D rasters, ensure threads in a warp access consecutive columns
(row-major layout). Flag any strided or AoS-like access patterns.

**5b. Stencil data reuse**

For multi-pass stencil operations, verify intermediate data stays in
L2 cache where possible.

### Step 6: CCCL/CuPy Tier Optimization (MEDIUM impact)

Check every CuPy operation against what CCCL offers:

| Operation | Current | Better | Why |
|-----------|---------|--------|-----|
| `cp.cumsum()` | CuPy | OK | Marginal CCCL advantage |
| `cp.arange(n)` | CuPy | `counting_iterator` | Zero allocation |
| Element-wise + reduce | 2 kernels | `transform_iterator` + reduce | Kernel fusion |
| Custom sort | CuPy argsort | CCCL radix_sort | Faster, stable |

### Step 7: Dtype Awareness (MEDIUM impact)

Check that kernels handle native raster dtypes correctly:

- Are uint8 rasters being unnecessarily upcast to float64?
- Is the kernel source templated on the actual dtype?
- Are intermediate computations using appropriate precision?

---

## Output Format

For each finding, produce:

```
### [PRIORITY] Finding Title

**File:** `path/to/file.py:LINE`
**Impact:** Brief explanation of why this matters and expected improvement
**Category:** (host-device | sync | kernel-source | launch-config | memory-access | tier | dtype)

**Before:**
\```python (or c)
<exact current code>
\```

**After:**
\```python (or c)
<concrete rewrite>
\```
```

Sort findings by priority: CRITICAL > HIGH > MEDIUM > LOW.

At the end, produce a summary table:

```
| # | Priority | Category | File:Line | Finding | Est. Impact |
|---|----------|----------|-----------|---------|-------------|
```

---

## Rules

- **Read before recommending.** Never suggest changes to code you
  haven't read. Always quote the exact current code in "Before."
- **Concrete rewrites only.** Every finding must have a "Before" and
  "After" block. No vague advice like "consider optimizing."
- **Don't break correctness.** Verify dtype changes don't lose precision
  where it matters.
- **One file at a time.** If the target imports GPU code from other
  modules, note cross-file findings but don't rewrite imported modules
  without being asked.
- **Respect existing patterns.** Use `runtime.launch()`, `runtime.pointer()`,
  `make_kernel_cache_key()`, and other vibespatial conventions.
