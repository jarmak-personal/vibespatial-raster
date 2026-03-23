---
name: new-kernel-checklist
description: "PROACTIVELY USE THIS SKILL when adding a new GPU kernel, NVRTC kernel, CCCL primitive wrapper, or any new GPU-dispatched operation to src/vibespatial/raster/. This checklist ensures every compilation, caching, dispatch, test, and documentation step is completed. Trigger on: \"new kernel\", \"add kernel\", \"implement kernel\", \"write kernel\", \"create kernel\", \"new GPU operation\", \"add GPU\", \"scaffold kernel\"."
user-invocable: true
argument-hint: <kernel-name or description of the new operation>
---

# New Kernel Checklist — vibespatial-raster

You are adding a new GPU kernel or GPU-dispatched operation. This checklist
ensures nothing is missed. Work through each section in order — every item
marked **[REQUIRED]** must be completed before the kernel can land.

Target kernel: **$ARGUMENTS**

---

## Phase 1: Classification and Design

### 1.1 Tier Classification [REQUIRED]

Classify your operation:

```
Is the inner loop raster-specific (stencil, CCL merge, marching-squares, PIP)?
  -> Yes: Custom NVRTC kernel
  -> No: Is it segmented (per-zone reduction, sort, scan)?
    -> Yes: CCCL / cuda.compute
    -> No: Is it element-wise/gather/scatter?
      -> Yes: CuPy
      -> No: Custom NVRTC kernel
```

Record: `Tier: ___  Rationale: ___`

### 1.2 Operation Type [REQUIRED]

| Type | Examples | Pattern |
|------|----------|---------|
| **Per-pixel** | algebra ops, classify | 1D grid-stride loop |
| **Focal/Stencil** | convolve, slope, aspect, morphology | 2D tiles + shared memory halo |
| **Iterative** | CCL (union-find) | Multi-phase convergence loop |
| **Segmented** | zonal stats | CCCL segmented_reduce |
| **Geometric** | rasterize, polygonize | Custom NVRTC with algorithm-specific structure |

Record: `Type: ___`

### 1.3 Dtype Support [REQUIRED]

Which dtypes must this kernel handle?
- uint8, int16, int32, float32, float64?
- Will the kernel source be templated on dtype?

Record: `dtypes: (___)`

---

## Phase 2: Kernel Source (Custom NVRTC only — skip for CCCL/CuPy)

### 2.1 Write Kernel Source String [REQUIRED for NVRTC]

Define as a module-level constant. Key rules:

- [ ] Source is a Python string constant: `_KERNEL_SOURCE = """..."""`
- [ ] Use `{dtype}` placeholder for dtype templating
- [ ] All read-only pointers use `const ... * __restrict__`
- [ ] Nodata handling via nullable `nodata_mask` pointer parameter
- [ ] Use `__syncthreads()` after shared memory loads (stencil kernels)
- [ ] Use SoA layout where applicable (separate arrays, not interleaved)
- [ ] Float constants use type-correct suffix

### 2.2 Define Kernel Names Tuple [REQUIRED for NVRTC]

```python
_KERNEL_NAMES = ("my_kernel_main",)
```

Every `extern "C" __global__` function in the source must appear here.

### 2.3 Dtype Variants [REQUIRED for NVRTC]

If the kernel needs to handle multiple dtypes:

```python
def _get_kernel_source(dtype_name: str) -> str:
    return _KERNEL_SOURCE_TEMPLATE.format(dtype=dtype_name)
```

---

## Phase 3: Compilation and Caching

### 3.1 Kernel Compilation (NVRTC) [REQUIRED for NVRTC]

```python
from vibespatial.cuda_runtime import get_cuda_runtime, make_kernel_cache_key

def _compile_my_kernel(dtype_name: str = "float"):
    source = _get_kernel_source(dtype_name)
    runtime = get_cuda_runtime()
    cache_key = make_kernel_cache_key(f"my-kernel-{dtype_name}", source)
    return runtime.compile_kernels(
        cache_key=cache_key,
        source=source,
        kernel_names=_KERNEL_NAMES,
    )
```

- [ ] Cache key includes dtype suffix
- [ ] Uses `make_kernel_cache_key()` (SHA1-based)
- [ ] No compilation in hot loops (compile once, launch many)

---

## Phase 4: Dispatch Wiring

### 4.1 Public API Function [REQUIRED]

```python
def my_operation(raster, *, use_gpu: bool | None = None, **kwargs):
    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)
    if use_gpu:
        return _my_operation_gpu(raster, **kwargs)
    else:
        return _my_operation_cpu(raster, **kwargs)
```

- [ ] Accepts `use_gpu: bool | None = None` parameter
- [ ] Auto-dispatch via `_should_use_gpu()` heuristic
- [ ] GPU path with CPU fallback

### 4.2 GPU Implementation [REQUIRED]

- [ ] Uses `runtime.launch_config()` for occupancy-based sizing (NEVER hardcode block)
- [ ] Uses correct parameter type constants (`KERNEL_PARAM_PTR`, `KERNEL_PARAM_I32`, etc.)
- [ ] No unnecessary `runtime.synchronize()` between same-stream operations
- [ ] Handles OwnedRasterArray residency correctly
- [ ] Logs RasterDiagnosticEvent

### 4.3 CPU Fallback [REQUIRED]

- [ ] Uses scipy or numpy for computation
- [ ] Produces identical results to GPU path
- [ ] Handles nodata consistently

---

## Phase 5: Tests

### 5.1 Unit Tests [REQUIRED]

Create or update `tests/test_raster_{module}.py`:

- [ ] CPU test (always runs, no GPU needed)
- [ ] `@pytest.mark.gpu` test that validates GPU matches CPU
- [ ] Tests null/empty inputs
- [ ] Tests nodata propagation
- [ ] Tests edge cases (single pixel, 1-row raster, etc.)
- [ ] Tests native dtypes (uint8, float32, float64 at minimum)

### 5.2 Auto-Dispatch Test [REQUIRED]

- [ ] Test that auto-dispatch works (falls back to CPU gracefully)
- [ ] Test that explicit `use_gpu=True` and `use_gpu=False` both work

---

## Phase 6: Integration

### 6.1 Package Exports [REQUIRED]

Update `src/vibespatial/raster/__init__.py`:

- [ ] Add to `__all__`
- [ ] Add to `__getattr__` lazy-import dispatcher

### 6.2 Documentation [REQUIRED]

- [ ] Docstring on public function
- [ ] Update AGENTS.md if the operation changes the codebase structure

### 6.3 Zero-Copy Compliance [REQUIRED]

- [ ] Run `uv run python scripts/check_zero_copy.py --all`
- [ ] No new violations (or baseline updated with justification)

---

## Phase 7: Quality Gates

### 7.1 Run /cuda-optimizer [RECOMMENDED]

Invoke the `cuda-optimizer` skill on your kernel file to verify:
- No unnecessary host round-trips
- No redundant synchronizations
- Efficient memory access patterns

### 7.2 Run /gpu-code-review [RECOMMENDED]

Invoke the `gpu-code-review` skill to catch:
- Anti-patterns (hardcoded block sizes, missing `__restrict__`)
- Memory management issues
- Synchronization bugs

### 7.3 Run /pre-land-review [REQUIRED before commit]

Final gate. Must pass before creating a git commit.

---

## Quick Reference: Files to Touch

| What | File(s) |
|------|---------|
| Kernel source | `src/vibespatial/raster/kernels/{kernel}.py` or inline in module |
| GPU implementation | `src/vibespatial/raster/{module}.py` |
| CPU implementation | Same module as GPU |
| Public API | Same module (dispatch function) |
| Package exports | `src/vibespatial/raster/__init__.py` |
| Tests | `tests/test_raster_{module}.py` |
| Zero-copy check | `scripts/check_zero_copy.py` (if baseline changes) |

---

## Conditional Checklist Summary

| Phase | NVRTC Kernel | CCCL Primitive | CuPy Op |
|-------|:-:|:-:|:-:|
| 1. Classification | YES | YES | YES |
| 2. Kernel Source | YES | skip | skip |
| 3. Compilation | YES | skip | skip |
| 4. Dispatch | YES | YES | YES |
| 5. Tests | YES | YES | YES |
| 6. Integration | YES | YES | YES |
| 7. Quality Gates | YES | YES | YES |
