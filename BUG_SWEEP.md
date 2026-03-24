# Bug Sweep — 2026-03-23

Comprehensive audit of vibespatial-raster by 6 parallel agents (GPU kernel, Python logic, device residency, test correctness, algorithm correctness, codebase explorer).

**Tally: 5 critical, 7 high, 6 medium, 4 low, 5 test issues = 27 findings**

---

## Critical — Correctness Bugs

- [x] **1. Resample kernel parameter type mismatch** (float32/int16/uint8 with nodata)
  - `src/vibespatial/raster/resample.py:121-136` + `kernels/resample.py`
  - Host passes `KERNEL_PARAM_I32` (4 bytes) for kernel params declared as `short` (2 bytes) or `unsigned char` (1 byte). For float32, passes `KERNEL_PARAM_F64` (8 bytes) for a `float` (4 bytes). Corrupts `nodata_val` and misaligns subsequent `has_nodata` parameter, causing silent wrong results.

- [x] **2. `nodata_mask` property reads stale/uninitialized host data on device-resident rasters**
  - `src/vibespatial/raster/buffers.py:277-290`
  - `nodata_mask` always operates on `self.data` (host numpy). For rasters created via `from_device()`, `self.data` is `np.empty()` (uninitialized). Any code path accessing `nodata_mask` on a device-resident raster gets garbage.
  - Affects: `_focal_stat_cpu`, `raster_morphology_tophat/blackhat`, CPU resample path, `_raster_expression_cpu`.

- [x] **3. `_ensure_host_state` silently ignores CuPy import failure**
  - `src/vibespatial/raster/buffers.py:393-397`
  - If `device_state` exists but CuPy can't import, function returns without materializing host data. `to_numpy()` then returns uninitialized garbage silently.

- [x] **4. Histogram equalize kernel param type mismatch**
  - `src/vibespatial/raster/histogram.py:491-500`
  - `nodata_val` passed as `KERNEL_PARAM_I32` (4 bytes) but kernel declares `const unsigned char nodata_val` (1 byte). Same class of bug as #1.

- [x] **5. Terrain derivative CPU: border pixels get -9999 but `nodata=None`**
  - `src/vibespatial/raster/algebra.py:2043-2059`
  - Output initialized to `-9999.0`, border pixels never overwritten, but result declares `nodata=None`. Downstream consumers can't distinguish border artifacts from real data.

---

## High — Logic / Semantic Bugs

- [x] **6. `_focal_stat_cpu` crashes on multi-band rasters (3D/2D shape mismatch)**
  - `src/vibespatial/raster/algebra.py:1834`
  - `nodata_mask` can be 3D while `data` is squeezed to 2D. Would cause shape mismatch crash.

- [x] **7. Histogram equalize CPU: NaN nodata not handled correctly**
  - `src/vibespatial/raster/histogram.py:109-123`
  - `data[data != raster.nodata]` when `nodata=NaN` includes all NaN values (since `NaN != NaN` is True). NaN values corrupt the normalization range.

- [ ] **8. Histogram equalize: nodata information loss for non-uint8**
  - `src/vibespatial/raster/histogram.py:132,170`
  - Non-uint8 rasters with nodata: equalized output maps nodata pixels to 0 but declares `nodata=None`. Consumers can't identify nodata pixels.

- [ ] **9. Hydrology convergence: may return unconverged result silently**
  - `src/vibespatial/raster/hydrology.py:336-349`
  - Convergence checked every 32 iterations. If `max_iterations` hit without convergence at a batch boundary, last batch's changes are never verified. Result returned as if converged.

- [ ] **10. `raster_divide` double-applies nodata masking**
  - `src/vibespatial/raster/algebra.py:110-125`
  - `safe_divide` replaces inf/nan with `nodata_val`, then `_binary_op` independently overlays nodata masking. Legitimate computed values matching `nodata_val` get spuriously masked.

- [ ] **11. GPU slope/aspect boundary: 0-fill vs CPU edge-replication**
  - `src/vibespatial/raster/kernels/focal.py:161-170`
  - GPU fills out-of-bounds halo with `0.0`; CPU uses `np.pad(mode="edge")`. Border pixels get artificially steep slopes on GPU, creating CPU/GPU inconsistency.

- [ ] **12. Focal std vs zonal std: ddof inconsistency**
  - `kernels/focal.py:763` uses `n-1` (sample std); `zonal.py:160` uses `n` (population std).
  - Same library, different conventions.

---

## Medium — Dispatch / Robustness Bugs

- [ ] **13. `slope/aspect` auto-dispatch uses `_has_cupy()` instead of `_should_use_gpu()`**
  - `src/vibespatial/raster/algebra.py:1120-1121, 1170-1171`
  - Only checks if CuPy is importable, ignoring raster size thresholds and CUDA runtime availability. Will attempt GPU on tiny rasters or when CUDA runtime is unavailable.

- [ ] **14. Polygonize `total_cells` integer overflow for large rasters**
  - `src/vibespatial/raster/kernels/polygonize.py:68`
  - `int total_cells = cell_width * cell_height` overflows int32 for rasters > ~46K x 46K pixels.

- [ ] **15. `morphology_gpu` D->H->D ping-pong**
  - `src/vibespatial/raster/label.py:591-607`
  - Legacy path uses `raster.to_numpy()` + `cp.asarray()` instead of device residency API. The newer `_morphology_nxn_gpu` does this correctly already.

- [ ] **16. Gaussian filter: no validation for `sigma=0`**
  - `src/vibespatial/raster/algebra.py:864-873`
  - `sigma=0` causes division by zero in kernel computation, producing NaN kernel weights.

- [ ] **17. Polygonize pad_value may collide with real data values**
  - `src/vibespatial/raster/polygonize.py:399-409`
  - `pad_value = min(data) - 1.0` could equal an actual data value at float64 extremes, causing that value to be silently excluded from polygonization.

- [ ] **18. `_should_use_gpu` redefined with different thresholds in `algebra.py`**
  - Line 1207: threshold 10,000; Line 1658: threshold 100,000. Later definition shadows earlier, potentially unintentional.

---

## Low — Performance / Architecture Issues

- [ ] **19. Systemic: all 19 GPU functions return HOST-resident results via `cp.asnumpy() + from_numpy()`**
  - Every GPU function forces D->H at output. Any multi-stage pipeline (e.g., label -> sieve -> polygonize) bounces through host between every step. `from_device()` exists but is never used for return values.
  - Affected functions: `_binary_op`, `raster_apply`, `raster_where`, `raster_classify`, `_raster_expression_gpu`, `_gpu_convolve`, `_gpu_slope_aspect`, `_hillshade_gpu`, `_terrain_derivative_gpu`, `_focal_stat_gpu`, `label_gpu`, `morphology_gpu`, `_morphology_nxn_gpu`, `_sieve_gpu`, `_distance_transform_gpu`, `rasterize_gpu`, `_resample_gpu`, `_fill_sinks_gpu`, `_raster_histogram_equalize_gpu`.

- [ ] **20. Multiple implicit syncs via `.size` instead of `.shape[0]`**
  - `histogram.py:250,406,424,385,452`
  - `.size` on CuPy arrays from boolean indexing forces sync. The zonal module already fixed this pattern.

- [ ] **21. `raster_morphology_tophat/blackhat`: host-side difference computation**
  - `src/vibespatial/raster/label.py:1510-1526, 1579-1594`
  - Pulls two GPU results to host just to compute a binary XOR.

- [ ] **22. Non-atomic `*changed = 1` in hydrology/CCL kernels**
  - `kernels/hydrology.py:163`, `kernels/ccl.py:97,108,247`
  - Technically UB per CUDA memory model (benign in practice since all threads write the same 32-bit value).

---

## Test Suite Issues

- [ ] **23. `zonal_stats_gdf` has ZERO test coverage**
  - Exported public function with no tests anywhere.

- [ ] **24. Inconsistent `requires_gpu` across test files**
  - Some files use `pytest.mark.gpu` (marker only, no skip), others use `pytest.mark.skipif`.
  - Files with marker-only: `test_raster_expression.py:17`, `test_raster_hillshade.py:17`.
  - Files with actual skip: `test_raster_distance.py:27`, `test_raster_buffers.py:31`.
  - New GPU tests in marker-only files will crash on CI without GPU instead of being skipped.

- [ ] **25. Missing assertion: `test_0_and_100_percentiles`**
  - `tests/test_raster_histogram.py:201`
  - 100th percentile assertion is commented but never written.

- [ ] **26. Curvature test docstring contradicts assertion**
  - `tests/test_terrain_derivatives.py:234-245`
  - Docstring says "positive curvature" but asserts `< 0.0`. The assertion is correct; the docstring is wrong.

- [ ] **27. Loose GPU test tolerances may hide real bugs**
  - GPU EDT: `atol=0.5` (`test_raster_distance.py:258,269,312,321,330`)
  - Rasterize GPU/CPU agreement: 85% threshold (`test_raster_rasterize.py:133`)
  - Histogram percentile GPU vs CPU: `decimal=0` (`test_raster_histogram.py:339`)
