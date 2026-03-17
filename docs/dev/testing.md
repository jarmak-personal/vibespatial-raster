# Testing

## Running tests

```bash
# All tests (CPU path — no GPU required)
uv run pytest

# GPU kernel tests only (requires CUDA GPU)
uv run pytest -m gpu

# Specific module
uv run pytest tests/test_raster_algebra.py

# Verbose with short tracebacks
uv run pytest -v --tb=short
```

## Test structure

| File | Coverage | Lines |
|---|---|---|
| `test_geokeys.py` | GeoKey parsing, affine conversion | 153 |
| `test_nvimgcodec_io.py` | nvImageCodec decode, format support | 218 |
| `test_raster_algebra.py` | Add/sub/mul/div, nodata, focal ops | 217 |
| `test_raster_buffers.py` | OwnedRasterArray, residency, diagnostics | 418 |
| `test_raster_io.py` | Read/write via rasterio, metadata | 277 |
| `test_raster_label.py` | CCL, connectivity, sieve, morphology | 672 |
| `test_raster_polygonize.py` | Polygonize CPU/GPU, ring orientation | 416 |
| `test_raster_rasterize.py` | GridSpec, CPU/GPU rasterize | 138 |
| `test_raster_zonal.py` | Zonal stats, GPU/CPU dispatch | 312 |

## GPU test marker

Tests that require a CUDA GPU are marked with `@pytest.mark.gpu`:

```python
@pytest.mark.gpu
def test_label_gpu():
    ...
```

GPU tests auto-skip when CuPy is not available. In CI, only CPU tests run
(no GPU hardware). GPU tests must be run locally before merging.

## Test patterns

- **CPU vs GPU comparison** — run the same operation on CPU and GPU, assert
  results match within tolerance
- **Roundtrip** — write then read, verify data preserved
- **Nodata propagation** — verify nodata sentinel values flow through operations
- **Edge cases** — single-pixel rasters, all-nodata, connectivity variants

## Linting

```bash
# Check
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Format check (CI uses this)
uv run ruff format --check src/ tests/
```

## CI

GitHub Actions runs on every push/PR to `main`:

- **Test job** — Python 3.12 + 3.13 matrix, `pytest --tb=short -m "not gpu"`
- **Lint job** — `ruff check` + `ruff format --check`

No GPU tests in CI (no GPU hardware). GPU tests are a local-only gate.
