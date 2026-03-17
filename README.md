# vibespatial-raster

GPU-first raster processing for [vibeSpatial](https://github.com/jarmak-personal/vibeSpatial). Custom NVRTC kernels and CCCL primitives for every operation — algebra, focal, zonal stats, labeling, rasterize, polygonize — with CPU fallbacks via scipy and rasterio.

## Operations

| Category | Operations | GPU Backend |
|---|---|---|
| **IO** | GeoTIFF, COG, JPEG2000 read/write | nvImageCodec (zero-copy decode to device) |
| **Algebra** | add, subtract, multiply, divide, apply, where, classify | CuPy element-wise |
| **Focal** | convolution, Gaussian filter, slope, aspect | NVRTC shared-memory stencil kernels |
| **Zonal** | count, sum, mean, min, max, std, median | CCCL segmented reduce |
| **Rasterize** | vector-to-raster (point-in-polygon) | NVRTC per-pixel PIP kernel |
| **Label** | connected components (4/8 connectivity) | NVRTC union-find |
| **Morphology** | erode, dilate, open, close, sieve | NVRTC tiled kernels |
| **Polygonize** | raster-to-vector (marching squares) | NVRTC classify + emit edges |

## Install

```bash
pip install vibespatial-raster            # core (CPU-only, scipy fallback)
pip install vibespatial-raster[io]        # + rasterio for GeoTIFF/COG
pip install vibespatial-raster[gpu-io]    # + nvImageCodec + CuPy for GPU-native decode
```

For development:

```bash
uv sync --group dev
```

## Usage

```python
from vibespatial.raster import read_raster, raster_slope, zonal_stats

# Read a GeoTIFF (GPU-native decode when available, rasterio fallback)
raster = read_raster("elevation.tif")

# Terrain analysis — NVRTC stencil kernels on GPU
slope = raster_slope(raster)

# Per-zone statistics — CCCL segmented reduce on GPU
stats = zonal_stats(zones, slope, ["mean", "max", "std"])
```

### Connected components + polygonize

```python
from vibespatial.raster import label_connected_components, sieve_filter, polygonize_to_gdf

labels = label_connected_components(binary_raster, connectivity=8)
labels = sieve_filter(labels, min_size=50)
gdf = polygonize_to_gdf(labels)
```

### Zero-copy GPU pipeline

`OwnedRasterArray` tracks HOST/DEVICE residency. Move data to GPU once, then all operations stay on device:

```python
from vibespatial.raster import from_numpy, raster_add

a = from_numpy(data_a, affine=affine, crs="EPSG:4326")
b = from_numpy(data_b, affine=affine, crs="EPSG:4326")
a.move_to("DEVICE")
b.move_to("DEVICE")

result = raster_add(a, b)  # no host-device round-trip
```

### GPU-native IO

When nvImageCodec is available, `read_raster()` decodes GeoTIFF and JPEG2000 directly to GPU memory — no host copy, no rasterio:

```python
raster = read_raster("large_image.tif")  # → OwnedRasterArray(DEVICE)
```

## Architecture

- **Namespace package** — installs as `vibespatial.raster` via `pkgutil.extend_path`, independent release cadence from core
- **Zero-copy NVRTC kernels** — no cuCIM, no CuPy ndimage; all GPU ops are custom CCCL/NVRTC
- **OwnedRasterArray** — mirrors core's `OwnedGeometryArray` (HOST/DEVICE residency, diagnostic events, nodata mask)
- **Dual IO** — HYBRID path (rasterio on CPU) + GPU_NATIVE path (nvImageCodec direct-to-device)
- **CPU fallback** — scipy.ndimage, rasterio.features for every operation; tests validate GPU against CPU

## Test

```bash
uv run pytest                              # all CPU tests
uv run pytest -m gpu                       # GPU kernel tests (requires CUDA)
uv run pytest tests/test_raster_algebra.py # specific module
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
