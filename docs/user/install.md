# Installation

## Requirements

- Python 3.12+
- vibespatial (core package)
- scipy 1.14+

For raster IO:

- rasterio 1.4+

For GPU acceleration:

- NVIDIA GPU with CUDA 12.x
- CuPy 13+

For GPU-native IO:

- nvidia-nvimgcodec (nvImageCodec)

## Install with pip

```bash
# Core (CPU-only, scipy fallback for all ops)
pip install vibespatial-raster

# With rasterio for GeoTIFF/COG IO
pip install vibespatial-raster[io]

# With GPU-native decode (nvImageCodec + CuPy)
pip install vibespatial-raster[gpu-io]
```

## Install with uv

```bash
# CPU-only
uv sync

# With dev tools
uv sync --group dev
```

## Verifying GPU support

```python
from vibespatial.runtime import has_gpu_runtime

print(has_gpu_runtime())  # True if CuPy + GPU detected
```

When CuPy is not installed or no GPU is available, all operations automatically
fall back to CPU (scipy, rasterio). No code changes required.

## Verifying IO support

```python
from vibespatial.raster import has_rasterio_support, has_nvimgcodec_support

print(has_rasterio_support())    # True if rasterio is installed
print(has_nvimgcodec_support())  # True if nvImageCodec is available
```
