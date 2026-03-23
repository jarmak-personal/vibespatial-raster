# API Overview

All public symbols are available from `vibespatial.raster`:

```python
from vibespatial.raster import read_raster, zonal_stats, rasterize_owned
```

For the full auto-generated API reference with complete signatures and
docstrings, see the {doc}`/api/vibespatial/raster/index`.

## Module summary

| Module | Description |
|--------|-------------|
| {py:mod}`~vibespatial.raster.buffers` | `OwnedRasterArray`, `RasterMetadata`, `GridSpec`, and other buffer types |
| {py:mod}`~vibespatial.raster.io` | `read_raster`, `write_raster`, rasterio-backed IO |
| {py:mod}`~vibespatial.raster.nvimgcodec_io` | GPU-native IO via nvImageCodec |
| {py:mod}`~vibespatial.raster.algebra` | Element-wise ops, convolution, slope, aspect |
| {py:mod}`~vibespatial.raster.zonal` | Per-zone statistics via CCCL segmented reduce |
| {py:mod}`~vibespatial.raster.rasterize` | Vector-to-raster with GPU/CPU dispatch |
| {py:mod}`~vibespatial.raster.label` | Connected-component labeling, sieve, morphology |
| {py:mod}`~vibespatial.raster.polygonize` | Raster-to-vector via marching squares |
| {py:mod}`~vibespatial.raster.distance` | Distance transforms |
| {py:mod}`~vibespatial.raster.histogram` | Histogram computation |
| {py:mod}`~vibespatial.raster.hydrology` | Flow direction and accumulation |
| {py:mod}`~vibespatial.raster.resample` | Nearest/bilinear/bicubic resampling |
| {py:mod}`~vibespatial.raster.geokeys` | GeoTIFF key parsing |
