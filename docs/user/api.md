# API Reference

All public symbols are available from `vibespatial.raster`:

```python
from vibespatial.raster import read_raster, zonal_stats, rasterize_owned
```

## Types

### `OwnedRasterArray`

Central raster buffer type with HOST/DEVICE residency.

- `data` — numpy or cupy array, shape `(bands, height, width)`
- `metadata` — `RasterMetadata` (height, width, band_count, dtype, nodata, affine, crs)
- `residency` — current location (`"HOST"` or `"DEVICE"`)
- `move_to(target)` — transfer between HOST and DEVICE
- `diagnostics` — list of `RasterDiagnosticEvent`

### `RasterMetadata`

Lightweight metadata container.

- `height`, `width`, `band_count` — grid dimensions
- `dtype` — numpy dtype
- `nodata` — sentinel value (or None)
- `affine` — 6-tuple affine transform
- `crs` — coordinate reference system string

### `GridSpec`

Grid definition for rasterization.

- `height`, `width` — output grid size
- `affine` — 6-tuple affine transform
- `crs` — coordinate reference system

### `ZonalSpec` / `ZonalStatistic`

Zonal statistics configuration.

### `PolygonizeSpec`

Polygonize parameters: connectivity, simplify tolerance, drop-nodata flag.

### `RasterDeviceState`

GPU-resident mirror holding `data` and `nodata_mask` cupy arrays.

### `RasterTileSpec`

Chunking specification with overlap for focal operations.

### `RasterWindow`

Rectangular crop specification.

## Factory functions

### `from_numpy(data, affine, crs, nodata=None)`

Create `OwnedRasterArray` from a numpy array.

### `from_device(data, affine, crs, nodata=None)`

Create `OwnedRasterArray` from a cupy array (DEVICE residency).

## IO

### `read_raster(path, window=None)`

Read a raster file. Tries GPU-native (nvImageCodec) first, falls back to rasterio.

### `read_raster_metadata(path)`

Read only metadata (no pixel data). Requires rasterio.

### `write_raster(path, raster, driver="GTiff", **kwargs)`

Write `OwnedRasterArray` to a GeoTIFF file. Requires rasterio.

### `has_rasterio_support()`

Returns `True` if rasterio is importable.

### `has_nvimgcodec_support()`

Returns `True` if nvImageCodec is available.

## Algebra

All algebra functions accept `OwnedRasterArray` and propagate nodata automatically.

### `raster_add(a, b)` / `raster_subtract(a, b)` / `raster_multiply(a, b)` / `raster_divide(a, b)`

Element-wise binary operations.

### `raster_apply(raster, func)`

Apply an element-wise function to raster data.

### `raster_where(raster, mask, nodata)`

Conditional masking.

### `raster_classify(raster, breaks, labels)`

Discrete classification by value breaks.

### `raster_convolve(raster, kernel)`

2D convolution with a user-supplied kernel. NVRTC shared-memory kernel on GPU.

### `raster_gaussian_filter(raster, sigma)`

Gaussian smoothing. Builds kernel from sigma, then dispatches to `raster_convolve`.

### `raster_slope(raster)` / `raster_aspect(raster)`

Terrain analysis from elevation data. NVRTC stencil kernels on GPU.

## Zonal

### `zonal_stats(zones, values, stats)`

Compute per-zone statistics. `stats` is a list of strings: `"count"`, `"sum"`, `"mean"`, `"min"`, `"max"`, `"std"`, `"median"`.

Returns a dict mapping `zone_id -> {stat_name: value}`.

### `zonal_stats_gdf(gdf, raster, stats)`

GeoDataFrame integration — rasterizes GeoDataFrame zones, then computes stats.

## Rasterize

### `rasterize_owned(geometries, values, grid_spec)`

Auto-dispatching rasterize. GPU path when available.

### `rasterize_gpu(geometries, values, grid_spec)`

Force GPU path (NVRTC per-pixel point-in-polygon kernel).

### `rasterize_cpu(geometries, values, grid_spec)`

Force CPU path (rasterio.features.rasterize).

## Label

### `label_connected_components(raster, connectivity=4)`

Auto-dispatching CCL. Returns labeled `OwnedRasterArray`.

### `label_gpu(data, connectivity=4)`

Force GPU path (NVRTC union-find).

### `sieve_filter(labels, min_size)`

Remove connected components smaller than `min_size` pixels.

### `raster_morphology(raster, operation, iterations=1)`

Morphological operation (`"erode"` or `"dilate"`).

### `morphology_gpu(data, operation, iterations=1)`

Force GPU path for morphology (NVRTC tiled kernels).

## Polygonize

### `polygonize_owned(raster)`

Auto-dispatching polygonize. Returns list of `(geometry, value)` tuples.

### `polygonize_gpu(data)`

Force GPU path (marching-squares NVRTC kernels).

### `polygonize_to_gdf(raster, affine=None, crs=None)`

Polygonize to GeoDataFrame with geometry column and value column.

### `plan_polygonize_pipeline(raster)`

Returns a fusion plan (list of `PipelineStep`) for the polygonize pipeline.
