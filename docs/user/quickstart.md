# Quickstart

## Reading a raster

```python
from vibespatial.raster import read_raster

# read_raster tries GPU-native decode first, falls back to rasterio
raster = read_raster("elevation.tif")

print(raster.metadata.height, raster.metadata.width)
print(raster.metadata.crs)
print(raster.residency)  # HOST or DEVICE
```

## Raster algebra

```python
from vibespatial.raster import raster_add, raster_multiply, raster_where

# Element-wise operations (nodata propagated automatically)
result = raster_add(raster_a, raster_b)
scaled = raster_multiply(raster, 2.0)

# Conditional masking
masked = raster_where(raster, condition_mask, nodata=-9999)
```

## Focal operations

```python
from vibespatial.raster import raster_slope, raster_aspect, raster_convolve

# Terrain analysis (NVRTC shared-memory stencils on GPU)
slope = raster_slope(elevation)
aspect = raster_aspect(elevation)

# Custom convolution kernel
import numpy as np
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
smoothed = raster_convolve(raster, kernel)
```

## Zonal statistics

```python
from vibespatial.raster import zonal_stats, zonal_stats_gdf

# Per-zone aggregation (CCCL segmented reduce on GPU)
stats = zonal_stats(zones, values, ["count", "mean", "max", "std"])

# GeoDataFrame integration
stats_gdf = zonal_stats_gdf(gdf, raster, ["mean", "min", "max"])
```

## Connected component labeling

```python
from vibespatial.raster import label_connected_components, sieve_filter

# Label connected regions (NVRTC union-find on GPU)
labels = label_connected_components(binary_raster, connectivity=8)

# Remove small components
sieved = sieve_filter(labels, min_size=50)
```

## Rasterize (vector to raster)

```python
from vibespatial.raster import rasterize_owned, GridSpec

# Burn vector geometries into a raster grid
grid = GridSpec(height=1000, width=1000, affine=affine, crs="EPSG:4326")
rasterized = rasterize_owned(geometries, values, grid)
```

## Polygonize (raster to vector)

```python
from vibespatial.raster import polygonize_to_gdf

# Convert labeled raster to GeoDataFrame (marching-squares on GPU)
gdf = polygonize_to_gdf(labels, affine=affine, crs="EPSG:4326")
```

## GPU residency

All operations work with `OwnedRasterArray`, which tracks HOST/DEVICE residency:

```python
from vibespatial.raster import from_numpy

raster = from_numpy(data, affine=affine, crs="EPSG:4326")
raster.move_to("DEVICE")  # transfer to GPU once

# All subsequent operations stay on GPU — no extra copies
slope = raster_slope(raster)
```
