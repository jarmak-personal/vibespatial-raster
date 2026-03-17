# vibeSpatial Integration

vibespatial-raster is a standalone namespace extension package for
{external+vibespatial:doc}`vibeSpatial <index>`. It installs as
`vibespatial.raster` and shares core infrastructure with the vector side.

## Namespace packaging

vibespatial-raster uses `pkgutil.extend_path` to extend the `vibespatial`
namespace. This means:

- Independent release cadence from vibespatial core
- Users who don't need raster don't install raster dependencies
- Clean dependency direction: raster depends on core, never the reverse

```python
# After installing both packages:
from vibespatial.geometry import OwnedGeometryArray  # from core
from vibespatial.raster import OwnedRasterArray       # from raster
```

## Shared core modules

| Core module | Used by | Purpose |
|---|---|---|
| `residency` | `buffers.py` | `Residency`, `TransferTrigger`, `select_residency_plan` |
| `runtime` | `buffers.py` | `RuntimeSelection`, `has_gpu_runtime` |
| `cuda_runtime` | `algebra.py`, `rasterize.py` | `get_cuda_runtime`, `compile_kernels` |
| `fusion` | `polygonize.py` | `PipelineStep`, `StepKind`, `plan_fusion` |

## OwnedRasterArray

The central type mirrors vibespatial's `OwnedGeometryArray` pattern
(see {external+vibespatial:doc}`API reference <user/api>`):

- **HOST/DEVICE residency** with `move_to()` and diagnostic event tracking
- **Band-first layout**: `(bands, height, width)` matching rasterio convention
- **Native dtype**: data stored in original dtype (not always fp64 like geometry coords)
- **Affine transform**: replaces coordinate arrays for spatial reference
- **Nodata mask**: lazily computed from sentinel value

```python
from vibespatial.raster import from_numpy, raster_slope

# Create from numpy array
raster = from_numpy(elevation_data, affine=affine, crs="EPSG:4326", nodata=-9999)

# Move to GPU once, operate many times
raster.move_to("DEVICE")
slope = raster_slope(raster)  # stays on GPU
```

## Zero-copy IO

When nvImageCodec is available, `read_raster()` decodes GeoTIFF/JPEG2000
directly to GPU memory, producing an `OwnedRasterArray(DEVICE)` with no
host-device copies:

```python
from vibespatial.raster import read_raster

# GPU-native path: file -> nvImageCodec -> device memory -> OwnedRasterArray(DEVICE)
raster = read_raster("large_image.tif")

# If nvImageCodec unavailable, falls back to rasterio (HOST) transparently
```
