# Design Decisions

Key architecture decisions for vibespatial-raster.

---

## 1. Standalone package via namespace extension

vibespatial-raster installs as `vibespatial.raster` using `pkgutil.extend_path`
in vibespatial core's `__init__.py`. This allows:
- Independent release cadence from core vibespatial
- Users who don't need raster don't install raster deps (scipy, rasterio)
- Clean dependency direction: raster depends on core, never the reverse

Core vibespatial modules used: `residency`, `runtime`, `cuda_runtime`, `fusion`.

## 2. Drop cuCIM -- custom CCCL kernels for labeling

The original Phase 8 plan specified cuCIM (a RAPIDS medical imaging library) for
connected-component labeling. Decision: **build from scratch using CCCL primitives
and custom NVRTC kernels.** No cuCIM, no CuPy ndimage.

GPU CCL algorithm (label equivalence / union-find):
1. Init: each pixel gets its own label (CCCL CountingIterator)
2. Local merge: NVRTC kernel, each thread checks 4/8 neighbors, atomic min
3. Pointer jumping: NVRTC kernel, compress label chains to roots
4. Iterate merge+jump until convergence (typically 3-5 passes)
5. Relabel: CCCL sort + unique_by_key for dense sequential labels

Current state: CPU baseline via scipy.ndimage. GPU kernel is next.

## 3. OwnedRasterArray follows OwnedGeometryArray pattern

Central type mirroring vibespatial's geometry buffer design:
- HOST/DEVICE residency with `move_to()` and `DiagnosticEvent` tracking
- Data stored in native dtype (not always fp64 like geometry coords)
- Band-first layout: `(bands, height, width)` matching rasterio convention
- Affine transform replaces coordinate arrays for spatial reference
- Nodata mask lazily computed from sentinel value

## 4. No new KernelClass values needed

| Raster Operation | KernelClass | Rationale |
|---|---|---|
| Raster algebra (local ops) | COARSE | Element-wise, memory-bound |
| Focal ops (convolution, slope) | METRIC | Neighborhood accumulation |
| Zonal statistics | METRIC | Segmented reductions |
| Rasterize (vector-to-raster) | CONSTRUCTIVE | Creates new representation |
| Connected component labeling | COARSE | Integer labeling |
| Polygonize (raster-to-vector) | CONSTRUCTIVE | Creates geometry from raster |

## 5. Raster crossover thresholds are very low

Even modest rasters have millions of pixels (1000x1000 = 1M):

| KernelClass | Pixels | Grid Size |
|---|---|---|
| COARSE (algebra, labeling) | 10,000 | ~100x100 |
| METRIC (focal, zonal) | 50,000 | ~224x224 |
| CONSTRUCTIVE (rasterize, polygonize) | 100,000 | ~316x316 |

## 6. Raster IO is HYBRID (with GPU_NATIVE alternative)

GeoTIFF/COG read via rasterio on host, then transfer to device. Same hybrid
pattern as Shapefile in vibespatial core. A GPU_NATIVE decode path is now
available via nvImageCodec for GeoTIFF and JPEG2000 (see decision 9 below).

## 7. Standalone function API surface

All operations are standalone functions, not methods on OwnedRasterArray:
```python
from vibespatial.raster import read_raster, zonal_stats, rasterize_owned
```

GeoDataFrame integration via helper functions that accept GeoDataFrame as
first argument (e.g., `zonal_stats_gdf(gdf, raster, stats)`).

## 8. No xarray core dependency

xarray/rioxarray interop is an optional stretch goal (o17.8.19), not a core
design requirement. Can be added later without affecting the core design.

## 9. GPU-Native Raster IO via nvImageCodec

GeoTIFF and JPEG2000 files can be decoded directly to GPU memory using
NVIDIA's nvImageCodec library (wraps nvTIFF + nvJPEG2000). This adds a
GPU_NATIVE decode tier that sits above the existing HYBRID rasterio path:

1. **GPU_NATIVE** -- nvImageCodec decodes file -> device memory -> `OwnedRasterArray(DEVICE)` via `from_device()` (zero-copy)
2. **HYBRID** (existing) -- rasterio decodes on CPU -> numpy -> `OwnedRasterArray(HOST)` -> optional GPU transfer

The `read_raster()` dispatcher tries GPU_NATIVE first when available, falling
back transparently to HYBRID. GeoTIFF metadata (CRS, affine transform) is
extracted from nvImageCodec's GeoKey API and parsed by `geokeys.py`. The
GDAL_NODATA tag is supplemented from a lightweight rasterio metadata-only read
when needed.

Supported compressions: LZW, DEFLATE, JPEG-in-TIFF, JPEG2000-in-TIFF, uncompressed.
Supported formats: GeoTIFF (including BigTIFF), JPEG2000 (JP2/J2K/HTJ2K).

Install: `pip install nvidia-nvimgcodec-cu12[all]` or `uv sync --extra gpu-io`
