# Architecture

## Overview

vibespatial-raster is a standalone namespace extension package that adds raster
processing to vibespatial. It installs as `vibespatial.raster` via
`pkgutil.extend_path` and depends on vibespatial core for GPU infrastructure.

## Module structure

| Module | Purpose |
|---|---|
| `buffers.py` | `OwnedRasterArray`, `GridSpec`, `ZonalSpec`, `PolygonizeSpec`, factory functions |
| `io.py` | `read_raster`, `write_raster` — rasterio HYBRID path |
| `nvimgcodec_io.py` | GPU-native decode via nvImageCodec (GeoTIFF, JPEG2000) |
| `geokeys.py` | GeoTIFF GeoKey parsing (affine, CRS extraction) |
| `algebra.py` | Local ops (CuPy element-wise) + focal ops (NVRTC stencils) |
| `zonal.py` | Zonal statistics via CCCL segmented reduce |
| `rasterize.py` | Vector-to-raster via NVRTC per-pixel PIP kernel |
| `label.py` | Connected component labeling, sieve filter, morphology |
| `polygonize.py` | Raster-to-vector via marching-squares NVRTC kernels |
| `kernels/` | Raw NVRTC kernel source strings |

## Design principles

1. **Zero-copy CCCL/NVRTC** — no cuCIM, no CuPy ndimage. All GPU operations
   use custom NVRTC kernels or CCCL primitives.

2. **OwnedRasterArray pattern** — mirrors `OwnedGeometryArray` from vibespatial
   core. HOST/DEVICE residency with diagnostic event tracking.

3. **Standalone function API** — all operations are free functions, not methods
   on buffer types. This keeps the API flat and composable.

4. **Dual IO paths** — HYBRID (rasterio on CPU, then transfer) and GPU_NATIVE
   (nvImageCodec direct-to-device). `read_raster()` dispatches automatically.

5. **CPU fallback** — every GPU operation has a scipy/rasterio CPU path.
   Tests validate GPU output against CPU reference.

## Kernel classification

Raster operations map to vibespatial's `KernelClass` system:

| Operation | KernelClass | Crossover | Rationale |
|---|---|---|---|
| Algebra (local ops) | COARSE | 10k pixels | Element-wise, memory-bound |
| Focal (convolution, slope) | METRIC | 50k pixels | Neighborhood accumulation |
| Zonal statistics | METRIC | 50k pixels | Segmented reductions |
| Rasterize | CONSTRUCTIVE | 100k pixels | Creates new representation |
| CCL (labeling) | COARSE | 10k pixels | Integer labeling |
| Polygonize | CONSTRUCTIVE | 100k pixels | Creates geometry from raster |

Even modest rasters (1000x1000 = 1M pixels) far exceed all crossover thresholds,
so GPU dispatch is the common path.

## IO architecture

```
read_raster(path)
  ├─ GPU_NATIVE:  nvImageCodec → device memory → OwnedRasterArray(DEVICE)
  │   └─ geokeys.py parses CRS + affine from GeoKey API
  └─ HYBRID:      rasterio → numpy → OwnedRasterArray(HOST)
      └─ optional move_to("DEVICE") for GPU ops
```

Supported formats: GeoTIFF (including BigTIFF), JPEG2000 (JP2/J2K/HTJ2K).
Supported compressions: LZW, DEFLATE, JPEG-in-TIFF, JPEG2000-in-TIFF, uncompressed.

## Dependency direction

```
vibespatial (core)
  ├── residency, runtime, cuda_runtime, fusion
  │
  └── vibespatial-raster (this package)
        ├── scipy (CPU fallback)
        ├── rasterio (optional: IO)
        ├── nvidia-nvimgcodec (optional: GPU-native IO)
        └── cupy-cuda12x (optional: GPU compute)
```

Core changes required in vibespatial:
1. `pkgutil.extend_path(__path__, __name__)` in `__init__.py`
2. `StepKind.RASTER = "raster"` in `fusion.py`
