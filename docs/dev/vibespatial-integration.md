# vibespatial Core Integration

vibespatial-raster depends on {external+vibespatial:doc}`vibespatial <index>`
for core GPU infrastructure. This document maps which core modules are
used and why. See also the
{external+vibespatial:doc}`architecture overview <dev/architecture>` and
{external+vibespatial:doc}`kernel developer guide <dev/kernels>`.

## Core modules used

| vibespatial module | Used by | Purpose |
|---|---|---|
| `residency` | `buffers.py` | `Residency`, `TransferTrigger`, `select_residency_plan` |
| `runtime` | `buffers.py` | `RuntimeSelection`, `has_gpu_runtime` |
| `cuda_runtime` | `algebra.py`, `rasterize.py` | `get_cuda_runtime`, `compile_kernels`, `KERNEL_PARAM_*` |
| `fusion` | `polygonize.py` | `PipelineStep`, `StepKind`, `plan_fusion` |

## Core changes required in vibespatial

Two small additions to vibespatial core enable the standalone package:

1. **`__init__.py`**: `pkgutil.extend_path(__path__, __name__)` at the top,
   allowing Python to discover `vibespatial.raster` from a separate installation.

2. **`fusion.py`**: `StepKind.RASTER = "raster"` enum value for raster
   pipeline steps.

3. **`io_support.py`**: `IOFormat.GEOTIFF` and `IOFormat.COG` enum values
   with `HYBRID` path classification (optional -- only needed if raster
   IO plans go through the core IO support matrix).

## Module structure

```
vibespatial.raster/
    __init__.py          # Public API with lazy imports
    buffers.py           # OwnedRasterArray, GridSpec, ZonalSpec, PolygonizeSpec
    io.py                # read_raster, write_raster (requires rasterio)
    algebra.py           # Local ops (CuPy) + focal ops (NVRTC stencil)
    zonal.py             # Zonal statistics via segmented reduce
    rasterize.py         # Vector-to-raster (CPU + GPU paths)
    label.py             # Connected component labeling, sieve, morphology
    polygonize.py        # Raster-to-vector pipeline
    kernels/
        __init__.py
        focal.py         # NVRTC convolution kernel source
```
