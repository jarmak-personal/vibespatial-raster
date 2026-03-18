"""vibespatial.raster -- GPU-first raster processing alongside vector data.

Standalone package that extends vibespatial with raster operations:
  - OwnedRasterArray with HOST/DEVICE residency
  - GeoTIFF/COG IO via rasterio
  - Raster algebra (local + focal) on GPU
  - Zonal statistics via CCCL segmented reduce
  - Vector-to-raster rasterize via NVRTC kernel
  - Raster-to-vector polygonize pipeline

Install: pip install vibespatial-raster
Usage:   from vibespatial.raster import read_raster, zonal_stats
"""

__version__ = "0.1.2"

from vibespatial.raster.buffers import (
    GridSpec,
    OwnedRasterArray,
    PolygonizeSpec,
    RasterDeviceState,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    RasterMetadata,
    RasterTileSpec,
    RasterWindow,
    ZonalSpec,
    ZonalStatistic,
    from_device,
    from_numpy,
)

__all__ = [
    # Types
    "OwnedRasterArray",
    "RasterDeviceState",
    "RasterDiagnosticEvent",
    "RasterDiagnosticKind",
    "RasterMetadata",
    "RasterTileSpec",
    "RasterWindow",
    "GridSpec",
    "ZonalSpec",
    "ZonalStatistic",
    "PolygonizeSpec",
    # Factory
    "from_device",
    "from_numpy",
    # IO (lazy imports to avoid hard rasterio dep)
    "read_raster",
    "read_raster_metadata",
    "write_raster",
    "has_rasterio_support",
    "has_nvimgcodec_support",
    # Algebra
    "raster_add",
    "raster_subtract",
    "raster_multiply",
    "raster_divide",
    "raster_apply",
    "raster_where",
    "raster_classify",
    "raster_convolve",
    "raster_gaussian_filter",
    "raster_slope",
    "raster_aspect",
    # Zonal
    "zonal_stats",
    "zonal_stats_gdf",
    # Rasterize
    "rasterize_owned",
    "rasterize_cpu",
    "rasterize_gpu",
    # Label
    "label_connected_components",
    "label_gpu",
    "morphology_gpu",
    "sieve_filter",
    "raster_morphology",
    # Polygonize
    "polygonize_owned",
    "polygonize_gpu",
    "polygonize_to_gdf",
    "plan_polygonize_pipeline",
    # NVRTC cache management (from vibespatial.cuda_runtime)
    "clear_nvrtc_cache",
    "nvrtc_cache_stats",
]


def __getattr__(name):
    """Lazy imports for optional-dependency functions."""
    # IO functions (require rasterio)
    if name in (
        "read_raster",
        "read_raster_metadata",
        "write_raster",
        "has_rasterio_support",
        "has_nvimgcodec_support",
    ):
        from vibespatial.raster import io

        return getattr(io, name)
    # Algebra functions (require CuPy for GPU ops)
    if name in (
        "raster_add",
        "raster_subtract",
        "raster_multiply",
        "raster_divide",
        "raster_apply",
        "raster_where",
        "raster_classify",
        "raster_convolve",
        "raster_gaussian_filter",
        "raster_slope",
        "raster_aspect",
    ):
        from vibespatial.raster import algebra

        return getattr(algebra, name)
    # Zonal
    if name in ("zonal_stats", "zonal_stats_gdf"):
        from vibespatial.raster import zonal

        return getattr(zonal, name)
    # Rasterize
    if name in ("rasterize_owned", "rasterize_cpu", "rasterize_gpu"):
        from vibespatial.raster import rasterize

        return getattr(rasterize, name)
    # Label
    if name in (
        "label_connected_components",
        "label_gpu",
        "morphology_gpu",
        "sieve_filter",
        "raster_morphology",
    ):
        from vibespatial.raster import label

        return getattr(label, name)
    # Polygonize
    if name in (
        "polygonize_owned",
        "polygonize_gpu",
        "polygonize_to_gdf",
        "plan_polygonize_pipeline",
    ):
        from vibespatial.raster import polygonize

        return getattr(polygonize, name)
    # NVRTC cache management
    if name in ("clear_nvrtc_cache", "nvrtc_cache_stats"):
        from vibespatial.cuda_runtime import clear_nvrtc_cache, nvrtc_cache_stats

        _cache_api = {
            "clear_nvrtc_cache": clear_nvrtc_cache,
            "nvrtc_cache_stats": nvrtc_cache_stats,
        }
        return _cache_api[name]
    raise AttributeError(f"module 'vibespatial.raster' has no attribute {name!r}")
