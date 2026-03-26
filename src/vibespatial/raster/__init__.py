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

__version__ = "0.1.6"

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
    "raster_expression",
    "raster_convolve",
    "raster_gaussian_filter",
    "raster_slope",
    "raster_aspect",
    "raster_hillshade",
    "raster_tri",
    "raster_tpi",
    "raster_curvature",
    # Focal statistics
    "raster_focal_min",
    "raster_focal_max",
    "raster_focal_mean",
    "raster_focal_std",
    "raster_focal_range",
    "raster_focal_variety",
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
    "raster_morphology_tophat",
    "raster_morphology_blackhat",
    "make_structuring_element",
    # Distance Transform
    "raster_distance_transform",
    # Polygonize
    "polygonize_owned",
    "polygonize_gpu",
    "polygonize_to_gdf",
    "plan_polygonize_pipeline",
    # Resample
    "raster_resample",
    # Histogram / CDF / equalization
    "raster_histogram",
    "raster_cumulative_distribution",
    "raster_histogram_equalize",
    "raster_percentile",
    # Hydrology
    "raster_fill_sinks",
    # Dispatch / VRAM budget
    "available_vram_bytes",
    "max_bands_for_budget",
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
        "raster_expression",
        "raster_convolve",
        "raster_gaussian_filter",
        "raster_slope",
        "raster_aspect",
        "raster_hillshade",
        "raster_tri",
        "raster_tpi",
        "raster_curvature",
        "raster_focal_min",
        "raster_focal_max",
        "raster_focal_mean",
        "raster_focal_std",
        "raster_focal_range",
        "raster_focal_variety",
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
        "raster_morphology_tophat",
        "raster_morphology_blackhat",
        "make_structuring_element",
    ):
        from vibespatial.raster import label

        return getattr(label, name)
    # Distance Transform
    if name in ("raster_distance_transform",):
        from vibespatial.raster import distance

        return getattr(distance, name)
    # Polygonize
    if name in (
        "polygonize_owned",
        "polygonize_gpu",
        "polygonize_to_gdf",
        "plan_polygonize_pipeline",
    ):
        from vibespatial.raster import polygonize

        return getattr(polygonize, name)
    # Resample
    if name == "raster_resample":
        from vibespatial.raster import resample

        return getattr(resample, name)
    # Histogram / CDF / equalization
    if name in (
        "raster_histogram",
        "raster_cumulative_distribution",
        "raster_histogram_equalize",
        "raster_percentile",
    ):
        from vibespatial.raster import histogram

        return getattr(histogram, name)
    # Hydrology
    if name in ("raster_fill_sinks",):
        from vibespatial.raster import hydrology

        return getattr(hydrology, name)
    # Dispatch / VRAM budget
    if name in ("available_vram_bytes", "max_bands_for_budget"):
        from vibespatial.raster import dispatch

        return getattr(dispatch, name)
    # NVRTC cache management
    if name in ("clear_nvrtc_cache", "nvrtc_cache_stats"):
        from vibespatial.cuda_runtime import clear_nvrtc_cache, nvrtc_cache_stats

        _cache_api = {
            "clear_nvrtc_cache": clear_nvrtc_cache,
            "nvrtc_cache_stats": nvrtc_cache_stats,
        }
        return _cache_api[name]
    raise AttributeError(f"module 'vibespatial.raster' has no attribute {name!r}")
