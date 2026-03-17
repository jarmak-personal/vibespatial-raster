# Changelog

## 0.1.0 — Unreleased

Initial release.

- OwnedRasterArray with HOST/DEVICE residency and diagnostic events
- GeoTIFF/COG IO via rasterio (HYBRID path)
- GPU-native raster decode via nvImageCodec (GeoTIFF + JPEG2000)
- Raster algebra: local ops (add, subtract, multiply, divide, apply, where, classify)
- Focal ops: convolution, Gaussian filter, slope, aspect (NVRTC kernels)
- Zonal statistics via CCCL segmented reduce (count, sum, mean, min, max, std, median)
- Vector-to-raster rasterize with NVRTC per-pixel point-in-polygon kernel
- Connected component labeling via NVRTC union-find (4/8 connectivity)
- Morphology operations: erode, dilate, sieve filter (NVRTC kernels)
- Raster-to-vector polygonize via marching-squares NVRTC kernels
- GeoDataFrame integration for zonal stats and polygonize output
- CPU fallback for all operations (scipy, rasterio)
