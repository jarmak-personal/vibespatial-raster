# vibespatial-raster

```{raw} html
<div class="cp-hero">
  <div class="cp-hero-content">
    <h1 class="cp-hero-title" data-glitch="vibespatial-raster">vibespatial-<span class="accent">raster</span></h1>
    <p class="cp-hero-subtitle">
      GPU-first raster processing. Custom NVRTC kernels and CCCL primitives for
      algebra, zonal stats, labeling, rasterize, and polygonize — all with
      zero-copy GPU residency.
    </p>
    <div class="cp-hero-actions">
      <a class="cp-btn cp-btn--primary" href="user/index.html">User Guide &rarr;</a>
      <a class="cp-btn cp-btn--secondary" href="dev/index.html">Developer Guide &rarr;</a>
    </div>
  </div>
</div>
```

```{raw} html
<div class="cp-features">
  <div class="cp-card cp-reveal">
    <h3>GPU Algebra</h3>
    <p>Element-wise add, subtract, multiply, divide, apply, where, classify. CuPy-backed with automatic nodata propagation.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Focal Ops</h3>
    <p>Convolution, Gaussian filter, slope, aspect. Custom NVRTC shared-memory stencil kernels with tiled execution.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Zonal Statistics</h3>
    <p>Per-zone count, sum, mean, min, max, std, median via CCCL segmented reduce. Radix-sorted zone dispatch.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Connected Components</h3>
    <p>Union-find CCL via NVRTC kernels. 4 or 8 connectivity, iterative merge+jump convergence, sieve filtering.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>GPU-Native IO</h3>
    <p>nvImageCodec decodes GeoTIFF and JPEG2000 directly to GPU memory. Zero host-device copies. Rasterio fallback.</p>
  </div>
  <div class="cp-card cp-reveal">
    <h3>Polygonize</h3>
    <p>Marching-squares NVRTC kernels classify cells and emit directed edges. Ring chaining produces GeoDataFrame output.</p>
  </div>
</div>
```

## Quick Example

```python
from vibespatial.raster import read_raster, zonal_stats, raster_slope

raster = read_raster("elevation.tif")
slope = raster_slope(raster)
stats = zonal_stats(zones, slope, ["mean", "max", "std"])
```

```{toctree}
:hidden:
:maxdepth: 2

user/index
dev/index
```
