---
name: raster-domain
description: "Use this skill to verify raster processing correctness, check nodata semantics, validate affine transforms, understand neighborhood operations, or answer domain questions about raster algorithms. This is the domain knowledge oracle for autonomous agents working on vibespatial-raster. Trigger on: \"is this correct\", \"should this return\", \"nodata\", \"affine\", \"CRS\", \"neighborhood\", \"stencil\", \"focal\", \"morphology\", \"CCL\", \"connected component\", \"zonal\", \"rasterize\", \"polygonize\", \"marching squares\"."
user-invocable: true
argument-hint: <operation or question to verify>
---

# Raster Domain Knowledge Oracle — vibespatial-raster

Use this reference to verify whether a raster operation produces the
correct result, to understand edge cases, or to validate design decisions.

Question: **$ARGUMENTS**

---

## 1. OwnedRasterArray Contract

### Core Fields
- `data`: numpy or CuPy array of shape `(height, width)` or `(bands, height, width)`
- `nodata`: scalar sentinel value (same dtype as data) or None
- `affine`: 6-parameter affine transform (a, b, c, d, e, f) mapping pixel -> world
- `crs`: coordinate reference system (pyproj.CRS or None)
- `residency`: HOST or DEVICE
- `diagnostics`: list of RasterDiagnosticEvent

### Residency Rules
- HOST: data is a numpy array, accessed via `.data`
- DEVICE: data is a CuPy array, accessed via `.device_data()`
- Use `move_to(Residency.DEVICE, ...)` for H->D transfer
- Use `cp.asnumpy()` for final D->H at pipeline end
- NEVER access `.data` on a DEVICE-resident raster (will raise or give stale data)

### Dtype Preservation
- Rasters store native dtype: uint8, int16, int32, float32, float64
- Operations should preserve input dtype when possible
- Exceptions: operations that produce a different type (e.g., slope -> float64)
- NEVER upcast uint8 -> float64 unless the computation requires it

---

## 2. Nodata Handling

### Semantics
- Nodata pixels represent "no information" — they are NOT zero or NaN
- Nodata is a specific sentinel value in the raster's native dtype
- Operations must propagate nodata: if any input pixel in an operation
  is nodata, the output pixel should be nodata

### Nodata in Neighborhood Operations (Focal/Stencil)
- If ANY neighbor is nodata, the center pixel result can be:
  - nodata (strict propagation — default for most operations)
  - computed using only valid neighbors (optional, for statistics)
- Edge pixels (outside raster bounds) are treated as nodata or clamped

### Nodata in Algebra Operations
- `add(a, b)`: if either pixel is nodata -> result is nodata
- `where(condition, a, b)`: condition can reference nodata pixels
- `classify(raster, breaks)`: nodata pixels remain nodata in output

### Nodata Mask Pattern
```python
# On GPU: nullable mask pointer
d_mask = raster.device_nodata_mask()  # CuPy bool array or None
# In kernel: if (nodata_mask != nullptr && nodata_mask[idx]) { output[idx] = nodata_val; return; }
```

### Common Pitfalls
- Integer nodata (e.g., 255 for uint8) is a valid pixel value in other contexts
- Float nodata should be checked with exact equality, not tolerance
- NaN is a special case: for float rasters, NaN may be used as nodata
- When NaN is nodata, use `isnan()` in kernels, not `==` comparison

---

## 3. Affine Transforms

### Definition
The affine transform maps pixel coordinates to world coordinates:
```
world_x = a * col + b * row + c
world_y = d * col + e * row + f
```
Where:
- `a` = pixel width (x-resolution)
- `b` = rotation (usually 0)
- `c` = x-coordinate of upper-left corner
- `d` = rotation (usually 0)
- `e` = pixel height (negative for north-up, i.e., y decreases downward)
- `f` = y-coordinate of upper-left corner

### Pixel Centers vs Corners
- The affine maps to the **upper-left corner** of the pixel
- Pixel center is at `(col + 0.5, row + 0.5)` in pixel space
- This matters for rasterize (point-in-pixel test) and polygonize (contour placement)

### Affine Preservation
- Most operations preserve the input affine (same grid)
- Resampling/reprojection changes the affine
- Subsetting changes `c` and `f` (origin shifts)
- Resolution changes affect `a` and `e`

---

## 4. Focal / Neighborhood Operations

### Convolution
- Applies a kernel (weight matrix) to each pixel's neighborhood
- Output[i,j] = sum(kernel * neighborhood around i,j)
- Boundary handling: zero-pad (default), reflect, or clamp

### Gaussian Blur
- Convolution with a Gaussian kernel
- Kernel size determined by sigma: `size = ceil(6 * sigma) | 1` (ensure odd)
- Separable: apply 1D horizontal then 1D vertical for efficiency

### Slope and Aspect
- Computed from elevation rasters (DEM)
- Use Horn's method (3x3 neighborhood):
  ```
  dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cell_size_x)
  dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cell_size_y)
  slope = arctan(sqrt(dz_dx^2 + dz_dy^2))
  aspect = arctan2(-dz_dy, dz_dx)
  ```
- Cell size comes from the affine transform (`abs(a)` for x, `abs(e)` for y)

### Shared Memory Tiling
- Load tile + halo into shared memory
- Halo width = kernel_radius (e.g., 1 for 3x3, 2 for 5x5)
- Tile dims: `(TILE_H + 2*halo, TILE_W + 2*halo)`
- `__syncthreads()` after loading, before computation

---

## 5. Connected Component Labeling (CCL)

### Algorithm: GPU Union-Find
1. **Initialize**: each foreground pixel gets its own label (= pixel index)
2. **Local merge**: check 4-connected neighbors, atomicMin to merge labels
3. **Pointer jumping**: path compression until convergence
4. **Relabel**: (optional) compact labels to sequential 1..N

### Convergence
- Iterate merge + pointer-jump until no pixel changes label
- Check convergence via device-side `changed` flag (atomicOr)
- Do NOT check convergence by transferring labels to host each iteration

### Connectivity
- 4-connected: up, down, left, right (default)
- 8-connected: adds diagonals (use_8_connectivity parameter)
- Background pixels (value == 0 or == nodata) are excluded

### Edge Cases
- All-background raster -> all labels = 0 (no components)
- All-foreground raster -> one component (label = 0)
- Single pixel component -> valid, label = pixel_index
- Very large components can take more iterations to converge

---

## 6. Morphology Operations

### Binary Erosion
- Output pixel = 1 only if ALL structuring element pixels are foreground
- Shrinks foreground regions, removes small features
- Structuring element typically 3x3 cross or 3x3 square

### Binary Dilation
- Output pixel = 1 if ANY structuring element pixel is foreground
- Grows foreground regions, fills small gaps
- Dilation of A by B = complement of erosion of complement(A) by B

### Opening (Erosion then Dilation)
- Removes small foreground features while preserving shape
- `open(A) = dilate(erode(A))`

### Closing (Dilation then Erosion)
- Removes small holes while preserving shape
- `close(A) = erode(dilate(A))`

### Sieve (Remove Small Components)
- Run CCL, count component sizes, remove components below threshold
- `sieve(raster, min_size=10)` removes components with < 10 pixels

### GPU Implementation
- Erosion/dilation are stencil operations -> shared memory tiling
- Opening/closing = two sequential stencil launches
- No sync needed between launches on same stream

---

## 7. Zonal Statistics

### Concept
- Given a raster and a zone raster (integer labels), compute statistics
  per zone: mean, sum, min, max, count, std
- Zones are non-overlapping labeled regions

### GPU Approach
1. Sort pixels by zone ID (CCCL radix_sort)
2. Build segment offsets (unique_by_key or histogram)
3. Segmented reduce per statistic (CCCL segmented_reduce)

### Edge Cases
- Zone ID = nodata -> exclude from statistics
- Value = nodata -> exclude from statistics
- Empty zone (no valid pixels) -> statistic = NaN or 0
- Very large zones (>1M pixels) -> still efficient with CCCL

---

## 8. Rasterize (Vector to Raster)

### Algorithm: Point-in-Polygon per Pixel
- For each pixel center, test if it falls inside any input polygon
- Assign the polygon's value to the pixel (or burn a constant)
- GPU: parallelize over pixels, each thread tests against polygon

### Pixel Center Convention
- Test point = pixel center = `(col + 0.5, row + 0.5)` transformed by affine
- A pixel on the boundary of a polygon should be included (follows rasterio convention)

### Edge Cases
- Overlapping polygons: last polygon wins (or configurable merge)
- Polygon with holes: winding number test correctly handles holes
- Multipolygon: treat each part independently
- Empty polygon set: all pixels get nodata

---

## 9. Polygonize (Raster to Vector)

### Algorithm: Marching Squares
- Classify each 2x2 pixel cell into one of 16 cases based on threshold
- Emit line segments for contour boundaries
- Connect segments into polygon rings
- Assign interior value to each polygon

### Cell Classification
- 4 corners of 2x2 cell, each above or below threshold
- 4 bits -> 16 cases (including ambiguous saddle points)
- Saddle resolution: use average of 4 corners to disambiguate

### Output
- List of (geometry, value) pairs
- Polygons follow the OGC right-hand rule (exterior CCW, holes CW)
- Coordinates in world space (apply affine transform)

### Edge Cases
- Uniform raster (all same value): one polygon covering entire extent
- Nodata pixels: treated as "below threshold" or excluded from polygons
- Sub-pixel resolution: vertices at cell edges (col, row) in pixel space

---

## 10. Local Algebra Operations

### Element-wise Operations
- `add(a, b)`, `subtract(a, b)`, `multiply(a, b)`, `divide(a, b)`
- `apply(raster, func)` — apply a Python function element-wise
- `where(condition, true_val, false_val)` — conditional selection
- `classify(raster, breaks, labels)` — reclassify by value ranges

### Rules
- Both inputs must have same shape (height, width)
- Both inputs should have same affine and CRS (warn if different)
- Nodata propagation: if either input is nodata -> output is nodata
- Division by zero -> nodata (not inf or NaN)
- Output dtype: result of numpy type promotion rules

### GPU Path
- Element-wise ops are CuPy-native (cp.add, cp.where, etc.)
- No custom NVRTC needed
- Zero-copy: operate directly on device arrays
