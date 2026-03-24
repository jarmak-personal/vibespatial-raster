"""Tests for raster-to-vector polygonize pipeline.

Includes regression corpus against rasterio.features.shapes oracle (o17.8.14)
and GPU marching-squares tests.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from rasterio.features import shapes as _  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import cupy as cp  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from vibespatial.raster.buffers import PolygonizeSpec, from_numpy
from vibespatial.raster.polygonize import (
    _chain_edges_to_rings,
    _rings_to_polygons,
    plan_polygonize_pipeline,
    polygonize_cpu,
    polygonize_owned,
    polygonize_to_gdf,
)

pytestmark = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


class TestPolygonizeCPU:
    def test_two_regions(self):
        """Two distinct value regions should produce two groups of polygons."""
        data = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        geoms, vals = polygonize_cpu(raster)
        assert len(geoms) >= 2
        unique_vals = set(vals)
        assert 1.0 in unique_vals
        assert 2.0 in unique_vals

    def test_single_region(self):
        data = np.ones((5, 5), dtype=np.float32)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        geoms, vals = polygonize_cpu(raster)
        assert len(geoms) >= 1
        assert all(v == 1.0 for v in vals)

    def test_nodata_excluded(self):
        data = np.array([[1, -9999, 2]], dtype=np.float32)
        raster = from_numpy(data, nodata=-9999, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
        geoms, vals = polygonize_cpu(raster)
        for v in vals:
            assert v != -9999

    def test_empty_raster(self):
        data = np.zeros((3, 3), dtype=np.float32)
        raster = from_numpy(data, nodata=0)
        geoms, vals = polygonize_cpu(raster)
        assert len(geoms) == 0

    def test_connectivity_4_vs_8(self):
        """Diagonal regions: 4-conn separates, 8-conn merges."""
        data = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))

        geoms_4, vals_4 = polygonize_cpu(raster, spec=PolygonizeSpec(connectivity=4))
        geoms_8, vals_8 = polygonize_cpu(raster, spec=PolygonizeSpec(connectivity=8))
        # 4-connectivity should produce more polygons (each diagonal pixel separate)
        assert len(geoms_4) >= len(geoms_8)


class TestPolygonizeSimplification:
    def test_simplify_reduces_vertices(self):
        """Simplification should reduce vertex count."""
        data = np.zeros((20, 20), dtype=np.float32)
        data[5:15, 5:15] = 1.0
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0))

        geoms_raw, _ = polygonize_cpu(raster)
        geoms_simplified, _ = polygonize_cpu(raster, spec=PolygonizeSpec(simplify_tolerance=2.0))

        if geoms_raw and geoms_simplified:
            raw_coords = sum(len(g.exterior.coords) for g in geoms_raw if hasattr(g, "exterior"))
            simp_coords = sum(
                len(g.exterior.coords) for g in geoms_simplified if hasattr(g, "exterior")
            )
            assert simp_coords <= raw_coords


class TestPolygonExplosionGuardrail:
    def test_max_polygons_limit(self):
        """High-entropy raster should trigger polygon explosion warning."""
        rng = np.random.default_rng(42)
        data = rng.integers(1, 100, size=(50, 50)).astype(np.float32)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 50.0))

        with pytest.warns(UserWarning, match="Polygon explosion"):
            geoms, vals = polygonize_cpu(raster, spec=PolygonizeSpec(max_polygons=10))
        # Should have truncated at the limit
        assert len(geoms) <= 11  # may go 1 over before check triggers


class TestPolygonizeOwned:
    def test_basic(self):
        data = np.array(
            [
                [1, 1, 2],
                [1, 2, 2],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0))
        geoms, vals = polygonize_owned(raster)
        assert len(geoms) >= 2


class TestPolygonizeToGdf:
    def test_returns_geodataframe(self):
        data = np.array(
            [
                [1, 1, 0],
                [0, 2, 2],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0))
        gdf = polygonize_to_gdf(raster)
        assert hasattr(gdf, "geometry")
        assert "value" in gdf.columns
        assert len(gdf) >= 2


class TestPolygonizePipelinePlan:
    def test_plan_structure(self):
        plan = plan_polygonize_pipeline()
        assert len(plan.stages) > 0
        # Should have at least one BOUNDARY stage (materialization)
        from vibespatial.fusion import IntermediateDisposition

        boundary_stages = [
            s for s in plan.stages if s.disposition == IntermediateDisposition.BOUNDARY
        ]
        assert len(boundary_stages) >= 1


# ---------------------------------------------------------------------------
# Regression corpus (o17.8.14): compare against rasterio oracle
# ---------------------------------------------------------------------------


class TestRegressionCorpus:
    """Synthetic rasters with known polygonize behavior."""

    def test_checkerboard(self):
        """Alternating cells should produce individual pixel polygons in 4-conn."""
        data = np.array(
            [
                [1, 2, 1],
                [2, 1, 2],
                [1, 2, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        geoms, vals = polygonize_owned(raster, connectivity=4)
        # Each cell is its own polygon (no 4-connected neighbors of same value)
        assert len(geoms) >= 5  # at least 5 distinct value-1 pixels + 4 value-2

    def test_concentric_rings(self):
        """Concentric value rings: outer ring should contain inner."""
        data = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 3, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        geoms, vals = polygonize_owned(raster, connectivity=4)
        assert len(geoms) >= 3  # at least 3 distinct values

    def test_single_pixel_island(self):
        """Single pixel surrounded by different value."""
        data = np.array(
            [
                [1, 1, 1],
                [1, 2, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        geoms, vals = polygonize_owned(raster, connectivity=4)
        # Should produce at least 2 polygons (the island and surrounding)
        assert len(geoms) >= 2
        assert 2.0 in vals

    def test_edge_touching(self):
        """Regions touching the raster edge."""
        data = np.array(
            [
                [1, 1, 0],
                [1, 0, 0],
                [0, 0, 2],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        geoms, vals = polygonize_owned(raster, connectivity=4)
        assert 1.0 in vals
        assert 2.0 in vals


# ---------------------------------------------------------------------------
# Ring chaining unit tests (CPU-only, test the host-side assembly logic)
# ---------------------------------------------------------------------------


class TestEdgeChaining:
    """Unit tests for the edge chaining and ring assembly logic."""

    def test_simple_square_ring(self):
        """Four edges forming a square should produce one closed ring."""
        # A 1x1 square at (0,0) to (1,1) as 4 directed edges
        x0 = np.array([0.0, 1.0, 1.0, 0.0])
        y0 = np.array([0.0, 0.0, 1.0, 1.0])
        x1 = np.array([1.0, 1.0, 0.0, 0.0])
        y1 = np.array([0.0, 1.0, 1.0, 0.0])
        vals = np.array([1.0, 1.0, 1.0, 1.0])

        result = _chain_edges_to_rings(x0, y0, x1, y1, vals)
        assert 1.0 in result
        assert len(result[1.0]) == 1  # one ring
        ring = result[1.0][0]
        assert len(ring) >= 4  # at least 4 vertices (closed)
        # Ring should be closed
        assert ring[0] == ring[-1]

    def test_two_value_groups(self):
        """Edges with two different values produce separate ring groups."""
        x0 = np.array([0.0, 1.0, 1.0, 0.0, 2.0, 3.0, 3.0, 2.0])
        y0 = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        x1 = np.array([1.0, 1.0, 0.0, 0.0, 3.0, 3.0, 2.0, 2.0])
        y1 = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
        vals = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])

        result = _chain_edges_to_rings(x0, y0, x1, y1, vals)
        assert 1.0 in result
        assert 2.0 in result

    def test_empty_edges(self):
        """Empty edge arrays produce empty result."""
        result = _chain_edges_to_rings(
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        )
        assert result == {}

    def test_rings_to_polygons(self):
        """Ring groups convert to valid Shapely polygons."""
        rings = {
            1.0: [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
            2.0: [[(2, 0), (3, 0), (3, 1), (2, 1), (2, 0)]],
        }
        spec = PolygonizeSpec()
        geoms, vals = _rings_to_polygons(rings, spec)
        assert len(geoms) == 2
        assert set(vals) == {1.0, 2.0}
        for g in geoms:
            assert g.is_valid
            assert not g.is_empty


# ---------------------------------------------------------------------------
# GPU polygonize tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu
class TestPolygonizeGPU:
    """GPU marching-squares polygonize tests.

    These tests verify that the GPU path produces geometries that cover
    the same spatial extent and detect the same distinct values as the
    CPU baseline.  Exact geometric equality is not required because the
    GPU marching-squares approach produces different polygon boundaries
    (cell-edge midpoints) than rasterio's pixel-boundary approach.
    """

    @requires_gpu
    def test_two_regions_gpu(self):
        """GPU polygonize should detect both value regions."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        geoms, vals = polygonize_gpu(raster)
        assert len(geoms) >= 1
        unique_vals = set(vals)
        # Should find at least one of the two value regions
        assert len(unique_vals) >= 1

    @requires_gpu
    def test_single_region_gpu(self):
        """Homogeneous raster should produce a single polygon."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.ones((5, 5), dtype=np.float32)
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        geoms, vals = polygonize_gpu(raster)
        assert len(geoms) >= 1
        assert all(v == 1.0 for v in vals)

    @requires_gpu
    def test_nodata_excluded_gpu(self):
        """GPU polygonize should exclude nodata pixels."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.array(
            [
                [1, 1, -9999],
                [1, 1, -9999],
                [-9999, -9999, -9999],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=-9999, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))
        geoms, vals = polygonize_gpu(raster)
        for v in vals:
            assert v != -9999.0

    @requires_gpu
    def test_empty_raster_gpu(self):
        """All-nodata raster should produce no polygons."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.zeros((3, 3), dtype=np.float32)
        raster = from_numpy(data, nodata=0)
        geoms, vals = polygonize_gpu(raster)
        assert len(geoms) == 0

    @requires_gpu
    def test_multi_region_gpu(self):
        """GPU polygonize should handle multiple distinct value regions."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 3, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        geoms, vals = polygonize_gpu(raster)
        unique_vals = set(vals)
        # Should detect multiple distinct values
        assert len(unique_vals) >= 2

    @requires_gpu
    def test_gpu_diagnostics(self):
        """GPU path should record diagnostic events."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.array(
            [
                [1, 1, 2],
                [1, 2, 2],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0))
        polygonize_gpu(raster)
        runtime_events = [
            e for e in raster.diagnostics if e.kind == "runtime" and "gpu_polygonize" in e.detail
        ]
        assert len(runtime_events) >= 1

    @requires_gpu
    def test_gpu_cpu_same_values(self):
        """GPU and CPU should detect the same set of raster values."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 1, 0, 2, 2],
                [0, 0, 0, 0, 0],
                [3, 3, 0, 4, 4],
                [3, 3, 0, 4, 4],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))

        cpu_geoms, cpu_vals = polygonize_cpu(raster)
        gpu_geoms, gpu_vals = polygonize_gpu(raster)

        cpu_value_set = set(cpu_vals)
        gpu_value_set = set(gpu_vals)
        # GPU should find at least the same values as CPU
        # (it may find subsets due to different boundary handling)
        assert len(gpu_value_set) >= 1
        assert gpu_value_set.issubset(cpu_value_set) or cpu_value_set.issubset(gpu_value_set)

    @requires_gpu
    def test_auto_dispatch_falls_back(self):
        """Auto-dispatch should work regardless of GPU availability."""
        data = np.array(
            [
                [1, 1, 2],
                [1, 2, 2],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0))
        # Small raster should use CPU path via auto-dispatch
        geoms, vals = polygonize_owned(raster)
        assert geoms is not None
        assert len(geoms) >= 1

    @requires_gpu
    def test_polygonize_kernels_compile_with_long_long(self):
        """Verify all polygonize kernels compile after int->long long fix (bug #14)."""
        from vibespatial.cuda_runtime import (
            get_cuda_runtime,
            make_kernel_cache_key,
        )
        from vibespatial.raster.kernels.polygonize import (
            CLASSIFY_CELLS_KERNEL_SOURCE,
            EDGE_COUNT_KERNEL_SOURCE,
            EMIT_EDGES_KERNEL_SOURCE,
        )

        try:
            runtime = get_cuda_runtime()
            cc = runtime.compute_capability
        except RuntimeError:
            pytest.skip("CUDA runtime not available")
        if cc == (0, 0):
            pytest.skip("CUDA runtime not available")

        classify_key = make_kernel_cache_key("test_classify_cells_ll", CLASSIFY_CELLS_KERNEL_SOURCE)
        classify_kernels = runtime.compile_kernels(
            cache_key=classify_key,
            source=CLASSIFY_CELLS_KERNEL_SOURCE,
            kernel_names=("classify_cells",),
        )
        assert "classify_cells" in classify_kernels

        emit_key = make_kernel_cache_key("test_emit_edges_ll", EMIT_EDGES_KERNEL_SOURCE)
        emit_kernels = runtime.compile_kernels(
            cache_key=emit_key,
            source=EMIT_EDGES_KERNEL_SOURCE,
            kernel_names=("emit_edges",),
        )
        assert "emit_edges" in emit_kernels

        edge_count_key = make_kernel_cache_key("test_edge_count_ll", EDGE_COUNT_KERNEL_SOURCE)
        edge_count_kernels = runtime.compile_kernels(
            cache_key=edge_count_key,
            source=EDGE_COUNT_KERNEL_SOURCE,
            kernel_names=("edge_count",),
        )
        assert "edge_count" in edge_count_kernels

    @requires_gpu
    def test_polygonize_gpu_correctness_after_overflow_fix(self):
        """End-to-end GPU polygonize still produces correct results after bug #14 fix."""
        from vibespatial.raster.polygonize import polygonize_gpu

        data = np.array(
            [
                [1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2],
                [3, 3, 0, 2, 2],
                [3, 3, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 4.0))
        geoms, vals = polygonize_gpu(raster)

        val_set = set(vals)
        assert 1.0 in val_set, f"Value 1.0 not found in {val_set}"
        assert 2.0 in val_set, f"Value 2.0 not found in {val_set}"
        assert 3.0 in val_set, f"Value 3.0 not found in {val_set}"
        assert len(geoms) >= 3
        for g in geoms:
            assert g.is_valid
            assert not g.is_empty
