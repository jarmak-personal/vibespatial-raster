"""Tests for vector-to-raster rasterize operations."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from vibespatial.raster.buffers import GridSpec, from_numpy
from vibespatial.raster.rasterize import rasterize_cpu, rasterize_gpu, rasterize_owned

try:
    from rasterio.features import rasterize as _  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False


@pytest.fixture
def simple_grid():
    return GridSpec.from_bounds(0.0, 0.0, 10.0, 10.0, resolution=1.0)


@pytest.fixture
def two_boxes():
    geoms = [box(1, 1, 4, 4), box(6, 6, 9, 9)]
    values = np.array([1.0, 2.0])
    return geoms, values


class TestGridSpec:
    def test_from_bounds(self):
        gs = GridSpec.from_bounds(0.0, 0.0, 100.0, 50.0, resolution=1.0)
        assert gs.width == 100
        assert gs.height == 50

    def test_from_raster(self):
        raster = from_numpy(np.zeros((10, 20)))
        gs = GridSpec.from_raster(raster)
        assert gs.width == 20
        assert gs.height == 10


@pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
class TestRasterizeCPU:
    def test_basic(self, simple_grid, two_boxes):
        geoms, values = two_boxes
        result = rasterize_cpu(geoms, values, simple_grid)
        data = result.to_numpy()
        assert data.shape == (10, 10)
        # Check that the boxes are filled
        assert data[6, 2] == 1.0  # inside first box (row=10-4=6, col=2)
        assert data[1, 7] == 2.0  # inside second box (row=10-9=1, col=7)

    def test_fill_value(self, simple_grid):
        gs = GridSpec(
            affine=simple_grid.affine,
            width=simple_grid.width,
            height=simple_grid.height,
            fill_value=-999.0,
        )
        result = rasterize_cpu([box(0, 0, 1, 1)], np.array([5.0]), gs)
        data = result.to_numpy()
        # Most pixels should be fill value
        assert (data == -999.0).sum() > 50

    def test_empty_geometries(self, simple_grid):
        result = rasterize_cpu([], np.array([]), simple_grid)
        data = result.to_numpy()
        assert (data == simple_grid.fill_value).all()


@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterizeGPU:
    def test_basic(self, simple_grid, two_boxes):
        geoms, values = two_boxes
        result = rasterize_gpu(geoms, values, simple_grid)
        data = result.to_numpy()
        assert data.shape == (10, 10)

    def test_single_polygon(self):
        gs = GridSpec.from_bounds(0.0, 0.0, 4.0, 4.0, resolution=1.0)
        poly = box(0.5, 0.5, 3.5, 3.5)
        result = rasterize_gpu([poly], np.array([42.0]), gs)
        data = result.to_numpy()
        # Center pixels should be filled
        assert data[1, 1] == 42.0
        assert data[2, 2] == 42.0

    def test_multipolygon(self):
        gs = GridSpec.from_bounds(0.0, 0.0, 10.0, 10.0, resolution=1.0)
        mp = MultiPolygon([box(1, 1, 3, 3), box(7, 7, 9, 9)])
        result = rasterize_gpu([mp], np.array([5.0]), gs)
        data = result.to_numpy()
        assert data.shape == (10, 10)

    def test_polygon_with_hole(self):
        gs = GridSpec.from_bounds(0.0, 0.0, 10.0, 10.0, resolution=1.0)
        outer = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(4, 4), (6, 4), (6, 6), (4, 6), (4, 4)]
        poly = Polygon(outer, [hole])
        result = rasterize_gpu([poly], np.array([1.0]), gs)
        data = result.to_numpy()
        # Pixel inside hole should not be filled
        assert data[5, 5] == 0.0  # center of hole
        # Pixel outside hole should be filled
        assert data[1, 1] == 1.0

    def test_empty_input(self):
        gs = GridSpec.from_bounds(0.0, 0.0, 5.0, 5.0, resolution=1.0)
        result = rasterize_gpu([], np.array([]), gs)
        assert (result.to_numpy() == gs.fill_value).all()

    @pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
    def test_gpu_cpu_agreement(self, simple_grid, two_boxes):
        """GPU and CPU rasterize should produce similar results."""
        geoms, values = two_boxes
        cpu_result = rasterize_cpu(geoms, values, simple_grid)
        gpu_result = rasterize_gpu(geoms, values, simple_grid)
        # Allow some boundary pixel differences due to center vs edge sampling
        cpu_data = cpu_result.to_numpy()
        gpu_data = gpu_result.to_numpy()
        agreement = (cpu_data == gpu_data).mean()
        assert agreement > 0.85, f"Agreement only {agreement:.1%}"


class TestRasterizeOwned:
    @pytest.mark.skipif(not HAS_GPU and not HAS_RASTERIO, reason="need GPU or rasterio")
    def test_dispatch(self, simple_grid, two_boxes):
        geoms, values = two_boxes
        result = rasterize_owned(geoms, values, simple_grid)
        assert result.to_numpy().shape == (10, 10)
