"""Tests for terrain derivative operations: TRI, TPI, and curvature."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import from_numpy

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_dem():
    """10x10 flat surface at elevation 100."""
    data = np.ones((10, 10), dtype=np.float64) * 100.0
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))


@pytest.fixture
def tilted_dem():
    """10x10 linear ramp increasing east (columns). Elevation = column index."""
    data = np.tile(np.arange(10, dtype=np.float64), (10, 1))
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))


@pytest.fixture
def peak_dem():
    """11x11 DEM with a peak at center (5,5). Radially decreasing elevation."""
    y, x = np.mgrid[0:11, 0:11]
    data = 100.0 - np.sqrt((x - 5.0) ** 2 + (y - 5.0) ** 2) * 5.0
    return from_numpy(data.astype(np.float64), affine=(1.0, 0.0, 0.0, 0.0, -1.0, 11.0))


@pytest.fixture
def nodata_dem():
    """10x10 DEM with nodata in center."""
    data = np.ones((10, 10), dtype=np.float64) * 50.0
    data[5, 5] = -9999.0
    return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))


# ---------------------------------------------------------------------------
# CPU tests for TRI
# ---------------------------------------------------------------------------


class TestTRI_CPU:
    def test_flat_surface(self, flat_dem):
        """TRI of a flat surface should be 0 for interior pixels."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(flat_dem, use_gpu=False)
        data = result.to_numpy()
        # Interior pixels (excluding 1-pixel border which is nodata)
        interior = data[2:-2, 2:-2]
        np.testing.assert_array_almost_equal(interior, 0.0)

    def test_tilted_surface(self, tilted_dem):
        """TRI of a tilted surface should be positive for interior pixels."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(tilted_dem, use_gpu=False)
        data = result.to_numpy()
        # Interior pixels should have positive TRI (neighbors differ from center)
        assert data[5, 5] > 0.0

    def test_peak_surface(self, peak_dem):
        """TRI around a peak should be positive."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(peak_dem, use_gpu=False)
        data = result.to_numpy()
        # Around the peak, neighbors differ from center
        assert data[5, 5] > 0.0

    def test_nodata_propagation(self, nodata_dem):
        """Nodata should propagate to neighbors."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(nodata_dem, use_gpu=False)
        data = result.to_numpy()
        # The nodata pixel itself should be nodata
        assert data[5, 5] == -9999.0
        # Adjacent pixels should also be nodata (their 3x3 window touches nodata)
        assert data[4, 5] == -9999.0
        assert data[5, 4] == -9999.0
        assert data[6, 5] == -9999.0
        assert data[5, 6] == -9999.0

    def test_border_is_nodata(self, flat_dem):
        """Border pixels should be nodata (no full 3x3 window)."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(flat_dem, use_gpu=False)
        data = result.to_numpy()
        nodata_val = -9999.0  # default when dem.nodata is None
        # Top row, bottom row, left col, right col
        assert data[0, 5] == nodata_val
        assert data[9, 5] == nodata_val
        assert data[5, 0] == nodata_val
        assert data[5, 9] == nodata_val

    def test_tri_manual_computation(self):
        """Verify TRI against manual calculation for a known 3x3 window."""
        from vibespatial.raster.algebra import raster_tri

        # Create a 5x5 DEM where center 3x3 is known
        data = np.array(
            [
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
            ],
            dtype=np.float64,
        )
        dem = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        result = raster_tri(dem, use_gpu=False)
        out = result.to_numpy()

        # Center pixel (2,2) has value 5.
        # Neighbors: 3, 4, 5, 4, 6, 5, 6, 7
        # Abs diffs: 2, 1, 0, 1, 1, 0, 1, 2
        # Mean = (2+1+0+1+1+0+1+2)/8 = 8/8 = 1.0
        np.testing.assert_almost_equal(out[2, 2], 1.0)


# ---------------------------------------------------------------------------
# CPU tests for TPI
# ---------------------------------------------------------------------------


class TestTPI_CPU:
    def test_flat_surface(self, flat_dem):
        """TPI of a flat surface should be 0 for interior pixels."""
        from vibespatial.raster.algebra import raster_tpi

        result = raster_tpi(flat_dem, use_gpu=False)
        data = result.to_numpy()
        interior = data[2:-2, 2:-2]
        np.testing.assert_array_almost_equal(interior, 0.0)

    def test_peak_positive(self, peak_dem):
        """TPI at a peak should be positive (center higher than mean)."""
        from vibespatial.raster.algebra import raster_tpi

        result = raster_tpi(peak_dem, use_gpu=False)
        data = result.to_numpy()
        # Peak at (5,5) is higher than its neighbors
        assert data[5, 5] > 0.0

    def test_valley_negative(self):
        """TPI at a valley should be negative (center lower than mean)."""
        from vibespatial.raster.algebra import raster_tpi

        # Create a valley: center is low, neighbors are high
        y, x = np.mgrid[0:11, 0:11]
        data = np.sqrt((x - 5.0) ** 2 + (y - 5.0) ** 2) * 5.0
        dem = from_numpy(data.astype(np.float64), affine=(1.0, 0.0, 0.0, 0.0, -1.0, 11.0))
        result = raster_tpi(dem, use_gpu=False)
        out = result.to_numpy()
        assert out[5, 5] < 0.0

    def test_nodata_propagation(self, nodata_dem):
        """Nodata should propagate to TPI output."""
        from vibespatial.raster.algebra import raster_tpi

        result = raster_tpi(nodata_dem, use_gpu=False)
        data = result.to_numpy()
        assert data[5, 5] == -9999.0
        assert data[4, 5] == -9999.0

    def test_tpi_manual_computation(self):
        """Verify TPI against manual calculation."""
        from vibespatial.raster.algebra import raster_tpi

        data = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 10, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.float64,
        )
        dem = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        result = raster_tpi(dem, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2) = 10, all 8 neighbors = 1
        # TPI = 10 - mean(1,1,1,1,1,1,1,1) = 10 - 1 = 9
        np.testing.assert_almost_equal(out[2, 2], 9.0)


# ---------------------------------------------------------------------------
# CPU tests for curvature
# ---------------------------------------------------------------------------


class TestCurvature_CPU:
    def test_flat_surface(self, flat_dem):
        """Curvature of a flat surface should be 0 for interior pixels."""
        from vibespatial.raster.algebra import raster_curvature

        result = raster_curvature(flat_dem, use_gpu=False)
        data = result.to_numpy()
        interior = data[2:-2, 2:-2]
        np.testing.assert_array_almost_equal(interior, 0.0)

    def test_linear_ramp(self, tilted_dem):
        """Curvature of a linear ramp should be 0 (no second derivative)."""
        from vibespatial.raster.algebra import raster_curvature

        result = raster_curvature(tilted_dem, use_gpu=False)
        data = result.to_numpy()
        # Interior pixels should have ~0 curvature for a linear surface
        interior = data[2:-2, 2:-2]
        np.testing.assert_array_almost_equal(interior, 0.0, decimal=10)

    def test_concave_surface(self):
        """Concave (bowl) surface should have positive curvature."""
        from vibespatial.raster.algebra import raster_curvature

        # Bowl: elevation = x^2 + y^2 (concave up)
        y, x = np.mgrid[0:11, 0:11]
        data = ((x - 5.0) ** 2 + (y - 5.0) ** 2).astype(np.float64)
        dem = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 11.0))
        result = raster_curvature(dem, use_gpu=False)
        out = result.to_numpy()
        # Curvature should be positive (concave) at interior points
        assert out[5, 5] < 0.0  # -2*(D+E)*100, D and E are both positive -> negative

    def test_nodata_propagation(self, nodata_dem):
        """Nodata should propagate to curvature output."""
        from vibespatial.raster.algebra import raster_curvature

        result = raster_curvature(nodata_dem, use_gpu=False)
        data = result.to_numpy()
        assert data[5, 5] == -9999.0

    def test_curvature_manual(self):
        """Verify curvature against manual Zevenbergen-Thorne calculation."""
        from vibespatial.raster.algebra import raster_curvature

        # 5x5 with known center 3x3 window
        data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 4, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        dem = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        result = raster_curvature(dem, use_gpu=False)
        out = result.to_numpy()

        # Center (2,2) = 4, z3=1, z5=1, z1=1, z7=1
        # D = ((z3+z5)/2 - z4) / (cx^2) = ((1+1)/2 - 4) / 1 = -3
        # E = ((z1+z7)/2 - z4) / (cy^2) = ((1+1)/2 - 4) / 1 = -3
        # curvature = -2*(D+E)*100 = -2*(-6)*100 = 1200
        np.testing.assert_almost_equal(out[2, 2], 1200.0)


# ---------------------------------------------------------------------------
# Border nodata metadata tests (Bug #5)
# ---------------------------------------------------------------------------


class TestBorderNodataMetadata:
    """Verify that terrain derivative output declares nodata correctly.

    When the input DEM has nodata=None, border pixels are filled with the
    sentinel -9999.0. The output must declare nodata=-9999.0 so downstream
    consumers can identify border artifacts. When the input DEM has an
    explicit nodata value, that value must be propagated to the output.
    """

    def test_tri_nodata_declared_when_input_has_no_nodata(self, flat_dem):
        """TRI output must declare nodata=-9999.0 for border sentinel."""
        from vibespatial.raster.algebra import raster_tri

        assert flat_dem.nodata is None  # precondition
        result = raster_tri(flat_dem, use_gpu=False)
        assert result.nodata == -9999.0
        # Border pixels match the declared nodata
        data = result.to_numpy()
        assert data[0, 0] == result.nodata
        assert data[0, 5] == result.nodata
        assert data[9, 9] == result.nodata

    def test_tpi_nodata_declared_when_input_has_no_nodata(self, flat_dem):
        """TPI output must declare nodata=-9999.0 for border sentinel."""
        from vibespatial.raster.algebra import raster_tpi

        assert flat_dem.nodata is None
        result = raster_tpi(flat_dem, use_gpu=False)
        assert result.nodata == -9999.0
        data = result.to_numpy()
        assert data[0, 0] == result.nodata

    def test_curvature_nodata_declared_when_input_has_no_nodata(self, flat_dem):
        """Curvature output must declare nodata=-9999.0 for border sentinel."""
        from vibespatial.raster.algebra import raster_curvature

        assert flat_dem.nodata is None
        result = raster_curvature(flat_dem, use_gpu=False)
        assert result.nodata == -9999.0
        data = result.to_numpy()
        assert data[0, 0] == result.nodata

    def test_tri_preserves_explicit_nodata(self, nodata_dem):
        """When input DEM has explicit nodata, output preserves it."""
        from vibespatial.raster.algebra import raster_tri

        assert nodata_dem.nodata == -9999.0  # precondition
        result = raster_tri(nodata_dem, use_gpu=False)
        assert result.nodata == -9999.0

    def test_border_nodata_mask_consistency(self, flat_dem):
        """All border pixels must be identifiable via nodata_mask."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(flat_dem, use_gpu=False)
        data = result.to_numpy()
        nodata_val = result.nodata
        assert nodata_val is not None

        # Entire border ring should be nodata
        h, w = data.shape
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True  # top row
        border_mask[-1, :] = True  # bottom row
        border_mask[:, 0] = True  # left col
        border_mask[:, -1] = True  # right col

        np.testing.assert_array_equal(data[border_mask], nodata_val)
        # Interior should NOT be nodata (flat surface -> all valid interior)
        interior = data[1:-1, 1:-1]
        assert not np.any(interior == nodata_val)

    def test_metadata_preservation(self, flat_dem):
        """Affine and CRS must be preserved through terrain derivatives."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(flat_dem, use_gpu=False)
        assert result.affine == flat_dem.affine
        assert result.crs == flat_dem.crs


# ---------------------------------------------------------------------------
# Auto-dispatch tests (work with or without GPU)
# ---------------------------------------------------------------------------


class TestAutoDispatch:
    def test_tri_auto(self, flat_dem):
        """raster_tri auto-dispatches without error."""
        from vibespatial.raster.algebra import raster_tri

        result = raster_tri(flat_dem)
        assert result is not None
        assert result.shape == flat_dem.shape

    def test_tpi_auto(self, flat_dem):
        """raster_tpi auto-dispatches without error."""
        from vibespatial.raster.algebra import raster_tpi

        result = raster_tpi(flat_dem)
        assert result is not None

    def test_curvature_auto(self, flat_dem):
        """raster_curvature auto-dispatches without error."""
        from vibespatial.raster.algebra import raster_curvature

        result = raster_curvature(flat_dem)
        assert result is not None


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestTRI_GPU:
    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_flat_surface(self, flat_dem):
        """GPU TRI of flat surface matches CPU."""
        from vibespatial.raster.algebra import raster_tri

        cpu = raster_tri(flat_dem, use_gpu=False)
        gpu = raster_tri(flat_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_tilted_surface(self, tilted_dem):
        """GPU TRI of tilted surface matches CPU."""
        from vibespatial.raster.algebra import raster_tri

        cpu = raster_tri(tilted_dem, use_gpu=False)
        gpu = raster_tri(tilted_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_nodata_propagation(self, nodata_dem):
        """GPU TRI nodata propagation matches CPU."""
        from vibespatial.raster.algebra import raster_tri

        cpu = raster_tri(nodata_dem, use_gpu=False)
        gpu = raster_tri(nodata_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_peak_surface(self, peak_dem):
        """GPU TRI of peak surface matches CPU."""
        from vibespatial.raster.algebra import raster_tri

        cpu = raster_tri(peak_dem, use_gpu=False)
        gpu = raster_tri(peak_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())


@requires_gpu
class TestTPI_GPU:
    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_flat_surface(self, flat_dem):
        """GPU TPI of flat surface matches CPU."""
        from vibespatial.raster.algebra import raster_tpi

        cpu = raster_tpi(flat_dem, use_gpu=False)
        gpu = raster_tpi(flat_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_peak_positive(self, peak_dem):
        """GPU TPI of peak matches CPU."""
        from vibespatial.raster.algebra import raster_tpi

        cpu = raster_tpi(peak_dem, use_gpu=False)
        gpu = raster_tpi(peak_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_nodata_propagation(self, nodata_dem):
        """GPU TPI nodata propagation matches CPU."""
        from vibespatial.raster.algebra import raster_tpi

        cpu = raster_tpi(nodata_dem, use_gpu=False)
        gpu = raster_tpi(nodata_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())


@requires_gpu
class TestCurvature_GPU:
    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_flat_surface(self, flat_dem):
        """GPU curvature of flat surface matches CPU."""
        from vibespatial.raster.algebra import raster_curvature

        cpu = raster_curvature(flat_dem, use_gpu=False)
        gpu = raster_curvature(flat_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_tilted_surface(self, tilted_dem):
        """GPU curvature of tilted surface matches CPU."""
        from vibespatial.raster.algebra import raster_curvature

        cpu = raster_curvature(tilted_dem, use_gpu=False)
        gpu = raster_curvature(tilted_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_nodata_propagation(self, nodata_dem):
        """GPU curvature nodata propagation matches CPU."""
        from vibespatial.raster.algebra import raster_curvature

        cpu = raster_curvature(nodata_dem, use_gpu=False)
        gpu = raster_curvature(nodata_dem, use_gpu=True)
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_manual_curvature(self):
        """GPU curvature matches manual calculation."""
        from vibespatial.raster.algebra import raster_curvature

        data = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 4, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        dem = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))
        gpu = raster_curvature(dem, use_gpu=True)
        np.testing.assert_almost_equal(gpu.to_numpy()[2, 2], 1200.0)


# ---------------------------------------------------------------------------
# GPU border nodata metadata tests (Bug #5)
# ---------------------------------------------------------------------------


@requires_gpu
class TestBorderNodataMetadata_GPU:
    """GPU path must also declare nodata correctly for border sentinels."""

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_tri_gpu_nodata_declared_when_input_has_no_nodata(self, flat_dem):
        """GPU TRI output must declare nodata=-9999.0 for border sentinel."""
        from vibespatial.raster.algebra import raster_tri

        assert flat_dem.nodata is None
        result = raster_tri(flat_dem, use_gpu=True)
        assert result.nodata == -9999.0
        data = result.to_numpy()
        assert data[0, 0] == result.nodata

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_gpu_cpu_nodata_metadata_parity(self, flat_dem):
        """GPU and CPU paths must produce identical nodata metadata."""
        from vibespatial.raster.algebra import raster_tri

        cpu = raster_tri(flat_dem, use_gpu=False)
        gpu = raster_tri(flat_dem, use_gpu=True)
        assert cpu.nodata == gpu.nodata
        np.testing.assert_array_almost_equal(gpu.to_numpy(), cpu.to_numpy())


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExports:
    def test_importable_from_package(self):
        """Terrain derivative functions are importable from vibespatial.raster."""
        from vibespatial.raster import raster_curvature, raster_tpi, raster_tri

        assert callable(raster_tri)
        assert callable(raster_tpi)
        assert callable(raster_curvature)
