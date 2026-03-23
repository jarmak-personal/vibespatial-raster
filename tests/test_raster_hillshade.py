"""Tests for raster hillshade: CPU and GPU paths."""

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
skip_no_gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_dem():
    """10x10 flat DEM at elevation 100."""
    data = np.ones((10, 10), dtype=np.float64) * 100.0
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))


@pytest.fixture
def tilted_dem():
    """20x20 DEM with linear east-west ramp (elevation increases with column)."""
    data = np.tile(np.arange(20, dtype=np.float64), (20, 1))
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 20.0))


@pytest.fixture
def nodata_dem():
    """10x10 DEM with nodata pixels."""
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    data[3, 5] = -9999.0
    data[7, 2] = -9999.0
    return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))


@pytest.fixture
def large_dem():
    """100x100 DEM with varied terrain for GPU dispatch."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((100, 100)).astype(np.float64) * 50 + 500
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0))


# ---------------------------------------------------------------------------
# CPU Tests
# ---------------------------------------------------------------------------


class TestHillshadeCPU:
    def test_flat_surface(self, flat_dem):
        """Flat surface should produce uniform, non-zero hillshade."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=False)
        hs = result.to_numpy()

        assert hs.dtype == np.uint8
        # Flat surface: slope=0, so hillshade = cos(zenith) everywhere
        # zenith = 90 - 45 = 45 deg, cos(45) ~= 0.707, so ~180
        expected_val = int(np.cos(np.radians(45.0)) * 255 + 0.5)
        # Interior pixels should be close to this value
        np.testing.assert_array_equal(hs[2:-2, 2:-2], expected_val)

    def test_output_dtype(self, flat_dem):
        """Hillshade output must be uint8."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=False)
        assert result.dtype == np.uint8

    def test_output_range(self, tilted_dem):
        """All hillshade values must be in [0, 255]."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(tilted_dem, use_gpu=False)
        hs = result.to_numpy()
        assert hs.min() >= 0
        assert hs.max() <= 255

    def test_tilted_nonzero_slope(self, tilted_dem):
        """Tilted surface should produce non-zero hillshade different from flat."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(tilted_dem, use_gpu=False)
        hs = result.to_numpy()
        # A uniform tilt has constant slope/aspect, so interior pixels
        # get a uniform hillshade value, but it differs from a flat surface.
        flat_val = int(np.cos(np.radians(45.0)) * 255 + 0.5)
        interior = hs[2:-2, 2:-2]
        # All interior pixels should be the same (uniform gradient)
        assert interior.min() == interior.max()
        # But that value should differ from the flat-surface value since
        # slope > 0 contributes a cos(azimuth - aspect) term
        tilted_val = int(interior[0, 0])
        assert tilted_val != flat_val, f"Expected != {flat_val}, got {tilted_val}"

    def test_varied_terrain_has_variation(self, large_dem):
        """Varied terrain should produce varying hillshade values."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(large_dem, use_gpu=False)
        hs = result.to_numpy()
        interior = hs[2:-2, 2:-2]
        assert interior.max() > interior.min()

    def test_different_azimuths(self, tilted_dem):
        """Different azimuths should produce different results."""
        from vibespatial.raster.algebra import raster_hillshade

        hs_nw = raster_hillshade(tilted_dem, azimuth=315.0, use_gpu=False).to_numpy()
        hs_se = raster_hillshade(tilted_dem, azimuth=135.0, use_gpu=False).to_numpy()
        # Opposite illumination directions should differ significantly
        assert not np.array_equal(hs_nw, hs_se)

    def test_different_altitudes(self, tilted_dem):
        """Different altitudes should produce different results."""
        from vibespatial.raster.algebra import raster_hillshade

        hs_low = raster_hillshade(tilted_dem, altitude=15.0, use_gpu=False).to_numpy()
        hs_high = raster_hillshade(tilted_dem, altitude=75.0, use_gpu=False).to_numpy()
        assert not np.array_equal(hs_low, hs_high)

    def test_z_factor(self, tilted_dem):
        """z_factor should exaggerate vertical relief."""
        from vibespatial.raster.algebra import raster_hillshade

        hs_1x = raster_hillshade(tilted_dem, z_factor=1.0, use_gpu=False).to_numpy()
        hs_5x = raster_hillshade(tilted_dem, z_factor=5.0, use_gpu=False).to_numpy()
        # Exaggerated relief should produce different shading
        assert not np.array_equal(hs_1x, hs_5x)

    def test_nodata_propagation(self, nodata_dem):
        """Nodata should propagate to neighbors in 3x3 window."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(nodata_dem, use_gpu=False)
        hs = result.to_numpy()

        # Nodata value should be 0
        assert result.nodata == 0

        # The nodata pixel itself should be 0
        assert hs[3, 5] == 0

        # Neighbors of nodata pixel should also be nodata (0)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                r, c = 3 + dy, 5 + dx
                if 0 <= r < 10 and 0 <= c < 10:
                    assert hs[r, c] == 0, f"Expected nodata at ({r},{c}), got {hs[r, c]}"

    def test_no_nodata_returns_none(self, flat_dem):
        """When input has no nodata, output nodata should be None."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=False)
        assert result.nodata is None

    def test_preserves_affine_and_crs(self, flat_dem):
        """Output should preserve affine transform and CRS."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=False)
        assert result.affine == flat_dem.affine
        assert result.crs == flat_dem.crs

    def test_overhead_illumination(self, tilted_dem):
        """With sun directly overhead (altitude=90), all slopes equally lit."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(tilted_dem, altitude=90.0, use_gpu=False)
        hs = result.to_numpy()
        # zenith = 0, so hillshade = cos(slope). Flat pixels get 255.
        # All values should be fairly high (close to cos(slope)*255)
        assert hs.min() > 100  # even steep slopes get illuminated overhead


# ---------------------------------------------------------------------------
# GPU Tests
# ---------------------------------------------------------------------------


@requires_gpu
@skip_no_gpu
class TestHillshadeGPU:
    def test_gpu_matches_cpu_flat(self, flat_dem):
        """GPU and CPU should produce identical results on flat terrain."""
        from vibespatial.raster.algebra import raster_hillshade

        cpu = raster_hillshade(flat_dem, use_gpu=False).to_numpy()
        gpu = raster_hillshade(flat_dem, use_gpu=True).to_numpy()
        np.testing.assert_array_equal(gpu, cpu)

    def test_gpu_matches_cpu_tilted(self, tilted_dem):
        """GPU and CPU should produce matching results on tilted terrain."""
        from vibespatial.raster.algebra import raster_hillshade

        cpu = raster_hillshade(tilted_dem, use_gpu=False).to_numpy()
        gpu = raster_hillshade(tilted_dem, use_gpu=True).to_numpy()
        # Allow +/- 1 tolerance for floating point rounding in uint8 quantization
        np.testing.assert_allclose(gpu.astype(np.int16), cpu.astype(np.int16), atol=1)

    def test_gpu_matches_cpu_large(self, large_dem):
        """GPU and CPU should match on a larger DEM."""
        from vibespatial.raster.algebra import raster_hillshade

        cpu = raster_hillshade(large_dem, use_gpu=False).to_numpy()
        gpu = raster_hillshade(large_dem, use_gpu=True).to_numpy()
        # Allow +/- 1 tolerance for floating point rounding
        np.testing.assert_allclose(gpu.astype(np.int16), cpu.astype(np.int16), atol=1)

    def test_gpu_nodata_propagation(self, nodata_dem):
        """GPU should propagate nodata identically to CPU."""
        from vibespatial.raster.algebra import raster_hillshade

        cpu = raster_hillshade(nodata_dem, use_gpu=False).to_numpy()
        gpu = raster_hillshade(nodata_dem, use_gpu=True).to_numpy()

        # Nodata pixels must match exactly
        cpu_nodata = cpu == 0
        gpu_nodata = gpu == 0
        np.testing.assert_array_equal(gpu_nodata, cpu_nodata)

        # Non-nodata pixels should be close
        valid = ~cpu_nodata
        if valid.any():
            np.testing.assert_allclose(
                gpu[valid].astype(np.int16), cpu[valid].astype(np.int16), atol=1
            )

    def test_gpu_different_azimuths(self, tilted_dem):
        """GPU results should vary with azimuth."""
        from vibespatial.raster.algebra import raster_hillshade

        hs_nw = raster_hillshade(tilted_dem, azimuth=315.0, use_gpu=True).to_numpy()
        hs_se = raster_hillshade(tilted_dem, azimuth=135.0, use_gpu=True).to_numpy()
        assert not np.array_equal(hs_nw, hs_se)

    def test_gpu_output_dtype(self, flat_dem):
        """GPU output must be uint8."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=True)
        assert result.dtype == np.uint8

    def test_gpu_diagnostics(self, flat_dem):
        """GPU path should produce diagnostic events."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=True)
        assert len(result.diagnostics) > 0
        detail = result.diagnostics[-1].detail
        assert "gpu_hillshade" in detail


# ---------------------------------------------------------------------------
# Auto-dispatch Tests
# ---------------------------------------------------------------------------


class TestHillshadeAutoDispatch:
    def test_auto_dispatch_works(self, flat_dem):
        """Auto dispatch should work regardless of GPU availability."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem)
        assert result is not None
        assert result.dtype == np.uint8

    def test_force_cpu(self, flat_dem):
        """use_gpu=False should always use CPU path."""
        from vibespatial.raster.algebra import raster_hillshade

        result = raster_hillshade(flat_dem, use_gpu=False)
        assert result is not None
        assert result.dtype == np.uint8
