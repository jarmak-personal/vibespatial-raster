"""Tests for GPU/CPU boundary consistency in slope, aspect, and hillshade.

Bug #11: GPU slope/aspect kernel used 0-fill for out-of-bounds halo pixels,
while CPU uses np.pad(mode="edge") (edge replication). This caused border
pixels to get artificially steep slopes on GPU because the 0-fill created
a false elevation drop at the boundary.

The fix changes GPU kernels to use coordinate clamping (edge replication)
so border pixel values match the CPU path.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import from_numpy

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_GPU, reason="CuPy not available"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dem(data: np.ndarray, nodata=None) -> from_numpy:
    """Create a DEM raster with unit cell size."""
    return from_numpy(
        data.astype(np.float64),
        nodata=nodata,
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, float(data.shape[0])),
    )


# ---------------------------------------------------------------------------
# Slope: GPU vs CPU border pixel comparison
# ---------------------------------------------------------------------------


class TestSlopeBorderConsistency:
    """GPU slope border pixels must match CPU slope border pixels."""

    def test_flat_surface_border_slope_is_zero(self):
        """On a flat surface, ALL pixels (including borders) should have 0 slope.

        Before the fix, GPU border pixels had non-zero slope because
        0-fill created a false elevation drop at the boundary.
        """
        from vibespatial.raster.algebra import raster_slope

        data = np.ones((8, 8), dtype=np.float64) * 500.0
        dem = _make_dem(data)

        gpu_result = raster_slope(dem, use_gpu=True).to_numpy()
        cpu_result = raster_slope(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # All pixels should be zero slope on a flat surface
        np.testing.assert_allclose(
            gpu_result, 0.0, atol=1e-10, err_msg="GPU slope on flat surface should be 0 everywhere"
        )
        np.testing.assert_allclose(
            cpu_result, 0.0, atol=1e-10, err_msg="CPU slope on flat surface should be 0 everywhere"
        )

    def test_elevated_flat_surface_border_matches(self):
        """Elevated flat surface: edge replication means border gradient is 0.

        With 0-fill, a 1000m flat surface would produce huge false slopes
        at borders because the halo would be 0 instead of 1000.
        """
        from vibespatial.raster.algebra import raster_slope

        data = np.ones((10, 10), dtype=np.float64) * 1000.0
        dem = _make_dem(data)

        gpu_result = raster_slope(dem, use_gpu=True).to_numpy()
        cpu_result = raster_slope(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # GPU and CPU should produce identical results for all pixels
        np.testing.assert_allclose(
            gpu_result,
            cpu_result,
            atol=1e-10,
            err_msg="GPU and CPU slope should match for flat elevated surface",
        )

    def test_tilted_surface_border_matches(self):
        """Linear ramp: border pixels should match between GPU and CPU.

        Edge replication at boundaries means the boundary gradient matches
        the interior gradient (constant ramp).
        """
        from vibespatial.raster.algebra import raster_slope

        # East-facing ramp: elevation = column index
        data = np.tile(np.arange(12, dtype=np.float64), (10, 1))
        dem = _make_dem(data)

        gpu_result = raster_slope(dem, use_gpu=True).to_numpy()
        cpu_result = raster_slope(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # Compare ALL pixels, including borders
        np.testing.assert_allclose(
            gpu_result,
            cpu_result,
            atol=1e-10,
            err_msg="GPU and CPU slope should match on tilted surface (all pixels)",
        )

    def test_border_row_zero_matches(self):
        """Specifically verify row 0 (top border) matches CPU."""
        from vibespatial.raster.algebra import raster_slope

        data = np.tile(np.arange(8, dtype=np.float64), (8, 1))
        dem = _make_dem(data)

        gpu_result = raster_slope(dem, use_gpu=True).to_numpy()
        cpu_result = raster_slope(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # Top row
        np.testing.assert_allclose(
            gpu_result[0, :], cpu_result[0, :], atol=1e-10, err_msg="Top border row mismatch"
        )
        # Bottom row
        np.testing.assert_allclose(
            gpu_result[-1, :], cpu_result[-1, :], atol=1e-10, err_msg="Bottom border row mismatch"
        )
        # Left column
        np.testing.assert_allclose(
            gpu_result[:, 0], cpu_result[:, 0], atol=1e-10, err_msg="Left border column mismatch"
        )
        # Right column
        np.testing.assert_allclose(
            gpu_result[:, -1], cpu_result[:, -1], atol=1e-10, err_msg="Right border column mismatch"
        )


# ---------------------------------------------------------------------------
# Aspect: GPU vs CPU border pixel comparison
# ---------------------------------------------------------------------------


class TestAspectBorderConsistency:
    """GPU aspect border pixels must match CPU aspect border pixels."""

    def test_east_facing_ramp_border_matches(self):
        """East-facing ramp: aspect at borders should match CPU."""
        from vibespatial.raster.algebra import raster_aspect

        # Elevation increases to the east
        data = np.tile(np.arange(10, dtype=np.float64), (10, 1))
        dem = _make_dem(data)

        gpu_result = raster_aspect(dem, use_gpu=True).to_numpy()
        cpu_result = raster_aspect(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # All pixels should match
        np.testing.assert_allclose(
            gpu_result,
            cpu_result,
            atol=1e-10,
            err_msg="GPU and CPU aspect should match (all pixels)",
        )

    def test_south_facing_ramp_border_matches(self):
        """South-facing ramp: aspect at borders should match CPU."""
        from vibespatial.raster.algebra import raster_aspect

        # Elevation increases to the south (rows increase)
        data = np.tile(np.arange(10, dtype=np.float64).reshape(10, 1), (1, 10))
        dem = _make_dem(data)

        gpu_result = raster_aspect(dem, use_gpu=True).to_numpy()
        cpu_result = raster_aspect(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        np.testing.assert_allclose(
            gpu_result,
            cpu_result,
            atol=1e-10,
            err_msg="GPU and CPU aspect should match for south-facing ramp",
        )

    def test_corner_pixels_match(self):
        """Corner pixels are most sensitive to boundary handling."""
        from vibespatial.raster.algebra import raster_aspect

        # Diagonal ramp
        y, x = np.mgrid[0:8, 0:8]
        data = (x + y).astype(np.float64)
        dem = _make_dem(data)

        gpu_result = raster_aspect(dem, use_gpu=True).to_numpy()
        cpu_result = raster_aspect(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # Check all four corners
        for label, r, c in [("TL", 0, 0), ("TR", 0, -1), ("BL", -1, 0), ("BR", -1, -1)]:
            np.testing.assert_allclose(
                gpu_result[r, c],
                cpu_result[r, c],
                atol=1e-10,
                err_msg=f"Corner {label} aspect mismatch",
            )


# ---------------------------------------------------------------------------
# Hillshade: GPU vs CPU border pixel comparison
# ---------------------------------------------------------------------------


class TestHillshadeBorderConsistency:
    """GPU hillshade border pixels must match CPU hillshade border pixels.

    Hillshade already used edge replication before the fix, so this test
    serves as a regression guard.
    """

    def test_flat_surface_uniform_hillshade(self):
        """Flat surface should produce uniform hillshade everywhere."""
        from vibespatial.raster.algebra import raster_hillshade

        data = np.ones((10, 10), dtype=np.float64) * 500.0
        dem = _make_dem(data)

        gpu_result = raster_hillshade(dem, use_gpu=True).to_numpy()
        cpu_result = raster_hillshade(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # All GPU pixels should match CPU pixels
        np.testing.assert_array_equal(
            gpu_result, cpu_result, err_msg="Hillshade GPU/CPU mismatch on flat surface"
        )

    def test_ramp_border_hillshade_matches(self):
        """Ramp surface: hillshade border pixels should match CPU."""
        from vibespatial.raster.algebra import raster_hillshade

        data = np.tile(np.arange(12, dtype=np.float64), (10, 1))
        dem = _make_dem(data)

        gpu_result = raster_hillshade(dem, use_gpu=True).to_numpy()
        cpu_result = raster_hillshade(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # Allow +/-1 for uint8 rounding differences
        np.testing.assert_allclose(
            gpu_result.astype(np.int16),
            cpu_result.astype(np.int16),
            atol=1,
            err_msg="Hillshade GPU/CPU border mismatch on ramp surface",
        )


# ---------------------------------------------------------------------------
# Regression test: 0-fill detection
# ---------------------------------------------------------------------------


class TestZeroFillRegression:
    """Verify that high-elevation flat surfaces do not produce border artifacts.

    Before the fix, a flat surface at elevation E would have border halo
    filled with 0.0 instead of E, creating a gradient of E/(8*cellsize)
    which translates to a large false slope at borders.
    """

    def test_high_elevation_no_border_slope(self):
        """A 5000m plateau should have zero slope at all borders."""
        from vibespatial.raster.algebra import raster_slope

        data = np.ones((6, 6), dtype=np.float64) * 5000.0
        dem = _make_dem(data)

        gpu_slope = raster_slope(dem, use_gpu=True).to_numpy()
        if gpu_slope.ndim == 3:
            gpu_slope = gpu_slope[0]

        # Extract border pixels only
        border_mask = np.zeros_like(gpu_slope, dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        border_slopes = gpu_slope[border_mask]

        # With edge replication, all border slopes must be 0
        np.testing.assert_allclose(
            border_slopes,
            0.0,
            atol=1e-10,
            err_msg="GPU border slopes should be 0 on flat 5000m surface "
            "(0-fill regression: halo was 0 instead of 5000)",
        )

    def test_varying_elevation_border_consistency(self):
        """Compare GPU vs CPU on a surface with varying elevation.

        This tests that the gradient computation at boundaries is consistent
        with edge replication, not just that flat surfaces work.
        """
        from vibespatial.raster.algebra import raster_slope

        np.random.seed(42)
        data = np.random.uniform(100, 1000, (10, 10)).astype(np.float64)
        dem = _make_dem(data)

        gpu_result = raster_slope(dem, use_gpu=True).to_numpy()
        cpu_result = raster_slope(dem, use_gpu=False).to_numpy()

        if gpu_result.ndim == 3:
            gpu_result = gpu_result[0]
        if cpu_result.ndim == 3:
            cpu_result = cpu_result[0]

        # Full array comparison
        np.testing.assert_allclose(
            gpu_result,
            cpu_result,
            atol=1e-10,
            err_msg="GPU and CPU slope should match on random surface",
        )
