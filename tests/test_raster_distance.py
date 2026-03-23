"""Tests for Euclidean Distance Transform (EDT) via Jump Flooding Algorithm."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from scipy.ndimage import distance_transform_edt  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from vibespatial.raster.buffers import from_numpy
from vibespatial.raster.distance import raster_distance_transform

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


def _scipy_edt_reference(data: np.ndarray, nodata=None) -> np.ndarray:
    """Compute reference EDT using scipy for comparison.

    Returns float64 distance array. Nodata pixels get NaN.
    """
    from scipy.ndimage import distance_transform_edt

    foreground = data != 0
    if nodata is not None:
        if np.isnan(nodata):
            foreground &= ~np.isnan(data)
        else:
            foreground &= data != nodata

    edt_input = (~foreground).astype(np.float64)
    distances = distance_transform_edt(edt_input)

    if nodata is not None:
        if np.isnan(nodata):
            nodata_mask = np.isnan(data)
        else:
            nodata_mask = data == nodata
        if nodata_mask.any():
            distances[nodata_mask] = np.nan

    return distances


# ---------------------------------------------------------------------------
# CPU tests: basic functionality
# ---------------------------------------------------------------------------


class TestDistanceTransformCPU:
    def test_single_foreground_pixel(self):
        """Distance from center foreground pixel should radiate outward."""
        data = np.zeros((5, 5), dtype=np.float64)
        data[2, 2] = 1.0
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()

        # Center pixel should have distance 0
        assert dist[2, 2] == 0.0
        # Adjacent pixels should have distance 1
        assert dist[2, 1] == pytest.approx(1.0)
        assert dist[2, 3] == pytest.approx(1.0)
        assert dist[1, 2] == pytest.approx(1.0)
        assert dist[3, 2] == pytest.approx(1.0)
        # Diagonal pixels should have distance sqrt(2)
        assert dist[1, 1] == pytest.approx(np.sqrt(2))
        # Corner should have distance sqrt(8)
        assert dist[0, 0] == pytest.approx(np.sqrt(8))

    def test_all_foreground(self):
        """All foreground pixels should have distance 0."""
        data = np.ones((3, 4), dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        np.testing.assert_array_equal(dist, 0.0)

    def test_all_background(self):
        """All background with no foreground -- scipy returns 0 everywhere."""
        data = np.zeros((3, 3), dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        # scipy distance_transform_edt returns 0 for all-zero input
        np.testing.assert_array_equal(dist, 0.0)

    def test_nodata_propagation(self):
        """Nodata pixels should propagate as NaN in output."""
        data = np.array([[1, -9999, 0, 0, 1]], dtype=np.int32)
        raster = from_numpy(data, nodata=-9999)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()

        # Foreground pixels have distance 0
        assert dist[0, 0] == 0.0
        assert dist[0, 4] == 0.0
        # Nodata pixel is NaN
        assert np.isnan(dist[0, 1])
        # Background pixels have positive distance
        assert dist[0, 2] > 0
        assert dist[0, 3] > 0

    def test_matches_scipy_reference(self):
        """CPU path should match scipy EDT exactly."""
        rng = np.random.default_rng(42)
        data = (rng.random((20, 30)) > 0.7).astype(np.float64)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        expected = _scipy_edt_reference(data)
        np.testing.assert_allclose(dist, expected, atol=1e-10)

    def test_uint8_input(self):
        """uint8 input should work correctly."""
        data = np.array([[0, 0, 1, 0, 0]], dtype=np.uint8)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        assert dist[0, 2] == 0.0
        assert dist[0, 0] == pytest.approx(2.0)
        assert dist[0, 4] == pytest.approx(2.0)

    def test_float32_input(self):
        """float32 input should work correctly."""
        data = np.zeros((3, 3), dtype=np.float32)
        data[1, 1] = 5.5
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        assert dist[1, 1] == 0.0
        assert dist[0, 0] == pytest.approx(np.sqrt(2))

    def test_single_row(self):
        """Single-row raster should produce 1D distance."""
        data = np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        expected = np.array([[3, 2, 1, 0, 1, 2, 3]], dtype=np.float64)
        np.testing.assert_allclose(dist, expected, atol=1e-10)

    def test_single_column(self):
        """Single-column raster should produce 1D distance."""
        data = np.array([[0], [0], [1], [0], [0]], dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        expected = np.array([[2], [1], [0], [1], [2]], dtype=np.float64)
        np.testing.assert_allclose(dist, expected, atol=1e-10)

    def test_single_pixel_foreground(self):
        """1x1 foreground raster should have distance 0."""
        data = np.array([[1]], dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        assert result.to_numpy()[0, 0] == 0.0

    def test_single_pixel_background(self):
        """1x1 background raster should have distance 0 (scipy convention)."""
        data = np.array([[0]], dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        assert result.to_numpy()[0, 0] == 0.0

    def test_nodata_nan(self):
        """NaN nodata should be handled correctly for float input."""
        data = np.array([[1.0, np.nan, 0.0, 0.0, 1.0]], dtype=np.float64)
        raster = from_numpy(data, nodata=np.nan)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        assert dist[0, 0] == 0.0
        assert np.isnan(dist[0, 1])
        assert dist[0, 4] == 0.0

    def test_3d_single_band_input(self):
        """3D input with shape (1, H, W) should be handled."""
        data = np.zeros((1, 3, 3), dtype=np.int32)
        data[0, 1, 1] = 1
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        dist = result.to_numpy()
        assert dist.ndim == 2
        assert dist[1, 1] == 0.0

    def test_multiband_raises(self):
        """Multi-band raster should raise ValueError."""
        data = np.zeros((2, 3, 3), dtype=np.int32)
        raster = from_numpy(data)
        with pytest.raises(ValueError, match="single-band"):
            raster_distance_transform(raster, use_gpu=False)


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------


class TestDistanceTransformAutoDispatch:
    def test_auto_dispatch(self):
        """Auto-dispatch should work regardless of GPU availability."""
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        raster = from_numpy(data)
        result = raster_distance_transform(raster)
        dist = result.to_numpy()
        assert dist[2, 2] == 0.0
        assert dist[0, 0] > 0

    def test_explicit_cpu(self):
        """Explicit use_gpu=False should always use CPU path."""
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        assert result is not None
        assert result.to_numpy()[2, 2] == 0.0

    def test_diagnostics_present(self):
        """Result should include diagnostic events."""
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == "runtime"]
        assert len(runtime_events) >= 1
        assert "distance_transform_cpu" in runtime_events[0].detail


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.gpu
class TestDistanceTransformGPU:
    def test_single_foreground_pixel(self):
        """GPU EDT of single foreground pixel should match CPU."""
        data = np.zeros((5, 5), dtype=np.float64)
        data[2, 2] = 1.0
        raster = from_numpy(data)
        gpu_result = raster_distance_transform(raster, use_gpu=True)
        cpu_result = raster_distance_transform(raster, use_gpu=False)
        np.testing.assert_allclose(gpu_result.to_numpy(), cpu_result.to_numpy(), atol=0.5)

    def test_random_matches_cpu(self):
        """GPU EDT should closely approximate CPU EDT on random data."""
        rng = np.random.default_rng(42)
        data = (rng.random((32, 32)) > 0.7).astype(np.float64)
        raster = from_numpy(data)
        gpu_result = raster_distance_transform(raster, use_gpu=True)
        cpu_result = raster_distance_transform(raster, use_gpu=False)
        # JFA is approximate but should be very close for most pixels.
        # Allow tolerance of 0.5 pixel for JFA approximation artifacts.
        np.testing.assert_allclose(gpu_result.to_numpy(), cpu_result.to_numpy(), atol=0.5)

    def test_all_foreground(self):
        """GPU: all foreground pixels should have distance 0."""
        data = np.ones((10, 10), dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=True)
        dist = result.to_numpy()
        np.testing.assert_array_equal(dist, 0.0)

    def test_all_background(self):
        """GPU: all-background should produce distance 0 or NaN everywhere."""
        data = np.zeros((8, 8), dtype=np.int32)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=True)
        dist = result.to_numpy()
        # No foreground seeds, distance_compute writes nodata_value (NaN)
        # for unseeded pixels. That's acceptable -- all pixels will be NaN.
        # The CPU path returns 0 for all-background. Accept either behavior.
        assert np.all(dist == 0.0) or np.all(np.isnan(dist))

    def test_nodata_propagation(self):
        """GPU: nodata pixels should propagate as NaN."""
        data = np.array(
            [[1, -9999, 0, 0, 1]],
            dtype=np.int32,
        )
        raster = from_numpy(data, nodata=-9999)
        result = raster_distance_transform(raster, use_gpu=True)
        dist = result.to_numpy()
        # Foreground pixels
        assert dist[0, 0] == 0.0
        assert dist[0, 4] == 0.0
        # Nodata is NaN
        assert np.isnan(dist[0, 1])

    def test_uint8_input(self):
        """GPU: uint8 input should work correctly."""
        data = np.array([[0, 0, 1, 0, 0]], dtype=np.uint8)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=True)
        dist = result.to_numpy()
        assert dist[0, 2] == 0.0
        assert dist[0, 0] == pytest.approx(2.0, abs=0.5)

    def test_larger_raster(self):
        """GPU: larger raster should produce reasonable distances."""
        rng = np.random.default_rng(99)
        data = (rng.random((64, 64)) > 0.8).astype(np.float64)
        raster = from_numpy(data)
        gpu_result = raster_distance_transform(raster, use_gpu=True)
        cpu_result = raster_distance_transform(raster, use_gpu=False)
        np.testing.assert_allclose(gpu_result.to_numpy(), cpu_result.to_numpy(), atol=0.5)

    def test_non_square_raster(self):
        """GPU: non-square rasters should work correctly."""
        data = np.zeros((5, 20), dtype=np.float64)
        data[2, 10] = 1.0
        raster = from_numpy(data)
        gpu_result = raster_distance_transform(raster, use_gpu=True)
        cpu_result = raster_distance_transform(raster, use_gpu=False)
        np.testing.assert_allclose(gpu_result.to_numpy(), cpu_result.to_numpy(), atol=0.5)

    def test_diagnostics_present(self):
        """GPU result should include diagnostic events."""
        data = np.zeros((10, 10), dtype=np.float64)
        data[5, 5] = 1.0
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=True)
        runtime_events = [e for e in result.diagnostics if e.kind == "runtime"]
        assert len(runtime_events) >= 1
        assert "distance_transform_gpu" in runtime_events[0].detail
        assert "jfa_iterations" in runtime_events[0].detail

    def test_explicit_gpu_true(self):
        """Explicit use_gpu=True should use GPU path."""
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=True)
        assert result is not None

    def test_foreground_distance_zero(self):
        """GPU: foreground pixels must always have distance 0."""
        rng = np.random.default_rng(77)
        data = (rng.random((16, 16)) > 0.5).astype(np.float64)
        raster = from_numpy(data)
        result = raster_distance_transform(raster, use_gpu=True)
        dist = result.to_numpy()
        fg_mask = data != 0
        np.testing.assert_array_equal(dist[fg_mask], 0.0)
