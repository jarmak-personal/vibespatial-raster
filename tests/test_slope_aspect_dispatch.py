"""Tests for slope/aspect GPU dispatch: verify auto-dispatch uses _should_use_gpu.

Validates Bug #13 fix (slope/aspect used _has_cupy instead of _should_use_gpu)
and Bug #18 fix (duplicate _should_use_gpu definitions with conflicting thresholds).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from vibespatial.raster.buffers import from_numpy

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

skip_no_gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_dem():
    """5x5 DEM — 25 pixels, well below the 10,000 dispatch threshold."""
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def above_threshold_dem():
    """110x100 DEM — 11,000 pixels, above the 10,000 dispatch threshold."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((110, 100)).astype(np.float64) * 50 + 500
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 110.0))


# ---------------------------------------------------------------------------
# Bug #13: slope/aspect must use _should_use_gpu, not _has_cupy
# ---------------------------------------------------------------------------


class TestSlopeDispatch:
    """Verify raster_slope auto-dispatch respects _should_use_gpu threshold."""

    def test_tiny_raster_auto_dispatches_to_cpu(self, tiny_dem):
        """A 5x5 raster (25 pixels) is below the 10k threshold and must use CPU."""
        from vibespatial.raster.algebra import raster_slope

        result = raster_slope(tiny_dem)
        # Check diagnostics to confirm CPU path was taken
        assert len(result.diagnostics) > 0
        detail = result.diagnostics[-1].detail
        assert "cpu_slope_fused" in detail, f"Expected CPU dispatch for tiny raster, got: {detail}"

    def test_force_cpu_always_uses_cpu(self, tiny_dem):
        """use_gpu=False must always select the CPU path."""
        from vibespatial.raster.algebra import raster_slope

        result = raster_slope(tiny_dem, use_gpu=False)
        detail = result.diagnostics[-1].detail
        assert "cpu_slope_fused" in detail

    @skip_no_gpu
    def test_force_gpu_always_uses_gpu(self, tiny_dem):
        """use_gpu=True must always select the GPU path, even for tiny rasters."""
        from vibespatial.raster.algebra import raster_slope

        result = raster_slope(tiny_dem, use_gpu=True)
        detail = result.diagnostics[-1].detail
        assert "gpu_slope_fused" in detail

    def test_auto_dispatch_calls_should_use_gpu(self, tiny_dem):
        """Auto-dispatch must call _should_use_gpu, not _has_cupy."""
        from vibespatial.raster import algebra as algebra_mod
        from vibespatial.raster.algebra import raster_slope

        with patch.object(algebra_mod, "_should_use_gpu", return_value=False) as mock_dispatch:
            raster_slope(tiny_dem)
            mock_dispatch.assert_called_once()

    @skip_no_gpu
    def test_above_threshold_auto_dispatches_to_gpu(self, above_threshold_dem):
        """An 11,000-pixel raster is above the 10k threshold and must use GPU."""
        from vibespatial.raster.algebra import raster_slope

        result = raster_slope(above_threshold_dem)
        detail = result.diagnostics[-1].detail
        assert "gpu_slope_fused" in detail, (
            f"Expected GPU dispatch for above-threshold raster, got: {detail}"
        )


class TestAspectDispatch:
    """Verify raster_aspect auto-dispatch respects _should_use_gpu threshold."""

    def test_tiny_raster_auto_dispatches_to_cpu(self, tiny_dem):
        """A 5x5 raster (25 pixels) is below the 10k threshold and must use CPU."""
        from vibespatial.raster.algebra import raster_aspect

        result = raster_aspect(tiny_dem)
        assert len(result.diagnostics) > 0
        detail = result.diagnostics[-1].detail
        assert "cpu_aspect_fused" in detail, f"Expected CPU dispatch for tiny raster, got: {detail}"

    def test_force_cpu_always_uses_cpu(self, tiny_dem):
        """use_gpu=False must always select the CPU path."""
        from vibespatial.raster.algebra import raster_aspect

        result = raster_aspect(tiny_dem, use_gpu=False)
        detail = result.diagnostics[-1].detail
        assert "cpu_aspect_fused" in detail

    @skip_no_gpu
    def test_force_gpu_always_uses_gpu(self, tiny_dem):
        """use_gpu=True must always select the GPU path, even for tiny rasters."""
        from vibespatial.raster.algebra import raster_aspect

        result = raster_aspect(tiny_dem, use_gpu=True)
        detail = result.diagnostics[-1].detail
        assert "gpu_aspect_fused" in detail

    def test_auto_dispatch_calls_should_use_gpu(self, tiny_dem):
        """Auto-dispatch must call _should_use_gpu, not _has_cupy."""
        from vibespatial.raster import algebra as algebra_mod
        from vibespatial.raster.algebra import raster_aspect

        with patch.object(algebra_mod, "_should_use_gpu", return_value=False) as mock_dispatch:
            raster_aspect(tiny_dem)
            mock_dispatch.assert_called_once()

    @skip_no_gpu
    def test_above_threshold_auto_dispatches_to_gpu(self, above_threshold_dem):
        """An 11,000-pixel raster is above the 10k threshold and must use GPU."""
        from vibespatial.raster.algebra import raster_aspect

        result = raster_aspect(above_threshold_dem)
        detail = result.diagnostics[-1].detail
        assert "gpu_aspect_fused" in detail, (
            f"Expected GPU dispatch for above-threshold raster, got: {detail}"
        )


# ---------------------------------------------------------------------------
# Bug #18: single _should_use_gpu with consistent 10k threshold
# ---------------------------------------------------------------------------


class TestShouldUseGpuThreshold:
    """Verify there is one _should_use_gpu with a 10,000-pixel threshold."""

    def test_below_threshold_returns_false(self):
        """Raster with fewer than 10,000 pixels should not trigger GPU."""
        from vibespatial.raster.algebra import _should_use_gpu

        small = from_numpy(
            np.zeros((50, 50), dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 50.0),
        )
        # 2,500 pixels < 10,000 threshold
        assert _should_use_gpu(small) is False

    @skip_no_gpu
    def test_above_threshold_returns_true(self):
        """Raster with 10,000+ pixels should trigger GPU when CUDA available."""
        from vibespatial.raster.algebra import _should_use_gpu

        large = from_numpy(
            np.zeros((100, 100), dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),
        )
        # 10,000 pixels == threshold
        assert _should_use_gpu(large) is True

    @skip_no_gpu
    def test_exactly_at_threshold_returns_true(self):
        """Raster with exactly 10,000 pixels should trigger GPU (>= comparison)."""
        from vibespatial.raster.algebra import _should_use_gpu

        exact = from_numpy(
            np.zeros((100, 100), dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 100.0),
        )
        assert exact.pixel_count == 10_000
        assert _should_use_gpu(exact) is True

    @skip_no_gpu
    def test_just_below_threshold_returns_false(self):
        """Raster with 9,999 pixels should not trigger GPU."""
        from vibespatial.raster.algebra import _should_use_gpu

        # 99 * 101 = 9,999 pixels
        below = from_numpy(
            np.zeros((99, 101), dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 99.0),
        )
        assert below.pixel_count == 9_999
        assert _should_use_gpu(below) is False

    def test_custom_threshold(self):
        """Custom threshold parameter should be respected."""
        from vibespatial.raster.algebra import _should_use_gpu

        small = from_numpy(
            np.zeros((5, 5), dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0),
        )
        # 25 pixels, below any reasonable threshold
        assert _should_use_gpu(small, threshold=1_000) is False

    def test_default_threshold_is_10k(self):
        """Verify the default threshold parameter is 10,000."""
        import inspect

        from vibespatial.raster.algebra import _should_use_gpu

        sig = inspect.signature(_should_use_gpu)
        default = sig.parameters["threshold"].default
        assert default == 10_000, f"Expected default threshold 10,000, got {default}"
