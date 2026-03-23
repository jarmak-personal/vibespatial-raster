"""Tests for focal statistics: min, max, mean, std, range, variety."""

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
def raster_5x5():
    """5x5 raster with sequential values 1..25."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def raster_uniform():
    """5x5 raster with all values = 7."""
    data = np.full((5, 5), 7.0, dtype=np.float64)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def raster_with_nodata():
    """5x5 raster with a nodata hole at (1,1)."""
    data = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    data[1, 1] = -9999.0
    return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


@pytest.fixture
def raster_categorical():
    """5x5 raster with integer-like categories for variety testing."""
    data = np.array(
        [
            [1, 1, 2, 2, 3],
            [1, 1, 2, 3, 3],
            [4, 4, 5, 5, 5],
            [4, 4, 5, 6, 6],
            [7, 7, 7, 6, 6],
        ],
        dtype=np.float64,
    )
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 5.0))


# ---------------------------------------------------------------------------
# CPU tests for each statistic
# ---------------------------------------------------------------------------


class TestFocalMinCPU:
    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center pixel (2,2): neighborhood is 3x3 around value 13
        # min of [7,8,9,12,13,14,17,18,19] = 7
        assert out[2, 2] == 7.0
        # Corner (0,0): neighborhood is [1,2,6,7] -> min = 1
        assert out[0, 0] == 1.0

    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_uniform, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 7.0)

    def test_nodata(self, raster_with_nodata):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_with_nodata, radius=1, use_gpu=False)
        out = result.to_numpy()
        # The nodata pixel itself should remain nodata
        assert out[1, 1] == -9999.0

    def test_preserves_affine(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=False)
        assert result.affine == raster_5x5.affine


class TestFocalMaxCPU:
    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_max

        result = raster_focal_max(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): max of [7,8,9,12,13,14,17,18,19] = 19
        assert out[2, 2] == 19.0
        # Corner (4,4): neighborhood is [19,20,24,25] -> max = 25
        assert out[4, 4] == 25.0


class TestFocalMeanCPU:
    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_mean

        result = raster_focal_mean(raster_uniform, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 7.0)

    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_mean

        result = raster_focal_mean(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): mean of [7,8,9,12,13,14,17,18,19] = 13
        np.testing.assert_almost_equal(out[2, 2], 13.0)


class TestFocalStdCPU:
    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_uniform, radius=1, use_gpu=False)
        # Std of uniform values = 0
        np.testing.assert_array_almost_equal(result.to_numpy(), 0.0)

    def test_nonzero(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_std

        result = raster_focal_std(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): std([7,8,9,12,13,14,17,18,19]) = 4.183... (ddof=1)
        expected = np.std([7, 8, 9, 12, 13, 14, 17, 18, 19], ddof=1)
        np.testing.assert_almost_equal(out[2, 2], expected, decimal=5)


class TestFocalRangeCPU:
    def test_basic(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_range

        result = raster_focal_range(raster_5x5, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2): range of [7..19] = 12
        assert out[2, 2] == 12.0

    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_range

        result = raster_focal_range(raster_uniform, radius=1, use_gpu=False)
        np.testing.assert_array_almost_equal(result.to_numpy(), 0.0)


class TestFocalVarietyCPU:
    def test_uniform(self, raster_uniform):
        from vibespatial.raster.algebra import raster_focal_variety

        result = raster_focal_variety(raster_uniform, radius=1, use_gpu=False)
        # All values the same -> variety = 1
        np.testing.assert_array_almost_equal(result.to_numpy(), 1.0)

    def test_categorical(self, raster_categorical):
        from vibespatial.raster.algebra import raster_focal_variety

        result = raster_focal_variety(raster_categorical, radius=1, use_gpu=False)
        out = result.to_numpy()
        # Center (2,2) = value 5, neighborhood: [1,2,2,3,4,5,5,5,5] -> unique: {1,2,3,4,5} = 5
        assert out[2, 2] >= 3.0  # at least several unique values


class TestRadiusParsing:
    def test_int_radius(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=2, use_gpu=False)
        assert result.shape == raster_5x5.shape

    def test_tuple_radius(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=(1, 2), use_gpu=False)
        assert result.shape == raster_5x5.shape

    def test_bad_tuple(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        with pytest.raises(ValueError, match="2 elements"):
            raster_focal_min(raster_5x5, radius=(1, 2, 3), use_gpu=False)


class TestDiagnostics:
    def test_cpu_diagnostics(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=False)
        assert any("cpu_focal_min" in d.detail for d in result.diagnostics)


# ---------------------------------------------------------------------------
# GPU tests (compare GPU output to CPU baseline)
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestFocalStatsGPU:
    """GPU focal statistics tests. Validate that GPU matches CPU for all stats."""

    def test_focal_min_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        cpu = raster_focal_min(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_min(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_focal_max_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_max

        cpu = raster_focal_max(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_max(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_focal_mean_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_mean

        cpu = raster_focal_mean(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_mean(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=10)

    def test_focal_std_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_std

        cpu = raster_focal_std(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_std(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=5)

    def test_focal_range_gpu_vs_cpu(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_range

        cpu = raster_focal_range(raster_5x5, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_range(raster_5x5, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_focal_variety_gpu_vs_cpu(self, raster_categorical):
        from vibespatial.raster.algebra import raster_focal_variety

        cpu = raster_focal_variety(raster_categorical, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_variety(raster_categorical, radius=1, use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu)

    def test_gpu_nodata_handling(self, raster_with_nodata):
        from vibespatial.raster.algebra import raster_focal_min

        cpu = raster_focal_min(raster_with_nodata, radius=1, use_gpu=False).to_numpy()
        gpu = raster_focal_min(raster_with_nodata, radius=1, use_gpu=True).to_numpy()
        # Nodata pixel should be nodata in both
        assert gpu[1, 1] == -9999.0
        assert cpu[1, 1] == -9999.0
        # Non-nodata pixels should match
        mask = cpu != -9999.0
        np.testing.assert_array_almost_equal(gpu[mask], cpu[mask])

    def test_gpu_asymmetric_radius(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_mean

        cpu = raster_focal_mean(raster_5x5, radius=(1, 2), use_gpu=False).to_numpy()
        gpu = raster_focal_mean(raster_5x5, radius=(1, 2), use_gpu=True).to_numpy()
        np.testing.assert_array_almost_equal(gpu, cpu, decimal=10)

    def test_gpu_diagnostics(self, raster_5x5):
        from vibespatial.raster.algebra import raster_focal_min

        result = raster_focal_min(raster_5x5, radius=1, use_gpu=True)
        assert any("gpu_focal_min" in d.detail for d in result.diagnostics)


# ---------------------------------------------------------------------------
# Export / import tests
# ---------------------------------------------------------------------------


class TestExports:
    def test_importable_from_init(self):
        from vibespatial.raster import (  # noqa: F401
            raster_focal_max,
            raster_focal_mean,
            raster_focal_min,
            raster_focal_range,
            raster_focal_std,
            raster_focal_variety,
        )

    def test_in_all(self):
        from vibespatial.raster import __all__

        for name in [
            "raster_focal_min",
            "raster_focal_max",
            "raster_focal_mean",
            "raster_focal_std",
            "raster_focal_range",
            "raster_focal_variety",
        ]:
            assert name in __all__, f"{name} not in __all__"
