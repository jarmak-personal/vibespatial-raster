"""Tests for raster algebra: local element-wise operations."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")


@pytest.fixture
def raster_a():
    from vibespatial.raster.buffers import from_numpy

    return from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


@pytest.fixture
def raster_b():
    from vibespatial.raster.buffers import from_numpy

    return from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


@pytest.fixture
def raster_with_nodata():
    from vibespatial.raster.buffers import from_numpy

    return from_numpy(
        np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float64),
        nodata=-9999.0,
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


class TestRasterAdd:
    def test_basic(self, raster_a, raster_b):
        from vibespatial.raster.algebra import raster_add

        result = raster_add(raster_a, raster_b)
        expected = np.array([[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_preserves_affine(self, raster_a, raster_b):
        from vibespatial.raster.algebra import raster_add

        result = raster_add(raster_a, raster_b)
        assert result.affine == raster_a.affine

    def test_shape_mismatch(self, raster_a):
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        other = from_numpy(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="shapes must match"):
            raster_add(raster_a, other)


class TestRasterSubtract:
    def test_basic(self, raster_a, raster_b):
        from vibespatial.raster.algebra import raster_subtract

        result = raster_subtract(raster_b, raster_a)
        expected = np.array([[9.0, 18.0], [27.0, 36.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestRasterMultiply:
    def test_basic(self, raster_a, raster_b):
        from vibespatial.raster.algebra import raster_multiply

        result = raster_multiply(raster_a, raster_b)
        expected = np.array([[10.0, 40.0], [90.0, 160.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestRasterDivide:
    def test_basic(self, raster_b, raster_a):
        from vibespatial.raster.algebra import raster_divide

        result = raster_divide(raster_b, raster_a)
        expected = np.array([[10.0, 10.0], [10.0, 10.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestNodataPropagation:
    def test_nodata_propagated(self, raster_with_nodata, raster_b):
        from vibespatial.raster.algebra import raster_add

        result = raster_add(raster_with_nodata, raster_b)
        data = result.to_numpy()
        assert data[0, 1] == -9999.0  # nodata propagated


class TestRasterApply:
    def test_sqrt(self, raster_a):
        import cupy as cp
        from vibespatial.raster.algebra import raster_apply

        result = raster_apply(raster_a, cp.sqrt)
        expected = np.sqrt(np.array([[1.0, 2.0], [3.0, 4.0]]))
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestRasterWhere:
    def test_basic(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import from_numpy

        cond = from_numpy(np.array([[1, 0], [0, 1]], dtype=np.float64))
        result = raster_where(cond, 10.0, 20.0)
        expected = np.array([[10.0, 20.0], [20.0, 10.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


class TestRasterClassify:
    def test_basic(self):
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import from_numpy

        data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
        raster = from_numpy(data)
        result = raster_classify(raster, bins=[1.0, 2.0, 3.0], labels=[0, 1, 2, 3])
        expected = np.array([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_label_count_mismatch(self, raster_a):
        from vibespatial.raster.algebra import raster_classify

        with pytest.raises(ValueError, match="labels must have"):
            raster_classify(raster_a, bins=[1.0, 2.0], labels=[0, 1])


# ---------------------------------------------------------------------------
# Focal operations (o17.8.5)
# ---------------------------------------------------------------------------


class TestRasterConvolve:
    def test_identity_kernel(self):
        from vibespatial.raster.algebra import raster_convolve
        from vibespatial.raster.buffers import from_numpy

        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        raster = from_numpy(data)
        # 1x1 identity kernel
        kernel = np.array([[1.0]])
        result = raster_convolve(raster, kernel)
        np.testing.assert_array_almost_equal(result.to_numpy(), data)

    def test_averaging_kernel(self):
        from vibespatial.raster.algebra import raster_convolve
        from vibespatial.raster.buffers import from_numpy

        data = np.ones((5, 5))
        raster = from_numpy(data)
        kernel = np.ones((3, 3)) / 9.0
        result = raster_convolve(raster, kernel)
        # Center pixel should still be 1.0 (average of all 1s)
        np.testing.assert_almost_equal(result.to_numpy()[2, 2], 1.0)

    def test_rejects_1d_kernel(self):
        from vibespatial.raster.algebra import raster_convolve
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="2D"):
            raster_convolve(raster, np.array([1, 2, 3]))


class TestGaussianFilter:
    def test_smoothing(self):
        from vibespatial.raster.algebra import raster_gaussian_filter
        from vibespatial.raster.buffers import from_numpy

        data = np.zeros((11, 11))
        data[5, 5] = 100.0  # impulse
        raster = from_numpy(data)
        result = raster_gaussian_filter(raster, sigma=1.0)
        smoothed = result.to_numpy()
        # Peak should be reduced
        assert smoothed[5, 5] < 100.0
        # Neighbors should have received some value
        assert smoothed[5, 6] > 0.0
        assert smoothed[4, 5] > 0.0


class TestSlope:
    def test_flat(self):
        from vibespatial.raster.algebra import raster_slope
        from vibespatial.raster.buffers import from_numpy

        data = np.ones((10, 10)) * 100.0
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))
        result = raster_slope(raster)
        # Flat surface = 0 slope everywhere (interior)
        np.testing.assert_array_almost_equal(result.to_numpy()[2:-2, 2:-2], 0.0, decimal=10)

    def test_tilted(self):
        from vibespatial.raster.algebra import raster_slope
        from vibespatial.raster.buffers import from_numpy

        # Linear ramp in y direction: each row increases by 1
        data = np.tile(np.arange(10, dtype=np.float64), (10, 1)).T
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))
        result = raster_slope(raster)
        slope = result.to_numpy()
        # Interior should have nonzero slope
        assert slope[5, 5] > 0.0


class TestAspect:
    def test_east_facing(self):
        from vibespatial.raster.algebra import raster_aspect
        from vibespatial.raster.buffers import from_numpy

        # Elevation increases to the east (columns increase)
        data = np.tile(np.arange(10, dtype=np.float64), (10, 1))
        raster = from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))
        result = raster_aspect(raster)
        aspect = result.to_numpy()
        # Interior pixels should face roughly east (90 degrees)
        center = aspect[5, 5]
        assert 45 < center < 135, f"Expected ~90, got {center}"
