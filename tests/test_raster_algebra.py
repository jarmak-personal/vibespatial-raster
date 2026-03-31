"""Tests for raster algebra: local element-wise operations."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not HAS_GPU, reason="CuPy not available"),
]


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

    def test_div_by_zero_yields_nodata(self):
        """Division by zero should produce nodata, not inf."""
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_divide(a, b)
        data = result.to_numpy()
        # div-by-zero positions should be nodata
        assert data[0, 0] == -9999.0
        assert data[1, 0] == -9999.0
        # valid positions should be correct
        np.testing.assert_almost_equal(data[0, 1], 2.0)
        np.testing.assert_almost_equal(data[1, 1], 2.0)

    def test_legitimate_result_equal_to_nodata_not_masked(self):
        """Regression test for Bug #10: a legitimate division result that
        equals the nodata sentinel must NOT be treated as nodata.

        If nodata=-9999.0 and a valid division produces -9999.0 (e.g.,
        -29997.0 / 3.0), the result pixel must be preserved, not masked.
        """
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

        # Construct rasters where a valid division produces exactly -9999.0
        # -29997.0 / 3.0 = -9999.0  (a legitimate result)
        a = from_numpy(
            np.array([[1.0, -29997.0], [6.0, -9999.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[2.0, 3.0], [3.0, 1.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_divide(a, b)
        data = result.to_numpy()

        # [0,0]: 1.0/2.0 = 0.5 — valid
        np.testing.assert_almost_equal(data[0, 0], 0.5)
        # [0,1]: -29997.0/3.0 = -9999.0 — legitimate result, NOT nodata
        # This is the key assertion: the value equals the nodata sentinel
        # but must be preserved because neither input pixel was nodata.
        np.testing.assert_almost_equal(data[0, 1], -9999.0)
        # [1,0]: 6.0/3.0 = 2.0 — valid
        np.testing.assert_almost_equal(data[1, 0], 2.0)
        # [1,1]: a[1,1] = -9999.0 is nodata — should propagate as nodata
        assert data[1, 1] == -9999.0

    def test_nodata_propagation_in_divide(self):
        """Input nodata pixels should propagate through division."""
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[10.0, -9999.0], [30.0, 40.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[2.0, 5.0], [-9999.0, 8.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_divide(a, b)
        data = result.to_numpy()
        # [0,0]: valid / valid = valid
        np.testing.assert_almost_equal(data[0, 0], 5.0)
        # [0,1]: nodata / valid = nodata
        assert data[0, 1] == -9999.0
        # [1,0]: valid / nodata = nodata
        assert data[1, 0] == -9999.0
        # [1,1]: valid / valid = valid
        np.testing.assert_almost_equal(data[1, 1], 5.0)

    def test_div_by_zero_no_nodata_sentinel(self):
        """Division by zero when neither raster has nodata should produce NaN."""
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[1.0, 2.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        )
        b = from_numpy(
            np.array([[0.0, 1.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        )
        result = raster_divide(a, b)
        data = result.to_numpy()
        # div-by-zero should become NaN (used as nodata sentinel)
        assert np.isnan(data[0, 0])
        assert result.nodata is not None and np.isnan(result.nodata)
        # valid division should be correct
        np.testing.assert_almost_equal(data[0, 1], 2.0)


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
# Multiband binary operations
# ---------------------------------------------------------------------------


class TestMultibandBinaryOps:
    """Verify multiband dispatch for binary algebra on GPU."""

    def test_both_multiband_add(self):
        """Both inputs 3-band: operate band-by-band."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a_data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        )
        b_data = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
                [[90.0, 100.0], [110.0, 120.0]],
            ]
        )
        a = from_numpy(a_data)
        b = from_numpy(b_data)
        result = raster_add(a, b)
        out = result.to_numpy()
        assert out.shape == (3, 2, 2)
        np.testing.assert_array_almost_equal(out, a_data + b_data)

    def test_multiband_broadcast_single(self):
        """3-band + single-band: broadcast single across all bands."""
        from vibespatial.raster.algebra import raster_multiply
        from vibespatial.raster.buffers import from_numpy

        a_data = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
                [[9.0, 10.0], [11.0, 12.0]],
            ]
        )
        b_data = np.array([[2.0, 3.0], [4.0, 5.0]])
        a = from_numpy(a_data)
        b = from_numpy(b_data)
        result = raster_multiply(a, b)
        out = result.to_numpy()
        assert out.shape == (3, 2, 2)
        for band in range(3):
            np.testing.assert_array_almost_equal(out[band], a_data[band] * b_data)

    def test_multiband_nodata_propagation(self):
        """Nodata in one band should propagate per-band."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a_data = np.array(
            [
                [[1.0, -9999.0], [3.0, 4.0]],
                [[5.0, 6.0], [-9999.0, 8.0]],
            ]
        )
        b_data = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        )
        a = from_numpy(a_data, nodata=-9999.0)
        b = from_numpy(b_data, nodata=-9999.0)
        result = raster_add(a, b)
        out = result.to_numpy()
        assert out.shape == (2, 2, 2)
        assert out[0, 0, 1] == -9999.0  # nodata in band 0
        assert out[1, 1, 0] == -9999.0  # nodata in band 1
        np.testing.assert_almost_equal(out[0, 0, 0], 11.0)
        np.testing.assert_almost_equal(out[1, 0, 0], 55.0)

    def test_multiband_divide_by_zero(self):
        """Division by zero in multiband should produce nodata per-band."""
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

        a_data = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        b_data = np.array([[[0.0, 1.0], [2.0, 0.0]]])
        a = from_numpy(a_data, nodata=-9999.0)
        b = from_numpy(b_data, nodata=-9999.0)
        result = raster_divide(a, b)
        out = result.to_numpy()
        assert out[0, 0, 0] == -9999.0  # div by zero
        assert out[0, 1, 1] == -9999.0  # div by zero
        np.testing.assert_almost_equal(out[0, 0, 1], 2.0)
        np.testing.assert_almost_equal(out[0, 1, 0], 1.5)

    def test_mismatched_band_counts_raises(self):
        """Two multiband rasters with different band counts should raise."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(np.random.rand(2, 3, 3))
        b = from_numpy(np.random.rand(3, 3, 3))
        with pytest.raises(ValueError, match="band counts must match"):
            raster_add(a, b)


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

    def test_sigma_zero_raises(self):
        from vibespatial.raster.algebra import raster_gaussian_filter
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(np.ones((5, 5)))
        with pytest.raises(ValueError, match="sigma must be positive"):
            raster_gaussian_filter(raster, sigma=0)

    def test_sigma_negative_raises(self):
        from vibespatial.raster.algebra import raster_gaussian_filter
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(np.ones((5, 5)))
        with pytest.raises(ValueError, match="sigma must be positive"):
            raster_gaussian_filter(raster, sigma=-1.5)


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

    def test_nodata_propagated(self):
        from vibespatial.raster.algebra import raster_slope
        from vibespatial.raster.buffers import from_numpy

        data = np.ones((10, 10)) * 100.0
        data[5, 5] = -9999.0
        raster = from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0))
        result = raster_slope(raster)
        out = result.to_numpy()
        # Handle both 2D and 3D output shapes
        pixel = out[0, 5, 5] if out.ndim == 3 else out[5, 5]
        assert pixel == pytest.approx(-9999.0)


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


# ---------------------------------------------------------------------------
# Integer dtype binary operations
# ---------------------------------------------------------------------------


class TestIntegerDtypeOps:
    """Verify compute-dtype dispatch for integer-typed rasters.

    Wide integers (int32/int64, itemsize >= 4) compute in float64.
    Narrow integers (uint8/int16, itemsize < 4) compute in float32.
    """

    def test_uint8_add(self):
        """uint8 + uint8: narrow integers compute in float32."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[10, 20], [30, 40]], dtype=np.uint8),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[1, 2], [3, 4]], dtype=np.uint8),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_add(a, b)
        expected = np.array([[11, 22], [33, 44]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_int16_subtract_nodata(self):
        """int16 - int16 with nodata sentinel: verify nodata propagation."""
        from vibespatial.raster.algebra import raster_subtract
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[100, -9999], [300, 400]], dtype=np.int16),
            nodata=-9999,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[10, 20], [-9999, 40]], dtype=np.int16),
            nodata=-9999,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_subtract(a, b)
        data = result.to_numpy()
        # [0,0]: valid - valid = valid
        np.testing.assert_almost_equal(data[0, 0], 90.0)
        # [0,1]: nodata - valid = nodata
        assert data[0, 1] == -9999.0
        # [1,0]: valid - nodata = nodata
        assert data[1, 0] == -9999.0
        # [1,1]: valid - valid = valid
        np.testing.assert_almost_equal(data[1, 1], 360.0)

    def test_int32_multiply_precision(self):
        """int32 * int32 with values > 2^24: must promote to float64.

        float32 has only 24 bits of mantissa, so 20_000_000 * 3 would
        lose precision in float32. The wide-integer path computes in
        float64 (53-bit mantissa), preserving the exact result.
        """
        from vibespatial.raster.algebra import raster_multiply
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[20_000_000, 20_000_001]], dtype=np.int32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        )
        b = from_numpy(
            np.array([[3, 7]], dtype=np.int32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        )
        result = raster_multiply(a, b)
        data = result.to_numpy()
        # Exact results: 60_000_000 and 140_000_007
        np.testing.assert_array_equal(data.ravel(), [60_000_000.0, 140_000_007.0])

    def test_mixed_int_float_add(self):
        """int16 + float32: mixed-dtype binary op should succeed."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[10, 20], [30, 40]], dtype=np.int16),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_add(a, b)
        expected = np.array([[10.5, 21.5], [32.5, 43.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)
