"""Tests for raster algebra CPU fallbacks: binary ops, apply, where, classify.

These tests do NOT require a GPU.  They exercise the numpy-based CPU
fallback paths by passing ``use_gpu=False`` to every operation.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Binary operations (add, subtract, multiply, divide) — CPU path
# ---------------------------------------------------------------------------


class TestBinaryOpCpuFallback:
    """Verify binary algebra ops work with use_gpu=False (numpy CPU path)."""

    def test_add_basic(self):
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_add(a, b, use_gpu=False)
        expected = np.array([[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_subtract_basic(self):
        from vibespatial.raster.algebra import raster_subtract
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_subtract(a, b, use_gpu=False)
        expected = np.array([[9.0, 18.0], [27.0, 36.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_multiply_basic(self):
        from vibespatial.raster.algebra import raster_multiply
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_multiply(a, b, use_gpu=False)
        expected = np.array([[10.0, 40.0], [90.0, 160.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_divide_basic(self):
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[2.0, 4.0], [5.0, 8.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_divide(a, b, use_gpu=False)
        expected = np.array([[5.0, 5.0], [6.0, 5.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_divide_by_zero_yields_nodata(self):
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
        result = raster_divide(a, b, use_gpu=False)
        data = result.to_numpy()
        # div-by-zero positions should be nodata
        assert data[0, 0] == -9999.0
        assert data[1, 0] == -9999.0
        # valid positions should be correct
        np.testing.assert_almost_equal(data[0, 1], 2.0)
        np.testing.assert_almost_equal(data[1, 1], 2.0)

    def test_divide_by_zero_no_nodata_sentinel(self):
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
        result = raster_divide(a, b, use_gpu=False)
        data = result.to_numpy()
        assert np.isnan(data[0, 0])
        assert result.nodata is not None and np.isnan(result.nodata)
        np.testing.assert_almost_equal(data[0, 1], 2.0)

    def test_nodata_propagation(self):
        """Input nodata pixels should propagate through binary ops."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(
            np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        b = from_numpy(
            np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_add(a, b, use_gpu=False)
        data = result.to_numpy()
        assert data[0, 1] == -9999.0  # nodata propagated
        np.testing.assert_almost_equal(data[0, 0], 11.0)
        np.testing.assert_almost_equal(data[1, 0], 33.0)
        np.testing.assert_almost_equal(data[1, 1], 44.0)

    def test_nodata_propagation_both_inputs(self):
        """Nodata in either input should propagate to output."""
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
        result = raster_divide(a, b, use_gpu=False)
        data = result.to_numpy()
        np.testing.assert_almost_equal(data[0, 0], 5.0)
        assert data[0, 1] == -9999.0  # nodata in a
        assert data[1, 0] == -9999.0  # nodata in b
        np.testing.assert_almost_equal(data[1, 1], 5.0)

    def test_preserves_affine_and_crs(self):
        """Result should preserve affine and CRS from the first input."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        a = from_numpy(np.ones((2, 2), dtype=np.float64), affine=affine)
        b = from_numpy(np.ones((2, 2), dtype=np.float64), affine=affine)
        result = raster_add(a, b, use_gpu=False)
        assert result.affine == affine

    def test_diagnostic_event_present(self):
        """CPU path should append a RUNTIME diagnostic event."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import RasterDiagnosticKind, from_numpy

        a = from_numpy(np.ones((2, 2), dtype=np.float64))
        b = from_numpy(np.ones((2, 2), dtype=np.float64))
        result = raster_add(a, b, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "CPU" in runtime_events[0].detail
        assert "raster_add" in runtime_events[0].detail

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(np.ones((2, 2), dtype=np.float64))
        b = from_numpy(np.ones((3, 3), dtype=np.float64))
        with pytest.raises(ValueError, match="shapes must match"):
            raster_add(a, b, use_gpu=False)

    def test_legitimate_result_equal_to_nodata_not_masked(self):
        """Regression: a valid result that equals the nodata sentinel must be preserved."""
        from vibespatial.raster.algebra import raster_divide
        from vibespatial.raster.buffers import from_numpy

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
        result = raster_divide(a, b, use_gpu=False)
        data = result.to_numpy()
        # [0,0]: 1.0/2.0 = 0.5 -- valid
        np.testing.assert_almost_equal(data[0, 0], 0.5)
        # [0,1]: -29997.0/3.0 = -9999.0 -- legitimate result, NOT nodata
        np.testing.assert_almost_equal(data[0, 1], -9999.0)
        # [1,0]: 6.0/3.0 = 2.0 -- valid
        np.testing.assert_almost_equal(data[1, 0], 2.0)
        # [1,1]: a[1,1] = -9999.0 is nodata -- should propagate as nodata
        assert data[1, 1] == -9999.0

    def test_single_pixel_raster(self):
        """Edge case: single-pixel rasters should work."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(np.array([[5.0]], dtype=np.float64))
        b = from_numpy(np.array([[3.0]], dtype=np.float64))
        result = raster_add(a, b, use_gpu=False)
        np.testing.assert_almost_equal(result.to_numpy()[0, 0], 8.0)

    def test_non_square_raster(self):
        """Edge case: non-square rasters should work."""
        from vibespatial.raster.algebra import raster_multiply
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(np.array([[1.0, 2.0, 3.0]], dtype=np.float64))
        b = from_numpy(np.array([[10.0, 20.0, 30.0]], dtype=np.float64))
        result = raster_multiply(a, b, use_gpu=False)
        expected = np.array([[10.0, 40.0, 90.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


# ---------------------------------------------------------------------------
# Multiband binary operations — CPU path
# ---------------------------------------------------------------------------


class TestMultibandBinaryOpsCpu:
    """Verify multiband dispatch for binary algebra on CPU."""

    def test_both_multiband_add(self):
        """Both inputs 3-band: operate band-by-band on CPU."""
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
        result = raster_add(a, b, use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (3, 2, 2)
        np.testing.assert_array_almost_equal(out, a_data + b_data)

    def test_multiband_broadcast_single(self):
        """3-band + single-band: broadcast single across all bands on CPU."""
        from vibespatial.raster.algebra import raster_subtract
        from vibespatial.raster.buffers import from_numpy

        a_data = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
                [[90.0, 100.0], [110.0, 120.0]],
            ]
        )
        b_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        a = from_numpy(a_data)
        b = from_numpy(b_data)
        result = raster_subtract(a, b, use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (3, 2, 2)
        for band in range(3):
            np.testing.assert_array_almost_equal(out[band], a_data[band] - b_data)

    def test_multiband_nodata_propagation_cpu(self):
        """Nodata propagation per-band on CPU."""
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
        result = raster_add(a, b, use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (2, 2, 2)
        assert out[0, 0, 1] == -9999.0  # nodata in band 0
        assert out[1, 1, 0] == -9999.0  # nodata in band 1
        np.testing.assert_almost_equal(out[0, 0, 0], 11.0)

    def test_mismatched_band_counts_raises_cpu(self):
        """Two multiband rasters with different band counts should raise."""
        from vibespatial.raster.algebra import raster_add
        from vibespatial.raster.buffers import from_numpy

        a = from_numpy(np.random.rand(2, 3, 3))
        b = from_numpy(np.random.rand(3, 3, 3))
        with pytest.raises(ValueError, match="band counts must match"):
            raster_add(a, b, use_gpu=False)


# ---------------------------------------------------------------------------
# raster_apply — CPU path
# ---------------------------------------------------------------------------


class TestApplyCpuFallback:
    """Verify raster_apply works with use_gpu=False (numpy CPU path)."""

    def test_sqrt(self):
        from vibespatial.raster.algebra import raster_apply
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(
            np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_apply(raster, np.sqrt, use_gpu=False)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_nodata_preserved(self):
        from vibespatial.raster.algebra import raster_apply
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(
            np.array([[4.0, -9999.0], [16.0, 25.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_apply(raster, np.sqrt, use_gpu=False)
        data = result.to_numpy()
        np.testing.assert_almost_equal(data[0, 0], 2.0)
        assert data[0, 1] == -9999.0  # nodata preserved
        np.testing.assert_almost_equal(data[1, 0], 4.0)
        np.testing.assert_almost_equal(data[1, 1], 5.0)

    def test_custom_function(self):
        from vibespatial.raster.algebra import raster_apply
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_apply(raster, lambda x: x * 2 + 1, use_gpu=False)
        expected = np.array([[3.0, 5.0], [7.0, 9.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_custom_nodata_override(self):
        """Explicit nodata parameter should override the input raster's nodata."""
        from vibespatial.raster.algebra import raster_apply
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(
            np.array([[4.0, -9999.0], [16.0, 25.0]], dtype=np.float64),
            nodata=-9999.0,
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_apply(raster, np.sqrt, nodata=-1.0, use_gpu=False)
        data = result.to_numpy()
        assert result.nodata == -1.0
        assert data[0, 1] == -1.0  # nodata replaced with custom value

    def test_preserves_affine(self):
        from vibespatial.raster.algebra import raster_apply
        from vibespatial.raster.buffers import from_numpy

        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        raster = from_numpy(np.ones((2, 2), dtype=np.float64), affine=affine)
        result = raster_apply(raster, np.sqrt, use_gpu=False)
        assert result.affine == affine

    def test_diagnostic_event_present(self):
        from vibespatial.raster.algebra import raster_apply
        from vibespatial.raster.buffers import RasterDiagnosticKind, from_numpy

        raster = from_numpy(np.ones((2, 2), dtype=np.float64))
        result = raster_apply(raster, np.sqrt, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "CPU" in runtime_events[0].detail
        assert "raster_apply" in runtime_events[0].detail


# ---------------------------------------------------------------------------
# raster_where — CPU path
# ---------------------------------------------------------------------------


class TestWhereCpuFallback:
    """Verify raster_where works with use_gpu=False (numpy CPU path)."""

    def test_basic_scalars(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import from_numpy

        cond = from_numpy(np.array([[1, 0], [0, 1]], dtype=np.float64))
        result = raster_where(cond, 10.0, 20.0, use_gpu=False)
        expected = np.array([[10.0, 20.0], [20.0, 10.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_raster_true_val(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import from_numpy

        cond = from_numpy(np.array([[1, 0], [0, 1]], dtype=np.float64))
        tv = from_numpy(np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64))
        result = raster_where(cond, tv, 0.0, use_gpu=False)
        expected = np.array([[100.0, 0.0], [0.0, 400.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_both_raster_vals(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import from_numpy

        cond = from_numpy(np.array([[1, 0], [0, 1]], dtype=np.float64))
        tv = from_numpy(np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64))
        fv = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        result = raster_where(cond, tv, fv, use_gpu=False)
        expected = np.array([[100.0, 2.0], [3.0, 400.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_nodata_in_condition(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import from_numpy

        cond = from_numpy(
            np.array([[1, -9999.0], [0, 1]], dtype=np.float64),
            nodata=-9999.0,
        )
        result = raster_where(cond, 10.0, 20.0, use_gpu=False)
        data = result.to_numpy()
        np.testing.assert_almost_equal(data[0, 0], 10.0)
        assert data[0, 1] == -9999.0  # nodata propagated from condition
        np.testing.assert_almost_equal(data[1, 0], 20.0)
        np.testing.assert_almost_equal(data[1, 1], 10.0)

    def test_preserves_affine(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import from_numpy

        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        cond = from_numpy(np.ones((2, 2), dtype=np.float64), affine=affine)
        result = raster_where(cond, 1.0, 0.0, use_gpu=False)
        assert result.affine == affine

    def test_diagnostic_event_present(self):
        from vibespatial.raster.algebra import raster_where
        from vibespatial.raster.buffers import RasterDiagnosticKind, from_numpy

        cond = from_numpy(np.ones((2, 2), dtype=np.float64))
        result = raster_where(cond, 1.0, 0.0, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "CPU" in runtime_events[0].detail
        assert "raster_where" in runtime_events[0].detail


# ---------------------------------------------------------------------------
# raster_classify — CPU path
# ---------------------------------------------------------------------------


class TestClassifyCpuFallback:
    """Verify raster_classify works with use_gpu=False (numpy CPU path)."""

    def test_basic(self):
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import from_numpy

        data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
        raster = from_numpy(data)
        result = raster_classify(raster, bins=[1.0, 2.0, 3.0], labels=[0, 1, 2, 3], use_gpu=False)
        expected = np.array([[0.0, 1.0], [2.0, 3.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_nodata_preserved(self):
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import from_numpy

        data = np.array([[0.5, -9999.0], [2.5, 3.5]], dtype=np.float64)
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_classify(raster, bins=[1.0, 2.0, 3.0], labels=[0, 1, 2, 3], use_gpu=False)
        out = result.to_numpy()
        np.testing.assert_almost_equal(out[0, 0], 0.0)
        assert out[0, 1] == -9999.0  # nodata preserved
        np.testing.assert_almost_equal(out[1, 0], 2.0)
        np.testing.assert_almost_equal(out[1, 1], 3.0)

    def test_label_count_mismatch_raises(self):
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import from_numpy

        raster = from_numpy(np.ones((2, 2), dtype=np.float64))
        with pytest.raises(ValueError, match="labels must have"):
            raster_classify(raster, bins=[1.0, 2.0], labels=[0, 1], use_gpu=False)

    def test_preserves_affine(self):
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import from_numpy

        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
        raster = from_numpy(data, affine=affine)
        result = raster_classify(raster, bins=[1.0, 2.0, 3.0], labels=[0, 1, 2, 3], use_gpu=False)
        assert result.affine == affine

    def test_diagnostic_event_present(self):
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import RasterDiagnosticKind, from_numpy

        data = np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float64)
        raster = from_numpy(data)
        result = raster_classify(raster, bins=[1.0, 2.0, 3.0], labels=[0, 1, 2, 3], use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "CPU" in runtime_events[0].detail
        assert "raster_classify" in runtime_events[0].detail

    def test_all_nodata_raster(self):
        """Edge case: all-nodata raster should remain all-nodata."""
        from vibespatial.raster.algebra import raster_classify
        from vibespatial.raster.buffers import from_numpy

        data = np.full((2, 2), -9999.0, dtype=np.float64)
        raster = from_numpy(data, nodata=-9999.0)
        result = raster_classify(raster, bins=[1.0, 2.0, 3.0], labels=[0, 1, 2, 3], use_gpu=False)
        out = result.to_numpy()
        assert np.all(out == -9999.0)


# ---------------------------------------------------------------------------
# Integer dtype binary operations — CPU path
# ---------------------------------------------------------------------------


class TestIntegerDtypeCpuFallback:
    """Verify binary algebra ops with integer-typed rasters on CPU path.

    The compute-dtype dispatch promotes:
      - wide integers (int32/int64, itemsize >= 4) -> float64
      - narrow integers (uint8/int16, itemsize < 4)  -> float32
    These tests exercise that logic via use_gpu=False.
    """

    def test_uint8_add(self):
        """Two uint8 rasters: addition should produce correct results."""
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
        result = raster_add(a, b, use_gpu=False)
        expected = np.array([[11, 22], [33, 44]])
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_int16_subtract_nodata(self):
        """Two int16 rasters with nodata=-9999: subtraction and nodata propagation."""
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
        result = raster_subtract(a, b, use_gpu=False)
        data = result.to_numpy()
        # [0,0]: 100 - 10 = 90
        np.testing.assert_almost_equal(data[0, 0], 90)
        # [0,1]: a is nodata -> propagate
        assert data[0, 1] == -9999
        # [1,0]: b is nodata -> propagate
        assert data[1, 0] == -9999
        # [1,1]: 400 - 40 = 360
        np.testing.assert_almost_equal(data[1, 1], 360)

    def test_int32_multiply_precision(self):
        """Two int32 rasters with values > 2**24: float64 promotion preserves precision."""
        from vibespatial.raster.algebra import raster_multiply
        from vibespatial.raster.buffers import from_numpy

        # 20_000_000 * 3 = 60_000_000.  In float32, 20_000_000 is exact
        # but the product 60_000_001 would lose precision.  Using a value
        # that distinguishes float32 vs float64: 20_000_003 * 3 = 60_000_009.
        # float32 can only represent integers exactly up to 2**24 = 16_777_216,
        # so 60_000_009 would be rounded in float32 but exact in float64.
        a = from_numpy(
            np.array([[20_000_003]], dtype=np.int32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        )
        b = from_numpy(
            np.array([[3]], dtype=np.int32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        )
        result = raster_multiply(a, b, use_gpu=False)
        data = result.to_numpy()
        # int32 promotion -> float64 means exact integer arithmetic is preserved
        assert data[0, 0] == 60_000_009

    def test_mixed_int_float_add(self):
        """One int16 raster + one float32 raster: addition should work."""
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
        result = raster_add(a, b, use_gpu=False)
        expected = np.array([[10.5, 21.5], [32.5, 43.5]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)
