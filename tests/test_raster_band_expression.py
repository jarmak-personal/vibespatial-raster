"""Tests for band-indexed expression parsing and evaluation.

Exercises the ``raster_expression("a[N] ...")`` syntax for multiband
rasters, including pure band-indexed, mixed multiband + single-band,
backward compatibility with existing single-band expressions, and
nodata propagation through band indices.

All tests use ``use_gpu=False`` (CPU path) so they run without a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.algebra import raster_expression
from vibespatial.raster.buffers import from_numpy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AFFINE = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def _make_multiband_f32(bands: int = 4, height: int = 10, width: int = 10):
    """Create a synthetic 4-band float32 raster with distinct per-band values."""
    rng = np.random.default_rng(42)
    # Each band has values in a different range to make them distinguishable:
    #   band 0: [100, 200)  band 1: [200, 300)
    #   band 2: [300, 400)  band 3: [400, 500)
    data = np.empty((bands, height, width), dtype=np.float32)
    for b in range(bands):
        data[b] = rng.uniform(100 * (b + 1), 100 * (b + 2), (height, width)).astype(np.float32)
    return from_numpy(data, affine=_AFFINE)


def _make_multiband_with_nodata(
    nodata: float = -9999.0,
    bands: int = 4,
    height: int = 10,
    width: int = 10,
):
    """Create a multiband raster with nodata in specific positions."""
    rng = np.random.default_rng(42)
    data = np.empty((bands, height, width), dtype=np.float32)
    for b in range(bands):
        data[b] = rng.uniform(100 * (b + 1), 100 * (b + 2), (height, width)).astype(np.float32)
    # Place nodata in band 2, row 0, cols 0-2
    data[2, 0, 0:3] = nodata
    # Place nodata in band 3, row 1, col 5
    data[3, 1, 5] = nodata
    return from_numpy(data, nodata=nodata, affine=_AFFINE)


def _make_singleband_f32(height: int = 10, width: int = 10):
    """Create a synthetic single-band float32 raster."""
    rng = np.random.default_rng(99)
    data = rng.uniform(1.0, 10.0, (height, width)).astype(np.float32)
    return from_numpy(data, affine=_AFFINE)


# ---------------------------------------------------------------------------
# Band-indexed NDVI
# ---------------------------------------------------------------------------


class TestBandIndexedNDVI:
    """Compute NDVI using band indexing on a multiband raster."""

    def test_expression_band_index_ndvi(self):
        """NDVI via (a[3] - a[2]) / (a[3] + a[2]) on a 4-band raster."""
        mb = _make_multiband_f32()
        result = raster_expression("(a[3] - a[2]) / (a[3] + a[2])", a=mb, use_gpu=False)
        data = result.to_numpy()

        # Compute expected NDVI manually
        mb_data = mb.to_numpy()
        nir = mb_data[3].astype(np.float32)
        red = mb_data[2].astype(np.float32)
        expected = (nir - red) / (nir + red)

        np.testing.assert_allclose(data, expected, atol=1e-6)

        # Result should be single-band 2D
        assert data.ndim == 2
        assert data.shape == (10, 10)

        # Metadata preservation
        assert result.affine == _AFFINE
        assert result.dtype == np.float32

    def test_expression_band_index_equivalent_to_separate_rasters(self):
        """Band-indexed NDVI should match the result from separate rasters."""
        mb = _make_multiband_f32()
        mb_data = mb.to_numpy()

        # Band-indexed approach
        result_band = raster_expression("(a[3] - a[2]) / (a[3] + a[2])", a=mb, use_gpu=False)

        # Separate single-band approach
        nir_raster = from_numpy(mb_data[3], affine=_AFFINE)
        red_raster = from_numpy(mb_data[2], affine=_AFFINE)
        result_sep = raster_expression(
            "(a - b) / (a + b)", a=nir_raster, b=red_raster, use_gpu=False
        )

        np.testing.assert_allclose(result_band.to_numpy(), result_sep.to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Out-of-range band index
# ---------------------------------------------------------------------------


class TestBandIndexOutOfRange:
    """Band index validation errors."""

    def test_expression_band_index_out_of_range(self):
        """a[10] on a 4-band raster should raise IndexError."""
        mb = _make_multiband_f32(bands=4)
        with pytest.raises(IndexError, match=r"a\[10\] is out of range"):
            raster_expression("a[10] + 1", a=mb, use_gpu=False)

    def test_expression_band_index_exactly_at_bound(self):
        """a[4] on a 4-band raster (0-indexed) should raise IndexError."""
        mb = _make_multiband_f32(bands=4)
        with pytest.raises(IndexError, match=r"a\[4\] is out of range"):
            raster_expression("a[4] + 1", a=mb, use_gpu=False)

    def test_expression_band_index_max_valid(self):
        """a[3] on a 4-band raster should succeed."""
        mb = _make_multiband_f32(bands=4)
        result = raster_expression("a[3] + 1.0", a=mb, use_gpu=False)
        mb_data = mb.to_numpy()
        expected = mb_data[3] + 1.0
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_expression_band_index_undefined_variable(self):
        """Band index on an undefined variable should raise ValueError."""
        mb = _make_multiband_f32()
        with pytest.raises(ValueError, match=r"band-indexed variable 'b' is not defined"):
            raster_expression("b[0] + 1", a=mb, use_gpu=False)


# ---------------------------------------------------------------------------
# Mixed multiband + single-band
# ---------------------------------------------------------------------------


class TestMixedMultibandSingleband:
    """Expressions mixing band-indexed multiband and standalone single-band."""

    def test_expression_mixed_multiband_singleband(self):
        """a[0] * b + a[1] should work with multiband a and single-band b."""
        mb = _make_multiband_f32()
        sb = _make_singleband_f32()
        result = raster_expression("a[0] * b + a[1]", a=mb, b=sb, use_gpu=False)
        data = result.to_numpy()

        mb_data = mb.to_numpy()
        sb_data = sb.to_numpy()
        expected = mb_data[0] * sb_data + mb_data[1]

        np.testing.assert_allclose(data, expected, atol=1e-4)
        assert data.ndim == 2

    def test_expression_mixed_subtraction(self):
        """a[2] - b should work."""
        mb = _make_multiband_f32()
        sb = _make_singleband_f32()
        result = raster_expression("a[2] - b", a=mb, b=sb, use_gpu=False)

        mb_data = mb.to_numpy()
        sb_data = sb.to_numpy()
        expected = mb_data[2] - sb_data

        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Existing single-band expressions must keep working unchanged."""

    def test_expression_backward_compat_simple(self):
        """Classic (a - b) / (a + b) with separate single-band rasters."""
        a = from_numpy(
            np.array([[4.0, 8.0], [12.0, 16.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        b = from_numpy(
            np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        result = raster_expression("(a - b) / (a + b)", a=a, b=b, use_gpu=False)
        expected = np.array(
            [[(4 - 2) / (4 + 2), (8 - 4) / (8 + 4)], [(12 - 6) / (12 + 6), (16 - 8) / (16 + 8)]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_expression_backward_compat_single_var(self):
        """a * 2.0 with a single raster."""
        a = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        result = raster_expression("a * 2.0", a=a, use_gpu=False)
        expected = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_expression_backward_compat_with_functions(self):
        """sqrt(a) + abs(b) with single-band rasters."""
        a = from_numpy(
            np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        b = from_numpy(
            np.array([[-1.0, -2.0], [-3.0, -4.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        result = raster_expression("sqrt(a) + abs(b)", a=a, b=b, use_gpu=False)
        expected = np.sqrt(a.to_numpy()) + np.abs(b.to_numpy())
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Nodata propagation
# ---------------------------------------------------------------------------


class TestBandIndexNodata:
    """Nodata in one band must propagate correctly through band expressions."""

    def test_expression_band_index_nodata_propagation(self):
        """Nodata in band 2 should produce nodata in NDVI output."""
        nodata = -9999.0
        mb = _make_multiband_with_nodata(nodata=nodata)
        result = raster_expression("(a[3] - a[2]) / (a[3] + a[2])", a=mb, use_gpu=False)
        data = result.to_numpy()

        # Band 2 has nodata at [0, 0:3] => those output pixels should be nodata
        assert result.nodata == nodata
        np.testing.assert_equal(data[0, 0], nodata)
        np.testing.assert_equal(data[0, 1], nodata)
        np.testing.assert_equal(data[0, 2], nodata)

        # Band 3 has nodata at [1, 5] => that output pixel should be nodata
        np.testing.assert_equal(data[1, 5], nodata)

        # Other pixels should be valid NDVI values
        mb_data = mb.to_numpy()
        # Check a pixel that should be valid: [5, 5]
        nir = mb_data[3, 5, 5]
        red = mb_data[2, 5, 5]
        expected_ndvi = (nir - red) / (nir + red)
        np.testing.assert_allclose(data[5, 5], expected_ndvi, atol=1e-6)

    def test_expression_band_index_nodata_single_band_referenced(self):
        """a[0] + 1.0 where band 0 has no nodata should produce no nodata."""
        nodata = -9999.0
        mb = _make_multiband_with_nodata(nodata=nodata)
        result = raster_expression("a[0] + 1.0", a=mb, use_gpu=False)
        data = result.to_numpy()

        # Band 0 has no explicit nodata placement, but the nodata sentinel
        # applies to ALL bands.  Since the raw data for band 0 was generated
        # from random values, none should collide with -9999.0.
        mb_data = mb.to_numpy()
        expected = mb_data[0] + 1.0
        np.testing.assert_allclose(data, expected, atol=1e-6)

    def test_expression_no_nodata_raster(self):
        """Band expression on a raster with nodata=None should work."""
        mb = _make_multiband_f32()
        assert mb.nodata is None
        result = raster_expression("a[0] + a[1]", a=mb, use_gpu=False)
        mb_data = mb.to_numpy()
        expected = mb_data[0] + mb_data[1]
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)
        assert result.nodata is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBandExpressionEdgeCases:
    """Edge cases for band-indexed expressions."""

    def test_single_band_with_band_index_zero(self):
        """a[0] on a single-band raster should work."""
        sb = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        result = raster_expression("a[0] * 2.0", a=sb, use_gpu=False)
        expected = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_single_band_with_band_index_one_raises(self):
        """a[1] on a single-band raster should raise IndexError."""
        sb = from_numpy(
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            affine=_AFFINE,
        )
        with pytest.raises(IndexError, match=r"a\[1\] is out of range"):
            raster_expression("a[1] + 1", a=sb, use_gpu=False)

    def test_non_square_raster(self):
        """Band expression on a non-square raster (5x20)."""
        data = np.random.default_rng(42).uniform(0, 100, (3, 5, 20)).astype(np.float32)
        mb = from_numpy(data, affine=_AFFINE)
        result = raster_expression("a[0] + a[2]", a=mb, use_gpu=False)
        expected = data[0] + data[2]
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)
        assert result.to_numpy().shape == (5, 20)

    def test_single_pixel_raster(self):
        """Band expression on a 1x1 raster."""
        data = np.array([[[10.0]], [[20.0]], [[30.0]]], dtype=np.float32)
        mb = from_numpy(data, affine=_AFFINE)
        result = raster_expression("a[0] + a[1] + a[2]", a=mb, use_gpu=False)
        expected = np.array([[60.0]], dtype=np.float32)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_same_band_referenced_twice(self):
        """a[1] * a[1] should square band 1."""
        data = np.array([[[2.0, 3.0]], [[4.0, 5.0]]], dtype=np.float32)
        mb = from_numpy(data, affine=_AFFINE)
        result = raster_expression("a[1] * a[1]", a=mb, use_gpu=False)
        expected = np.array([[16.0, 25.0]], dtype=np.float32)
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)

    def test_dtype_float64_output(self):
        """Float64 input should produce float64 output."""
        data = np.random.default_rng(42).uniform(0, 100, (3, 5, 5)).astype(np.float64)
        mb = from_numpy(data, affine=_AFFINE)
        result = raster_expression("a[0] - a[2]", a=mb, use_gpu=False)
        assert result.dtype == np.float64
        expected = data[0] - data[2]
        np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Diagnostic event
# ---------------------------------------------------------------------------


class TestBandExpressionDiagnostics:
    """Every band expression path must emit a RasterDiagnosticEvent."""

    def test_diagnostic_event_cpu_band_expression(self):
        """CPU band expression emits a RUNTIME diagnostic."""
        from vibespatial.raster.buffers import RasterDiagnosticKind

        mb = _make_multiband_f32()
        result = raster_expression("(a[3] - a[2]) / (a[3] + a[2])", a=mb, use_gpu=False)
        runtime_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        detail = runtime_events[0].detail
        assert "band_expression" in detail
        assert "CPU" in detail
