"""Tests for raster_expression: fused element-wise algebra kernel."""

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
def band_nir():
    """Simulated NIR band with known values."""
    return from_numpy(
        np.array([[0.5, 0.8], [0.6, 0.9]], dtype=np.float64),
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


@pytest.fixture
def band_red():
    """Simulated red band with known values."""
    return from_numpy(
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


@pytest.fixture
def raster_with_nodata():
    """Raster with a nodata pixel."""
    return from_numpy(
        np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float64),
        nodata=-9999.0,
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


@pytest.fixture
def raster_simple():
    """Simple 2x2 raster."""
    return from_numpy(
        np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64),
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


@pytest.fixture
def raster_f32():
    """float32 raster."""
    return from_numpy(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
    )


# ---------------------------------------------------------------------------
# CPU-only tests (always run, no GPU needed)
# ---------------------------------------------------------------------------


class TestExpressionValidation:
    """Test expression parsing and validation."""

    def test_empty_expression_raises(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        with pytest.raises(ValueError, match="empty"):
            raster_expression("", use_gpu=False, a=raster_simple)

    def test_invalid_token_raises(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        with pytest.raises(ValueError, match="invalid tokens"):
            raster_expression("import os", use_gpu=False, a=raster_simple)

    def test_undefined_variable_raises(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        with pytest.raises(ValueError, match="undefined variables"):
            raster_expression("a + b", use_gpu=False, a=raster_simple)

    def test_invalid_variable_name_raises(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        with pytest.raises(ValueError, match="variable name"):
            raster_expression("x + 1", use_gpu=False, x=raster_simple)

    def test_no_rasters_raises(self):
        from vibespatial.raster.algebra import raster_expression

        with pytest.raises(ValueError, match="at least one"):
            raster_expression("a + 1", use_gpu=False)

    def test_shape_mismatch_raises(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.zeros((2, 2)))
        r2 = from_numpy(np.zeros((3, 3)))
        with pytest.raises(ValueError, match="same shape"):
            raster_expression("a + b", use_gpu=False, a=r1, b=r2)


class TestExpressionCPU:
    """Test CPU fallback path for raster_expression."""

    def test_simple_addition(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        r2 = from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64))
        result = raster_expression("a + b", use_gpu=False, a=r1, b=r2)
        expected = np.array([[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_ndvi_formula(self, band_nir, band_red):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("(a - b) / (a + b)", use_gpu=False, a=band_nir, b=band_red)
        data = result.to_numpy()
        # Manual NDVI calculation
        nir = np.array([[0.5, 0.8], [0.6, 0.9]])
        red = np.array([[0.1, 0.2], [0.3, 0.4]])
        expected = (nir - red) / (nir + red)
        np.testing.assert_array_almost_equal(data, expected)

    def test_single_raster_expression(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a * 2.0 + 1.0", use_gpu=False, a=raster_simple)
        expected = np.array([[21.0, 41.0], [61.0, 81.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_sqrt_function(self):
        from vibespatial.raster.algebra import raster_expression

        r = from_numpy(np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float64))
        result = raster_expression("sqrt(a)", use_gpu=False, a=r)
        expected = np.array([[2.0, 3.0], [4.0, 5.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_abs_function(self):
        from vibespatial.raster.algebra import raster_expression

        r = from_numpy(np.array([[-4.0, 9.0], [-16.0, 25.0]], dtype=np.float64))
        result = raster_expression("abs(a)", use_gpu=False, a=r)
        expected = np.array([[4.0, 9.0], [16.0, 25.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_min_max_functions(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 5.0], [3.0, 7.0]], dtype=np.float64))
        r2 = from_numpy(np.array([[4.0, 2.0], [6.0, 1.0]], dtype=np.float64))
        result = raster_expression("min(a, b)", use_gpu=False, a=r1, b=r2)
        expected = np.array([[1.0, 2.0], [3.0, 1.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_clamp_function(self):
        from vibespatial.raster.algebra import raster_expression

        r = from_numpy(np.array([[-1.0, 0.5], [1.5, 2.0]], dtype=np.float64))
        result = raster_expression("clamp(a, 0.0, 1.0)", use_gpu=False, a=r)
        expected = np.array([[0.0, 0.5], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_nodata_propagation(self, raster_with_nodata, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a + b", use_gpu=False, a=raster_with_nodata, b=raster_simple)
        data = result.to_numpy()
        assert result.nodata == -9999.0
        assert data[0, 1] == -9999.0  # nodata propagated from input a

    def test_division_by_zero_produces_nodata(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), nodata=-9999.0)
        r2 = from_numpy(np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64), nodata=-9999.0)
        result = raster_expression("a / b", use_gpu=False, a=r1, b=r2)
        data = result.to_numpy()
        # Division by zero at [0,0] and [1,0] should produce nodata
        assert data[0, 0] == -9999.0
        assert data[1, 0] == -9999.0
        # Valid divisions should be correct
        np.testing.assert_almost_equal(data[0, 1], 2.0)
        np.testing.assert_almost_equal(data[1, 1], 2.0)

    def test_multi_step_chain(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float64))
        r2 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        # (a - b) * (a + b) = a^2 - b^2
        result = raster_expression("(a - b) * (a + b)", use_gpu=False, a=r1, b=r2)
        expected = np.array([[3.0, 12.0], [27.0, 48.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_preserves_affine_and_crs(self, band_nir, band_red):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a + b", use_gpu=False, a=band_nir, b=band_red)
        assert result.affine == band_nir.affine

    def test_float32_output_dtype(self, raster_f32):
        from vibespatial.raster.algebra import raster_expression

        r2 = from_numpy(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_expression("a + b", use_gpu=False, a=raster_f32, b=r2)
        assert result.dtype == np.float32

    def test_pow_function(self):
        from vibespatial.raster.algebra import raster_expression

        r = from_numpy(np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64))
        result = raster_expression("pow(a, 2.0)", use_gpu=False, a=r)
        expected = np.array([[4.0, 9.0], [16.0, 25.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_three_input_expression(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        r2 = from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64))
        r3 = from_numpy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64))
        result = raster_expression("a + b * c", use_gpu=False, a=r1, b=r2, c=r3)
        expected = np.array([[6.0, 12.0], [18.0, 24.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestExpressionGPU:
    """Test GPU path for raster_expression."""

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_ndvi_formula(self, band_nir, band_red):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("(a - b) / (a + b)", use_gpu=True, a=band_nir, b=band_red)
        data = result.to_numpy()
        nir = np.array([[0.5, 0.8], [0.6, 0.9]])
        red = np.array([[0.1, 0.2], [0.3, 0.4]])
        expected = (nir - red) / (nir + red)
        np.testing.assert_array_almost_equal(data, expected, decimal=6)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_simple_addition(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        r2 = from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64))
        result = raster_expression("a + b", use_gpu=True, a=r1, b=r2)
        expected = np.array([[11.0, 22.0], [33.0, 44.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_single_raster(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a * 2.0 + 1.0", use_gpu=True, a=raster_simple)
        expected = np.array([[21.0, 41.0], [61.0, 81.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_nodata_propagation(self, raster_with_nodata, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a + b", use_gpu=True, a=raster_with_nodata, b=raster_simple)
        data = result.to_numpy()
        assert result.nodata == -9999.0
        assert data[0, 1] == -9999.0

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_division_by_zero(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64), nodata=-9999.0)
        r2 = from_numpy(np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64), nodata=-9999.0)
        result = raster_expression("a / b", use_gpu=True, a=r1, b=r2)
        data = result.to_numpy()
        assert data[0, 0] == -9999.0
        assert data[1, 0] == -9999.0
        np.testing.assert_almost_equal(data[0, 1], 2.0)
        np.testing.assert_almost_equal(data[1, 1], 2.0)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_sqrt_function(self):
        from vibespatial.raster.algebra import raster_expression

        r = from_numpy(np.array([[4.0, 9.0], [16.0, 25.0]], dtype=np.float64))
        result = raster_expression("sqrt(a)", use_gpu=True, a=r)
        expected = np.array([[2.0, 3.0], [4.0, 5.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_clamp_function(self):
        from vibespatial.raster.algebra import raster_expression

        r = from_numpy(np.array([[-1.0, 0.5], [1.5, 2.0]], dtype=np.float64))
        result = raster_expression("clamp(a, 0.0, 1.0)", use_gpu=True, a=r)
        expected = np.array([[0.0, 0.5], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_gpu_matches_cpu(self, band_nir, band_red):
        """Verify GPU produces same results as CPU for NDVI."""
        from vibespatial.raster.algebra import raster_expression

        cpu_result = raster_expression("(a - b) / (a + b)", use_gpu=False, a=band_nir, b=band_red)
        gpu_result = raster_expression("(a - b) / (a + b)", use_gpu=True, a=band_nir, b=band_red)
        np.testing.assert_array_almost_equal(
            gpu_result.to_numpy(), cpu_result.to_numpy(), decimal=6
        )

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_diagnostics_logged(self, band_nir, band_red):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a + b", use_gpu=True, a=band_nir, b=band_red)
        runtime_events = [e for e in result.diagnostics if e.kind == "runtime"]
        assert len(runtime_events) > 0
        assert "raster_expression" in runtime_events[-1].detail

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_three_inputs(self):
        from vibespatial.raster.algebra import raster_expression

        r1 = from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
        r2 = from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64))
        r3 = from_numpy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64))
        result = raster_expression("a + b * c", use_gpu=True, a=r1, b=r2, c=r3)
        expected = np.array([[6.0, 12.0], [18.0, 24.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    @pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
    def test_float32_inputs(self, raster_f32):
        from vibespatial.raster.algebra import raster_expression

        r2 = from_numpy(
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
        )
        result = raster_expression("a + b", use_gpu=True, a=raster_f32, b=r2)
        expected = np.array([[6.0, 8.0], [10.0, 12.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)
        assert result.dtype == np.float32


class TestAutoDispatch:
    """Test auto-dispatch logic."""

    def test_cpu_fallback_works(self, raster_simple):
        """Auto-dispatch should always work (CPU fallback)."""
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a * 2.0", a=raster_simple)
        assert result is not None
        expected = np.array([[20.0, 40.0], [60.0, 80.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_explicit_cpu(self, raster_simple):
        from vibespatial.raster.algebra import raster_expression

        result = raster_expression("a * 2.0", use_gpu=False, a=raster_simple)
        expected = np.array([[20.0, 40.0], [60.0, 80.0]])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)
