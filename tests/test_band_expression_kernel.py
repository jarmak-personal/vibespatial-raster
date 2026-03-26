"""Tests for band-fused expression kernel template.

CPU-only tests verify source generation without requiring a GPU.
GPU tests (marked ``requires_gpu``) compile and launch the kernel via CuPy.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.kernels.algebra import (
    band_expression_cache_key,
    generate_band_expression_kernel,
)

try:
    import cupy as cp

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")


# ---------------------------------------------------------------------------
# CPU-only tests — source generation (no GPU needed)
# ---------------------------------------------------------------------------


class TestGenerateBandExpressionKernel:
    """Verify that generate_band_expression_kernel produces correct CUDA C."""

    def test_generate_ndvi_kernel(self):
        """NDVI kernel source contains correct band offsets and expression."""
        source = generate_band_expression_kernel(
            band_refs={"nir": 3, "red": 2},
            expression="(nir - red) / (nir + red)",
            dtype="double",
            nodata_val=-9999.0,
        )
        # Must be valid NVRTC: contains the kernel entry point
        assert 'extern "C" __global__' in source
        assert "void band_expression(" in source

        # Band loads use compile-time literal offsets
        assert "data[3 * band_stride + i]" in source
        assert "data[2 * band_stride + i]" in source

        # Variable declarations with correct dtype
        assert "double nir = data[3 * band_stride + i];" in source
        assert "double red = data[2 * band_stride + i];" in source

        # Expression is embedded verbatim
        assert "(nir - red) / (nir + red)" in source

        # Nodata sentinel
        assert "-9999.0" in source

        # ILP=4 pragma
        assert "#pragma unroll" in source

    def test_generate_kernel_nodata_checks_2_bands(self):
        """Two-band kernel has exactly two nodata mask checks."""
        source = generate_band_expression_kernel(
            band_refs={"a": 0, "b": 1},
            expression="a + b",
            dtype="double",
            nodata_val=0.0,
        )
        # Each band gets a nodata_mask nullptr-guarded check
        assert "nodata_mask[0 * band_stride + i]" in source
        assert "nodata_mask[1 * band_stride + i]" in source
        # Exactly 2 checks (not more)
        assert source.count("nodata_mask != nullptr &&") == 2

    def test_generate_kernel_nodata_checks_4_bands(self):
        """Four-band kernel has exactly four nodata mask checks."""
        source = generate_band_expression_kernel(
            band_refs={"b1": 0, "b2": 1, "b3": 2, "b4": 3},
            expression="(b1 + b2 + b3 + b4) / 4.0",
            dtype="float",
            nodata_val=-1.0,
        )
        assert source.count("nodata_mask != nullptr &&") == 4
        for idx in range(4):
            assert f"nodata_mask[{idx} * band_stride + i]" in source

    def test_generate_kernel_float64(self):
        """Double dtype substitution is applied throughout the kernel."""
        source = generate_band_expression_kernel(
            band_refs={"x": 0},
            expression="x * 2.0",
            dtype="double",
            nodata_val=0.0,
        )
        assert "const double* __restrict__ data" in source
        assert "double* __restrict__ output" in source
        assert "double x = data[0 * band_stride + i];" in source
        assert "(double)(x * 2.0)" in source

    def test_generate_kernel_float32(self):
        """Float dtype substitution uses float type and f-suffixed nodata."""
        source = generate_band_expression_kernel(
            band_refs={"x": 0},
            expression="x * 2.0",
            dtype="float",
            nodata_val=-9999.0,
        )
        assert "const float* __restrict__ data" in source
        assert "float* __restrict__ output" in source
        assert "float x = data[0 * band_stride + i];" in source
        assert "(float)(x * 2.0)" in source
        # f-suffixed nodata literal for float kernels
        assert "-9999.0f" in source

    def test_generate_kernel_nodata_none_defaults_to_zero(self):
        """When nodata_val is None, the kernel uses 0.0 as the sentinel."""
        source = generate_band_expression_kernel(
            band_refs={"a": 0},
            expression="a",
            dtype="double",
            nodata_val=None,
        )
        # The nodata literal should be 0.0 (the default)
        assert "output[i] = 0.0;" in source

    def test_generate_kernel_empty_band_refs_raises(self):
        """Empty band_refs dict raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            generate_band_expression_kernel(
                band_refs={},
                expression="a",
                dtype="double",
                nodata_val=0.0,
            )

    def test_generate_kernel_invalid_dtype_raises(self):
        """Invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="dtype must be"):
            generate_band_expression_kernel(
                band_refs={"a": 0},
                expression="a",
                dtype="int",
                nodata_val=0.0,
            )

    def test_generate_kernel_grid_stride_ilp4(self):
        """Kernel uses ILP=4 grid-stride loop pattern."""
        source = generate_band_expression_kernel(
            band_refs={"a": 0},
            expression="a",
            dtype="double",
            nodata_val=0.0,
        )
        assert "base += stride * 4" in source
        assert "for (int j = 0; j < 4; j++)" in source
        assert "int i = base + j * stride;" in source

    def test_generate_kernel_inf_nan_guard(self):
        """Kernel guards against inf/nan results."""
        source = generate_band_expression_kernel(
            band_refs={"a": 0, "b": 1},
            expression="a / b",
            dtype="double",
            nodata_val=-9999.0,
        )
        assert "isinf(_result)" in source
        assert "isnan(_result)" in source


class TestBandExpressionCacheKey:
    """Verify cache key uniqueness properties."""

    def test_cache_key_varies_by_expression(self):
        """Different expressions produce different cache keys."""
        key1 = band_expression_cache_key("a + b", 2, "double")
        key2 = band_expression_cache_key("a - b", 2, "double")
        assert key1 != key2

    def test_cache_key_varies_by_dtype(self):
        """float vs double produce different cache keys."""
        key_f = band_expression_cache_key("a + b", 2, "float")
        key_d = band_expression_cache_key("a + b", 2, "double")
        assert key_f != key_d

    def test_cache_key_varies_by_band_count(self):
        """Different band counts produce different cache keys."""
        key2 = band_expression_cache_key("a + b", 2, "double")
        key3 = band_expression_cache_key("a + b", 3, "double")
        assert key2 != key3

    def test_cache_key_same_inputs_same_key(self):
        """Identical inputs produce the same cache key (deterministic)."""
        key1 = band_expression_cache_key("(a - b) / (a + b)", 2, "double")
        key2 = band_expression_cache_key("(a - b) / (a + b)", 2, "double")
        assert key1 == key2

    def test_cache_key_contains_dtype_and_band_count(self):
        """Cache key format includes dtype and band count for readability."""
        key = band_expression_cache_key("a + b", 2, "float")
        assert "float" in key
        assert "2b" in key
        assert key.startswith("band_expr_")

    def test_cache_key_contains_hash(self):
        """Cache key contains a hex hash substring."""
        key = band_expression_cache_key("a + b", 2, "double")
        # The SHA-1 hex digest is 40 chars after the last dash
        parts = key.rsplit("-", 1)
        assert len(parts) == 2
        assert len(parts[1]) == 40
        # All hex characters
        assert all(c in "0123456789abcdef" for c in parts[1])


# ---------------------------------------------------------------------------
# GPU tests — compile and launch (requires CuPy + CUDA device)
# ---------------------------------------------------------------------------


@requires_gpu
class TestBandExpressionGPU:
    """Compile and launch band-fused expression kernels on a real GPU."""

    def test_band_expression_ndvi_gpu(self):
        """Compile and launch NDVI kernel, verify output values."""
        # Construct a 2-band BSQ buffer: band 0 = red, band 1 = nir
        # Shape: (2, 4, 4) = 2 bands, 4x4 pixels
        height, width = 4, 4
        n_pixels = height * width
        band_stride = n_pixels

        red = np.linspace(0.1, 0.4, n_pixels, dtype=np.float64).reshape(1, height, width)
        nir = np.linspace(0.5, 0.9, n_pixels, dtype=np.float64).reshape(1, height, width)
        bsq = np.concatenate([red, nir], axis=0)  # (2, 4, 4)

        # Expected NDVI
        expected = ((nir[0] - red[0]) / (nir[0] + red[0])).ravel()

        # Generate kernel source
        source = generate_band_expression_kernel(
            band_refs={"nir": 1, "red": 0},
            expression="(nir - red) / (nir + red)",
            dtype="double",
            nodata_val=-9999.0,
        )

        # Transfer to device
        d_data = cp.asarray(bsq.ravel())
        d_output = cp.zeros(n_pixels, dtype=cp.float64)
        d_output_mask = cp.zeros(n_pixels, dtype=cp.uint8)

        # Compile and launch via CuPy RawKernel
        kernel = cp.RawKernel(source, "band_expression")
        block = 256
        grid = (n_pixels + block - 1) // block
        # No nodata mask: pass null pointer (0)
        kernel(
            (grid,),
            (block,),
            (
                d_data,
                np.intp(0),
                d_output,
                d_output_mask,
                np.int32(n_pixels),
                np.int32(band_stride),
            ),
        )

        result = cp.asnumpy(d_output)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

        # No nodata should be flagged
        mask = cp.asnumpy(d_output_mask)
        assert np.all(mask == 0)

    def test_band_expression_nodata_propagation_gpu(self):
        """Verify union nodata: if ANY referenced band is nodata, output is nodata."""
        height, width = 2, 4
        n_pixels = height * width
        band_stride = n_pixels

        # Band 0: red values
        red = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        # Band 1: nir values
        nir = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6], dtype=np.float64)
        bsq = np.concatenate([red, nir])  # (2 * n_pixels,)

        # Nodata mask: pixel 2 is nodata in band 0, pixel 5 is nodata in band 1
        nodata_mask = np.zeros(2 * n_pixels, dtype=np.uint8)
        nodata_mask[2] = 1  # band 0, pixel 2
        nodata_mask[band_stride + 5] = 1  # band 1, pixel 5

        nodata_val = -9999.0

        source = generate_band_expression_kernel(
            band_refs={"nir": 1, "red": 0},
            expression="(nir - red) / (nir + red)",
            dtype="double",
            nodata_val=nodata_val,
        )

        d_data = cp.asarray(bsq)
        d_nodata_mask = cp.asarray(nodata_mask)
        d_output = cp.zeros(n_pixels, dtype=cp.float64)
        d_output_mask = cp.zeros(n_pixels, dtype=cp.uint8)

        kernel = cp.RawKernel(source, "band_expression")
        block = 256
        grid = (n_pixels + block - 1) // block
        kernel(
            (grid,),
            (block,),
            (
                d_data,
                d_nodata_mask,
                d_output,
                d_output_mask,
                np.int32(n_pixels),
                np.int32(band_stride),
            ),
        )

        result = cp.asnumpy(d_output)
        mask = cp.asnumpy(d_output_mask)

        # Pixels 2 and 5 should be nodata
        assert result[2] == nodata_val
        assert result[5] == nodata_val
        assert mask[2] == 1
        assert mask[5] == 1

        # All other pixels should have valid NDVI and mask=0
        valid_indices = [i for i in range(n_pixels) if i not in (2, 5)]
        for idx in valid_indices:
            expected_ndvi = (nir[idx] - red[idx]) / (nir[idx] + red[idx])
            np.testing.assert_allclose(result[idx], expected_ndvi, rtol=1e-10)
            assert mask[idx] == 0

    def test_band_expression_float32_gpu(self):
        """Float32 kernel compiles and produces correct results."""
        height, width = 2, 2
        n_pixels = height * width
        band_stride = n_pixels

        a_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b_vals = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        bsq = np.concatenate([a_vals, b_vals])

        source = generate_band_expression_kernel(
            band_refs={"a": 0, "b": 1},
            expression="a + b",
            dtype="float",
            nodata_val=0.0,
        )

        d_data = cp.asarray(bsq)
        d_output = cp.zeros(n_pixels, dtype=cp.float32)
        d_output_mask = cp.zeros(n_pixels, dtype=cp.uint8)

        kernel = cp.RawKernel(source, "band_expression")
        block = 256
        grid = (n_pixels + block - 1) // block
        kernel(
            (grid,),
            (block,),
            (
                d_data,
                np.intp(0),
                d_output,
                d_output_mask,
                np.int32(n_pixels),
                np.int32(band_stride),
            ),
        )

        result = cp.asnumpy(d_output)
        expected = a_vals + b_vals
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_band_expression_division_by_zero_gpu(self):
        """Division by zero yields nodata via inf/nan guard."""
        n_pixels = 4
        band_stride = n_pixels
        nodata_val = -9999.0

        a_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        b_vals = np.array([0.0, 1.0, 0.0, 2.0], dtype=np.float64)
        bsq = np.concatenate([a_vals, b_vals])

        source = generate_band_expression_kernel(
            band_refs={"a": 0, "b": 1},
            expression="a / b",
            dtype="double",
            nodata_val=nodata_val,
        )

        d_data = cp.asarray(bsq)
        d_output = cp.zeros(n_pixels, dtype=cp.float64)
        d_output_mask = cp.zeros(n_pixels, dtype=cp.uint8)

        kernel = cp.RawKernel(source, "band_expression")
        block = 256
        grid = (n_pixels + block - 1) // block
        kernel(
            (grid,),
            (block,),
            (
                d_data,
                np.intp(0),
                d_output,
                d_output_mask,
                np.int32(n_pixels),
                np.int32(band_stride),
            ),
        )

        result = cp.asnumpy(d_output)
        mask = cp.asnumpy(d_output_mask)

        # Pixels 0 and 2 have b=0 -> division by zero -> nodata
        assert result[0] == nodata_val
        assert result[2] == nodata_val
        assert mask[0] == 1
        assert mask[2] == 1

        # Pixels 1 and 3 are valid
        np.testing.assert_allclose(result[1], 2.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 2.0, rtol=1e-10)
        assert mask[1] == 0
        assert mask[3] == 0
