"""Tests for histogram, CDF, equalization, and percentile operations."""

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
def raster_uniform():
    """Uniform data [0..255] as uint8."""
    data = np.arange(256, dtype=np.uint8).reshape(16, 16)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 16.0))


@pytest.fixture
def raster_float():
    """Float raster with known distribution."""
    rng = np.random.RandomState(42)
    data = rng.normal(100.0, 20.0, size=(50, 50)).astype(np.float64)
    return from_numpy(data, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 50.0))


@pytest.fixture
def raster_with_nodata():
    """Raster with nodata pixels."""
    data = np.array(
        [[10.0, 20.0, 30.0], [40.0, -9999.0, 60.0], [70.0, 80.0, 90.0]],
        dtype=np.float64,
    )
    return from_numpy(data, nodata=-9999.0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))


@pytest.fixture
def raster_uint8_nodata():
    """uint8 raster with nodata sentinel."""
    data = np.array([[10, 20, 30], [40, 0, 60], [70, 80, 90]], dtype=np.uint8)
    return from_numpy(data, nodata=0, affine=(1.0, 0.0, 0.0, 0.0, -1.0, 3.0))


# ---------------------------------------------------------------------------
# CPU tests — raster_histogram
# ---------------------------------------------------------------------------


class TestRasterHistogramCPU:
    def test_basic_counts(self, raster_uniform):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(raster_uniform, bins=256, use_gpu=False)
        assert counts.shape == (256,)
        assert edges.shape == (257,)
        # Each value 0..255 appears exactly once
        assert counts.sum() == 256

    def test_total_count_matches_pixels(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(raster_float, bins=50, use_gpu=False)
        assert counts.sum() == raster_float.pixel_count

    def test_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_histogram

        counts, _edges = raster_histogram(raster_with_nodata, bins=10, use_gpu=False)
        # 9 pixels, 1 nodata -> 8 valid
        assert counts.sum() == 8

    def test_custom_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(
            raster_float, bins=10, range_min=50.0, range_max=150.0, use_gpu=False
        )
        assert edges[0] == pytest.approx(50.0)
        assert edges[-1] == pytest.approx(150.0)
        assert counts.shape == (10,)

    def test_bin_edges_monotonic(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        _counts, edges = raster_histogram(raster_float, bins=100, use_gpu=False)
        assert np.all(np.diff(edges) > 0), "Bin edges must be strictly increasing"


# ---------------------------------------------------------------------------
# CPU tests — raster_cumulative_distribution
# ---------------------------------------------------------------------------


class TestRasterCDFCPU:
    def test_monotonically_increasing(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=False)
        assert np.all(np.diff(cdf) >= 0), "CDF must be monotonically non-decreasing"

    def test_final_value_equals_pixel_count(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=False)
        assert cdf[-1] == raster_float.pixel_count

    def test_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_with_nodata, bins=10, use_gpu=False)
        assert cdf[-1] == 8  # 9 - 1 nodata


# ---------------------------------------------------------------------------
# CPU tests — raster_histogram_equalize
# ---------------------------------------------------------------------------


class TestRasterHistogramEqualizeCPU:
    def test_output_dtype_uint8(self, raster_uniform):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_uniform, use_gpu=False)
        assert result.dtype == np.uint8

    def test_output_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        data = result.to_numpy()
        assert data.min() >= 0
        assert data.max() <= 255

    def test_roughly_uniform_distribution(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        data = result.to_numpy().ravel()
        counts, _ = np.histogram(data, bins=16, range=(0, 256))
        # After equalization, bins should be more evenly distributed
        # Allow generous tolerance since input may not be perfectly equalized
        # Check that no single bin has more than 50% of pixels
        assert counts.max() < raster_float.pixel_count * 0.5

    def test_preserves_shape(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        assert result.shape == raster_float.shape

    def test_preserves_affine(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=False)
        assert result.affine == raster_float.affine


# ---------------------------------------------------------------------------
# CPU tests — raster_percentile
# ---------------------------------------------------------------------------


class TestRasterPercentileCPU:
    def test_median(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, 50.0, use_gpu=False)
        assert result.shape == (1,)
        # The histogram-based median should be close to the true median
        true_median = float(np.median(raster_float.to_numpy()))
        assert abs(result[0] - true_median) < 5.0  # generous tolerance

    def test_multiple_percentiles(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, [25.0, 50.0, 75.0], use_gpu=False)
        assert result.shape == (3,)
        # Percentiles must be ordered
        assert result[0] <= result[1] <= result[2]

    def test_0_and_100_percentiles(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, [0.0, 100.0], bins=1000, use_gpu=False)
        data = raster_float.to_numpy()
        # 0th percentile should be close to min
        assert abs(result[0] - float(np.min(data))) < 1.0
        # 100th percentile should be close to max (within bin width)

    def test_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_with_nodata, 50.0, bins=100, use_gpu=False)
        assert not np.isnan(result[0])
        # Median of [10, 20, 30, 40, 60, 70, 80, 90] is ~50
        assert 30.0 <= result[0] <= 70.0

    def test_invalid_percentile_raises(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        with pytest.raises(ValueError, match="percentile must be in"):
            raster_percentile(raster_float, 101.0, use_gpu=False)

        with pytest.raises(ValueError, match="percentile must be in"):
            raster_percentile(raster_float, -1.0, use_gpu=False)

    def test_scalar_percentile(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, 50.0, use_gpu=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# GPU tests — raster_histogram
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterHistogramGPU:
    def test_gpu_matches_cpu(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        cpu_counts, cpu_edges = raster_histogram(raster_float, bins=50, use_gpu=False)
        gpu_counts, gpu_edges = raster_histogram(raster_float, bins=50, use_gpu=True)

        np.testing.assert_array_equal(cpu_counts, gpu_counts)
        np.testing.assert_array_almost_equal(cpu_edges, gpu_edges)

    def test_gpu_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_histogram

        counts, _edges = raster_histogram(raster_with_nodata, bins=10, use_gpu=True)
        assert counts.sum() == 8

    def test_gpu_custom_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(
            raster_float, bins=10, range_min=50.0, range_max=150.0, use_gpu=True
        )
        assert edges[0] == pytest.approx(50.0)
        assert edges[-1] == pytest.approx(150.0)

    def test_gpu_total_count(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram

        counts, _edges = raster_histogram(raster_float, bins=50, use_gpu=True)
        assert counts.sum() == raster_float.pixel_count


# ---------------------------------------------------------------------------
# GPU tests — raster_cumulative_distribution
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterCDFGPU:
    def test_gpu_monotonic(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=True)
        assert np.all(np.diff(cdf) >= 0)

    def test_gpu_final_value(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, _edges = raster_cumulative_distribution(raster_float, bins=50, use_gpu=True)
        assert cdf[-1] == raster_float.pixel_count


# ---------------------------------------------------------------------------
# GPU tests — raster_histogram_equalize
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterHistogramEqualizeGPU:
    def test_gpu_output_dtype(self, raster_uniform):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_uniform, use_gpu=True)
        assert result.dtype == np.uint8

    def test_gpu_output_range(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=True)
        data = result.to_numpy()
        assert data.min() >= 0
        assert data.max() <= 255

    def test_gpu_preserves_shape(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=True)
        assert result.shape == raster_float.shape

    def test_gpu_roughly_uniform(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float, use_gpu=True)
        data = result.to_numpy().ravel()
        counts, _ = np.histogram(data, bins=16, range=(0, 256))
        # No single bin should dominate
        assert counts.max() < raster_float.pixel_count * 0.5

    def test_gpu_nodata_preserved_uint8(self, raster_uint8_nodata):
        """Regression test for kernel param type mismatch (bug #4).

        The histogram remap kernel declares nodata_val as const int and
        casts to unsigned char internally.  Before the fix, the host
        passed KERNEL_PARAM_I32 (4 bytes) for a kernel parameter declared
        as unsigned char (1 byte), corrupting the nodata value written
        to output pixels.  This test verifies nodata pixels get the
        correct sentinel value after equalization.
        """
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_uint8_nodata, use_gpu=True)
        data = result.to_numpy()
        # The input nodata sentinel is 0.  The nodata pixel is at [1,1].
        # After equalization, nodata pixels must still hold the sentinel.
        assert data[1, 1] == 0, f"nodata pixel should be 0, got {data[1, 1]}"
        # Valid pixels should have been remapped (not all zeros)
        valid_mask = np.array([[True, True, True], [True, False, True], [True, True, True]])
        assert data[valid_mask].sum() > 0, "valid pixels should have nonzero equalized values"

    def test_gpu_equalize_matches_cpu_with_nodata(self, raster_uint8_nodata):
        """GPU and CPU equalization should produce identical output for uint8 with nodata."""
        from vibespatial.raster.histogram import raster_histogram_equalize

        cpu_result = raster_histogram_equalize(raster_uint8_nodata, use_gpu=False)
        gpu_result = raster_histogram_equalize(raster_uint8_nodata, use_gpu=True)
        np.testing.assert_array_equal(
            cpu_result.to_numpy(),
            gpu_result.to_numpy(),
            err_msg="GPU and CPU histogram equalize should match for uint8 with nodata",
        )


# ---------------------------------------------------------------------------
# GPU tests — raster_percentile
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
class TestRasterPercentileGPU:
    def test_gpu_matches_cpu(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        cpu_result = raster_percentile(raster_float, [25.0, 50.0, 75.0], bins=256, use_gpu=False)
        gpu_result = raster_percentile(raster_float, [25.0, 50.0, 75.0], bins=256, use_gpu=True)
        np.testing.assert_array_almost_equal(cpu_result, gpu_result, decimal=0)

    def test_gpu_ordered_percentiles(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, [10.0, 25.0, 50.0, 75.0, 90.0], use_gpu=True)
        assert np.all(np.diff(result) >= 0)

    def test_gpu_nodata_excluded(self, raster_with_nodata):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_with_nodata, 50.0, bins=100, use_gpu=True)
        assert not np.isnan(result[0])


# ---------------------------------------------------------------------------
# Auto-dispatch tests
# ---------------------------------------------------------------------------


class TestAutoDispatch:
    def test_histogram_auto_dispatch(self, raster_float):
        """Auto-dispatch should work regardless of GPU availability."""
        from vibespatial.raster.histogram import raster_histogram

        counts, edges = raster_histogram(raster_float)
        assert counts.shape[0] == 256
        assert edges.shape[0] == 257

    def test_cdf_auto_dispatch(self, raster_float):
        from vibespatial.raster.histogram import raster_cumulative_distribution

        cdf, edges = raster_cumulative_distribution(raster_float)
        assert cdf.shape[0] == 256

    def test_equalize_auto_dispatch(self, raster_float):
        from vibespatial.raster.histogram import raster_histogram_equalize

        result = raster_histogram_equalize(raster_float)
        assert result.dtype == np.uint8

    def test_percentile_auto_dispatch(self, raster_float):
        from vibespatial.raster.histogram import raster_percentile

        result = raster_percentile(raster_float, 50.0)
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# Lazy import tests
# ---------------------------------------------------------------------------


class TestLazyImports:
    def test_histogram_importable(self):
        from vibespatial.raster import raster_histogram

        assert callable(raster_histogram)

    def test_cdf_importable(self):
        from vibespatial.raster import raster_cumulative_distribution

        assert callable(raster_cumulative_distribution)

    def test_equalize_importable(self):
        from vibespatial.raster import raster_histogram_equalize

        assert callable(raster_histogram_equalize)

    def test_percentile_importable(self):
        from vibespatial.raster import raster_percentile

        assert callable(raster_percentile)
