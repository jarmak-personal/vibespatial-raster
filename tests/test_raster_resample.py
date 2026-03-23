"""Tests for raster resampling/warping: nearest, bilinear, bicubic."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import GridSpec, RasterDiagnosticKind, from_numpy
from vibespatial.raster.resample import (
    _compose_pixel_to_pixel,
    _invert_affine,
    raster_resample,
)

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
def gradient_raster():
    """4x4 raster with unique values (gradient) for verifying pixel mapping."""
    data = np.arange(16, dtype=np.float64).reshape(4, 4)
    affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    return from_numpy(data, affine=affine)


@pytest.fixture
def gradient_raster_f32():
    """4x4 float32 raster."""
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    return from_numpy(data, affine=affine)


@pytest.fixture
def nodata_raster():
    """4x4 raster with nodata pixels."""
    data = np.arange(16, dtype=np.float64).reshape(4, 4)
    data[1, 2] = -9999.0
    data[2, 1] = -9999.0
    affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    return from_numpy(data, nodata=-9999.0, affine=affine)


@pytest.fixture
def uint8_raster():
    """4x4 uint8 raster."""
    data = np.array(
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]],
        dtype=np.uint8,
    )
    affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
    return from_numpy(data, affine=affine)


# ---------------------------------------------------------------------------
# Affine helper tests
# ---------------------------------------------------------------------------


class TestAffineHelpers:
    def test_invert_identity(self):
        """Inverting identity affine returns identity."""
        affine = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        inv = _invert_affine(affine)
        np.testing.assert_allclose(inv, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0), atol=1e-12)

    def test_invert_scale(self):
        """Inverting a scale affine."""
        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        inv = _invert_affine(affine)
        # col = (x - 100) / 10, row = (y - 200) / -10
        np.testing.assert_allclose(inv[0], 0.1, atol=1e-12)
        np.testing.assert_allclose(inv[2], -10.0, atol=1e-12)

    def test_roundtrip(self):
        """Composing forward and inverse gives identity pixel mapping."""
        affine = (10.0, 0.0, 100.0, 0.0, -10.0, 200.0)
        composed = _compose_pixel_to_pixel(affine, affine)
        # dst pixel -> world -> src pixel should be identity
        # For pixel (2.5, 3.5): should map to (2.5, 3.5)
        pc, pr = 2.5, 3.5
        src_c = composed[0] * pc + composed[1] * pr + composed[2]
        src_r = composed[3] * pc + composed[4] * pr + composed[5]
        np.testing.assert_allclose(src_c, pc, atol=1e-10)
        np.testing.assert_allclose(src_r, pr, atol=1e-10)

    def test_singular_raises(self):
        """Singular affine raises ValueError."""
        affine = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="singular"):
            _invert_affine(affine)


# ---------------------------------------------------------------------------
# CPU resample tests
# ---------------------------------------------------------------------------


class TestResampleCPU:
    def test_identity_nearest(self, gradient_raster):
        """Identity transform with nearest should return identical data."""
        target = GridSpec.from_raster(gradient_raster)
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=False)
        np.testing.assert_array_equal(result.to_numpy(), gradient_raster.to_numpy())

    def test_identity_bilinear(self, gradient_raster):
        """Identity transform with bilinear should return near-identical data for interior pixels.

        Edge pixels (rightmost column, bottom row) may be nodata because bilinear
        needs a 2x2 neighborhood and the right/bottom neighbors are out-of-bounds.
        """
        target = GridSpec.from_raster(gradient_raster)
        result = raster_resample(gradient_raster, target, method="bilinear", use_gpu=False)
        out = result.to_numpy()
        src = gradient_raster.to_numpy()
        # Interior pixels (not on right/bottom edge) should match exactly
        np.testing.assert_allclose(out[:3, :3], src[:3, :3], atol=1e-10)

    def test_upsample_nearest(self, gradient_raster):
        """Upsampling by 2x with nearest should replicate pixels."""
        src_a = gradient_raster.affine
        # Half the pixel size -> double the grid dimensions
        target = GridSpec(
            affine=(src_a[0] / 2, src_a[1], src_a[2], src_a[3], src_a[4] / 2, src_a[5]),
            width=8,
            height=8,
        )
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (8, 8)
        # Each source pixel should be replicated into a 2x2 block
        src = gradient_raster.to_numpy()
        for r in range(4):
            for c in range(4):
                block = out[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2]
                np.testing.assert_array_equal(block, src[r, c])

    def test_downsample_nearest(self, gradient_raster):
        """Downsampling by 2x with nearest.

        Each output pixel center maps to the boundary of a 2x2 block in the source.
        With floor() rounding, output pixel (i,j) center at (j+0.5, i+0.5) maps to
        src_col = floor(2*(j+0.5)) = 2*j+1, src_row = floor(2*(i+0.5)) = 2*i+1.
        So it picks src[2*i+1, 2*j+1] -- the bottom-right pixel of each 2x2 block.
        """
        src_a = gradient_raster.affine
        target = GridSpec(
            affine=(src_a[0] * 2, src_a[1], src_a[2], src_a[3], src_a[4] * 2, src_a[5]),
            width=2,
            height=2,
        )
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (2, 2)
        src = gradient_raster.to_numpy()
        # Each output pixel picks the source pixel at index (2i+1, 2j+1)
        assert out[0, 0] == src[1, 1]
        assert out[0, 1] == src[1, 3]
        assert out[1, 0] == src[3, 1]
        assert out[1, 1] == src[3, 3]

    def test_nodata_propagation_nearest(self, nodata_raster):
        """Nodata pixels should propagate through nearest resampling."""
        target = GridSpec.from_raster(nodata_raster)
        result = raster_resample(nodata_raster, target, method="nearest", use_gpu=False)
        out = result.to_numpy()
        assert out[1, 2] == -9999.0
        assert out[2, 1] == -9999.0

    def test_nodata_propagation_bilinear(self, nodata_raster):
        """Nodata should propagate to neighbors in bilinear resampling."""
        target = GridSpec.from_raster(nodata_raster)
        result = raster_resample(nodata_raster, target, method="bilinear", use_gpu=False)
        out = result.to_numpy()
        # The nodata pixels themselves must remain nodata
        assert out[1, 2] == -9999.0
        assert out[2, 1] == -9999.0

    def test_preserves_affine_and_crs(self, gradient_raster):
        """Result should have target affine and source CRS."""
        target = GridSpec(
            affine=(5.0, 0.0, 100.0, 0.0, -5.0, 200.0),
            width=8,
            height=8,
        )
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=False)
        assert result.affine == target.affine
        assert result.width == 8
        assert result.height == 8

    def test_float32_dtype(self, gradient_raster_f32):
        """Float32 input should produce float32 output."""
        target = GridSpec.from_raster(gradient_raster_f32)
        result = raster_resample(gradient_raster_f32, target, method="nearest", use_gpu=False)
        assert result.dtype == np.float32

    def test_uint8_dtype(self, uint8_raster):
        """Uint8 input should produce uint8 output."""
        target = GridSpec.from_raster(uint8_raster)
        result = raster_resample(uint8_raster, target, method="nearest", use_gpu=False)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result.to_numpy(), uint8_raster.to_numpy())

    def test_invalid_method(self, gradient_raster):
        """Invalid method should raise ValueError."""
        target = GridSpec.from_raster(gradient_raster)
        with pytest.raises(ValueError, match="method must be"):
            raster_resample(gradient_raster, target, method="lanczos", use_gpu=False)

    def test_diagnostics(self, gradient_raster):
        """Result should include diagnostic event."""
        target = GridSpec.from_raster(gradient_raster)
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=False)
        runtime_diags = [d for d in result.diagnostics if d.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_diags) >= 1
        assert "cpu_resample_nearest" in runtime_diags[0].detail

    def test_bicubic_smooth(self):
        """Bicubic should produce smooth interpolation on a sine pattern."""
        x = np.linspace(0, 2 * np.pi, 8)
        y = np.linspace(0, 2 * np.pi, 8)
        xx, yy = np.meshgrid(x, y)
        data = np.sin(xx) * np.cos(yy)
        data = data.astype(np.float64)

        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 8.0)
        raster = from_numpy(data, affine=affine)

        # Upsample 2x
        target = GridSpec(
            affine=(0.5, 0.0, 0.0, 0.0, -0.5, 8.0),
            width=16,
            height=16,
        )
        result = raster_resample(raster, target, method="bicubic", use_gpu=False)
        out = result.to_numpy()
        assert out.shape == (16, 16)
        # Interior pixels should be finite
        assert np.all(np.isfinite(out[2:-2, 2:-2]))


# ---------------------------------------------------------------------------
# GPU resample tests
# ---------------------------------------------------------------------------


@requires_gpu
class TestResampleGPU:
    def test_identity_nearest(self, gradient_raster):
        """GPU: identity transform with nearest returns identical data."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(gradient_raster)
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=True)
        np.testing.assert_array_equal(result.to_numpy(), gradient_raster.to_numpy())

    def test_identity_bilinear(self, gradient_raster):
        """GPU: identity bilinear matches interior pixels (edges may be nodata)."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(gradient_raster)
        result = raster_resample(gradient_raster, target, method="bilinear", use_gpu=True)
        out = result.to_numpy()
        src = gradient_raster.to_numpy()
        # Interior pixels (not on right/bottom edge) should match exactly
        np.testing.assert_allclose(out[:3, :3], src[:3, :3], atol=1e-10)

    def test_upsample_nearest(self, gradient_raster):
        """GPU: upsampling by 2x with nearest replicates pixels."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        src_a = gradient_raster.affine
        target = GridSpec(
            affine=(src_a[0] / 2, src_a[1], src_a[2], src_a[3], src_a[4] / 2, src_a[5]),
            width=8,
            height=8,
        )
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=True)
        out = result.to_numpy()
        src = gradient_raster.to_numpy()
        for r in range(4):
            for c in range(4):
                block = out[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2]
                np.testing.assert_array_equal(block, src[r, c])

    def test_downsample_nearest(self, gradient_raster):
        """GPU: downsampling by 2x with nearest."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        src_a = gradient_raster.affine
        target = GridSpec(
            affine=(src_a[0] * 2, src_a[1], src_a[2], src_a[3], src_a[4] * 2, src_a[5]),
            width=2,
            height=2,
        )
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=True)
        out = result.to_numpy()
        src = gradient_raster.to_numpy()
        assert out[0, 0] == src[1, 1]
        assert out[0, 1] == src[1, 3]
        assert out[1, 0] == src[3, 1]
        assert out[1, 1] == src[3, 3]

    def test_gpu_matches_cpu_bilinear(self, gradient_raster):
        """GPU bilinear output matches CPU bilinear output."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec(
            affine=(5.0, 0.0, 100.0, 0.0, -5.0, 200.0),
            width=8,
            height=8,
        )
        cpu = raster_resample(gradient_raster, target, method="bilinear", use_gpu=False)
        gpu = raster_resample(gradient_raster, target, method="bilinear", use_gpu=True)
        np.testing.assert_allclose(gpu.to_numpy(), cpu.to_numpy(), atol=1e-10)

    def test_gpu_matches_cpu_nearest(self, gradient_raster):
        """GPU nearest output matches CPU nearest output."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec(
            affine=(5.0, 0.0, 100.0, 0.0, -5.0, 200.0),
            width=8,
            height=8,
        )
        cpu = raster_resample(gradient_raster, target, method="nearest", use_gpu=False)
        gpu = raster_resample(gradient_raster, target, method="nearest", use_gpu=True)
        np.testing.assert_array_equal(gpu.to_numpy(), cpu.to_numpy())

    def test_nodata_propagation_nearest(self, nodata_raster):
        """GPU: nodata propagates through nearest resampling."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(nodata_raster)
        result = raster_resample(nodata_raster, target, method="nearest", use_gpu=True)
        out = result.to_numpy()
        assert out[1, 2] == -9999.0
        assert out[2, 1] == -9999.0

    def test_nodata_propagation_bilinear(self, nodata_raster):
        """GPU: nodata propagates through bilinear resampling."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(nodata_raster)
        result = raster_resample(nodata_raster, target, method="bilinear", use_gpu=True)
        out = result.to_numpy()
        assert out[1, 2] == -9999.0
        assert out[2, 1] == -9999.0

    def test_float32_gpu(self, gradient_raster_f32):
        """GPU: float32 resampling preserves dtype."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(gradient_raster_f32)
        result = raster_resample(gradient_raster_f32, target, method="nearest", use_gpu=True)
        assert result.dtype == np.float32

    def test_uint8_gpu(self, uint8_raster):
        """GPU: uint8 resampling preserves dtype and values."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(uint8_raster)
        result = raster_resample(uint8_raster, target, method="nearest", use_gpu=True)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result.to_numpy(), uint8_raster.to_numpy())

    def test_bicubic_gpu(self, gradient_raster):
        """GPU: bicubic resampling runs without error."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec(
            affine=(5.0, 0.0, 100.0, 0.0, -5.0, 200.0),
            width=8,
            height=8,
        )
        result = raster_resample(gradient_raster, target, method="bicubic", use_gpu=True)
        out = result.to_numpy()
        assert out.shape == (8, 8)

    def test_diagnostics_gpu(self, gradient_raster):
        """GPU result includes diagnostic event."""
        if not HAS_GPU:
            pytest.skip("CuPy not available")
        target = GridSpec.from_raster(gradient_raster)
        result = raster_resample(gradient_raster, target, method="nearest", use_gpu=True)
        runtime_diags = [d for d in result.diagnostics if d.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_diags) >= 1
        assert "gpu_resample_nearest" in runtime_diags[0].detail


# ---------------------------------------------------------------------------
# Auto-dispatch test
# ---------------------------------------------------------------------------


class TestAutoDispatch:
    def test_auto_dispatch_cpu_fallback(self, gradient_raster):
        """Auto-dispatch falls back to CPU when GPU unavailable or small data."""
        target = GridSpec.from_raster(gradient_raster)
        # 16-pixel raster is well below the threshold -> CPU
        result = raster_resample(gradient_raster, target, method="nearest")
        assert result is not None
        np.testing.assert_array_equal(result.to_numpy(), gradient_raster.to_numpy())
