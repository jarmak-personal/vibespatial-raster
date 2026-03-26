"""Tests for VRAM budget functions in vibespatial.raster.dispatch."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from vibespatial.raster.dispatch import (
    _VRAM_HEADROOM_FRACTION,
    available_vram_bytes,
    max_bands_for_budget,
)

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# ---------------------------------------------------------------------------
# available_vram_bytes
# ---------------------------------------------------------------------------


class TestAvailableVramBytes:
    def test_returns_int(self):
        result = available_vram_bytes()
        assert isinstance(result, int)

    def test_returns_non_negative(self):
        assert available_vram_bytes() >= 0

    def test_returns_zero_when_cupy_unavailable(self):
        """Simulate CuPy not installed by making the import raise."""
        with patch.dict("sys.modules", {"cupy": None}):
            # Force re-import failure inside the function
            with patch(
                "builtins.__import__",
                side_effect=_make_import_blocker("cupy"),
            ):
                assert available_vram_bytes() == 0

    @gpu
    def test_returns_positive_with_gpu(self):
        result = available_vram_bytes()
        assert result > 0, "Expected positive VRAM on a machine with a GPU"

    @gpu
    def test_headroom_applied(self):
        """Verify the returned value is strictly less than raw free + pool."""
        import cupy as cp

        free, _ = cp.cuda.runtime.memGetInfo()
        pool_free = cp.get_default_memory_pool().free_bytes()
        raw_total = free + pool_free
        result = available_vram_bytes()
        # Result should be at most (1 - headroom) * raw_total
        assert result <= int(raw_total * (1.0 - _VRAM_HEADROOM_FRACTION))


# ---------------------------------------------------------------------------
# max_bands_for_budget — deterministic tests via mocking
# ---------------------------------------------------------------------------


class TestMaxBandsForBudget:
    """Use mocked available_vram_bytes to get deterministic answers."""

    def test_basic_float32(self):
        # 1000x1000 float32, 2 buffers => 8 MB per band
        # 80 MB budget => 10 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32)
            assert result == 10

    def test_basic_uint8(self):
        # 1000x1000 uint8, 2 buffers => 2 MB per band
        # 80 MB budget => 40 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.uint8)
            assert result == 40

    def test_basic_float64(self):
        # 1000x1000 float64, 2 buffers => 16 MB per band
        # 80 MB budget => 5 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float64)
            assert result == 5

    def test_scratch_bytes_subtracted(self):
        # 80 MB total, 30 MB scratch => 50 MB usable
        # 1000x1000 float32 x 2 buffers = 8 MB/band => 6 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32, scratch_bytes=30_000_000)
            assert result == 6

    def test_scratch_exceeds_available_returns_one(self):
        """When scratch_bytes > available VRAM, must still return >= 1."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=1_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32, scratch_bytes=999_000_000)
            assert result == 1

    def test_zero_vram_returns_one(self):
        """No GPU memory at all => still return 1 for CPU fallback."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=0,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32)
            assert result == 1

    def test_custom_buffers_per_band(self):
        # 1000x1000 float32, 4 buffers => 16 MB per band
        # 80 MB budget => 5 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.float32, buffers_per_band=4)
            assert result == 5

    def test_small_raster_many_bands(self):
        # 10x10 float32, 2 buffers => 800 bytes per band
        # 80 MB budget => 100_000 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(10, 10, np.float32)
            assert result == 100_000

    def test_accepts_dtype_instance_and_type(self):
        """Both np.float32 (type) and np.dtype(np.float32) (instance) work."""
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result_type = max_bands_for_budget(1000, 1000, np.float32)
            result_inst = max_bands_for_budget(1000, 1000, np.dtype(np.float32))
            assert result_type == result_inst

    def test_int16_dtype(self):
        # 1000x1000 int16, 2 buffers => 4 MB per band
        # 80 MB budget => 20 bands
        with patch(
            "vibespatial.raster.dispatch.available_vram_bytes",
            return_value=80_000_000,
        ):
            result = max_bands_for_budget(1000, 1000, np.int16)
            assert result == 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_import_blocker(blocked_name: str):
    """Return a side_effect for builtins.__import__ that blocks *blocked_name*."""
    import builtins

    _real_import = builtins.__import__

    def _blocker(name, *args, **kwargs):
        if name == blocked_name or name.startswith(blocked_name + "."):
            raise ImportError(f"mocked: {name} is not installed")
        return _real_import(name, *args, **kwargs)

    return _blocker
