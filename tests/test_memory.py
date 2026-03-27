"""Tests for tiered GPU memory pool manager (ADR-0040)."""

from __future__ import annotations

import os
from unittest import mock

import pytest


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _has_rmm() -> bool:
    try:
        import rmm  # noqa: F401

        return True
    except ImportError:
        return False


requires_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")
requires_rmm = pytest.mark.skipif(not _has_rmm(), reason="RMM not available")


def _reset_memory_module():
    """Reset the memory module's global state for test isolation.

    This must be called in a subprocess or with care -- resetting mid-process
    while RMM is active can lead to undefined behaviour.  For tests that
    only inspect the *configuration logic* (not actual GPU allocation),
    we reset the module-level sentinels so ``configure_memory_pool()`` runs
    its selection logic again.
    """
    import vibespatial.raster.memory as mem

    mem._configured = False
    mem._active_tier = ""
    mem._stats_adaptor = None
    mem._oom_retry_count = 0
    mem._oom_last_retry_time = 0.0


# ---------------------------------------------------------------------------
# Tier selection tests
# ---------------------------------------------------------------------------


@requires_gpu
@requires_rmm
class TestConfigureDefaultTier:
    """Verify default tier selection when RMM is available and no env vars set.

    On a GPU with >=50% VRAM free, Tier A (pool) is selected.  On a shared
    GPU with <50% free, Tier C (managed memory) is auto-selected to avoid
    OOM.  Both are valid default behaviours.
    """

    def test_configure_default_tier(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        # Ensure neither OOM-safety nor managed-memory env vars are set
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            tier = configure_memory_pool()

        # Auto-detection selects A (plenty of VRAM) or C (shared GPU)
        assert tier in ("A", "C")

    def test_configure_tier_a_when_plenty_of_vram(self):
        """When VRAM is >=50% free, Tier A is selected."""
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            # Mock available_device_memory to report >50% free
            with mock.patch(
                "rmm.mr.available_device_memory", return_value=(20_000_000_000, 24_000_000_000)
            ):
                tier = configure_memory_pool()

        assert tier == "A"

    def test_configure_tier_c_when_low_vram(self):
        """When VRAM is <50% free, Tier C is auto-selected."""
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            # Mock available_device_memory to report <50% free
            with mock.patch(
                "rmm.mr.available_device_memory", return_value=(4_000_000_000, 24_000_000_000)
            ):
                tier = configure_memory_pool()

        assert tier == "C"

    def test_idempotent(self):
        """Calling configure_memory_pool twice returns the same tier."""
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            tier1 = configure_memory_pool()
            tier2 = configure_memory_pool()

        assert tier1 == tier2


@requires_gpu
class TestConfigureFallbackWithoutRMM:
    """Verify fallback tier when RMM import fails."""

    def test_configure_fallback_without_rmm(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        # Mock RMM away so the import inside configure_memory_pool fails
        with mock.patch.dict("sys.modules", {"rmm.mr": None, "rmm.allocators.cupy": None}):
            tier = configure_memory_pool()

        assert tier == "fallback"


@requires_gpu
@requires_rmm
class TestOomSafetyTier:
    """Verify env var activates Tier B."""

    def test_oom_safety_tier(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        env = dict(os.environ)
        env["VIBESPATIAL_GPU_OOM_SAFETY"] = "1"
        env.pop("VIBESPATIAL_GPU_MANAGED_MEMORY", None)

        with mock.patch.dict(os.environ, env, clear=True):
            tier = configure_memory_pool()

        assert tier == "B"


@requires_gpu
@requires_rmm
class TestManagedMemoryTier:
    """Verify env var activates Tier C."""

    def test_managed_memory_tier(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        env = dict(os.environ)
        env["VIBESPATIAL_GPU_MANAGED_MEMORY"] = "1"
        env.pop("VIBESPATIAL_GPU_OOM_SAFETY", None)

        with mock.patch.dict(os.environ, env, clear=True):
            tier = configure_memory_pool()

        assert tier == "C"

    def test_managed_takes_precedence_over_oom_safety(self):
        """When both env vars are set, managed memory (C) wins."""
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool

        env = dict(os.environ)
        env["VIBESPATIAL_GPU_MANAGED_MEMORY"] = "1"
        env["VIBESPATIAL_GPU_OOM_SAFETY"] = "1"

        with mock.patch.dict(os.environ, env, clear=True):
            tier = configure_memory_pool()

        assert tier == "C"


# ---------------------------------------------------------------------------
# Stats and free tests
# ---------------------------------------------------------------------------


@requires_gpu
@requires_rmm
class TestMemoryPoolStats:
    """Verify stats return dict with expected keys."""

    def test_memory_pool_stats(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool, memory_pool_stats

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            configure_memory_pool()

        stats = memory_pool_stats()

        assert isinstance(stats, dict)
        assert stats["tier"] in ("A", "C")  # depends on available VRAM
        assert stats["configured"] is True
        assert "current_bytes" in stats
        assert "current_count" in stats
        assert "peak_bytes" in stats
        assert "peak_count" in stats
        assert "total_bytes" in stats
        assert "total_count" in stats

    def test_stats_before_configure(self):
        _reset_memory_module()
        from vibespatial.raster.memory import memory_pool_stats

        stats = memory_pool_stats()
        assert stats["configured"] is False
        assert stats["tier"] == ""

    def test_stats_reflect_allocation(self):
        """After a CuPy allocation, stats should show nonzero bytes."""
        _reset_memory_module()
        import cupy as cp

        from vibespatial.raster.memory import configure_memory_pool, memory_pool_stats

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            configure_memory_pool()

        # Allocate 1 MB on device through CuPy (which now uses RMM)
        _buf = cp.zeros(1024 * 1024 // 8, dtype=cp.float64)

        stats = memory_pool_stats()
        assert stats["current_bytes"] > 0
        assert stats["current_count"] >= 1

        # Clean up
        del _buf


@requires_gpu
class TestFreePoolMemory:
    """Verify free_pool_memory does not crash."""

    def test_free_pool_memory_before_configure(self):
        _reset_memory_module()
        from vibespatial.raster.memory import free_pool_memory

        # Should be a no-op, not raise
        free_pool_memory()

    def test_free_pool_memory_after_configure(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool, free_pool_memory

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            configure_memory_pool()

        # Should not raise regardless of tier
        free_pool_memory()

    def test_free_pool_memory_fallback(self):
        _reset_memory_module()
        from vibespatial.raster.memory import configure_memory_pool, free_pool_memory

        with mock.patch.dict("sys.modules", {"rmm.mr": None, "rmm.allocators.cupy": None}):
            configure_memory_pool()

        free_pool_memory()


# ---------------------------------------------------------------------------
# OOM callback unit test
# ---------------------------------------------------------------------------


class TestOomCallback:
    """Unit test the OOM callback logic (no GPU needed)."""

    def test_retry_returns_true(self):
        import vibespatial.raster.memory as mem
        from vibespatial.raster.memory import _oom_callback

        mem._oom_retry_count = 0
        mem._oom_last_retry_time = 0.0

        # First three calls should return True (retry)
        assert _oom_callback(1024) is True
        assert _oom_callback(1024) is True
        assert _oom_callback(1024) is True

    def test_retry_exhausted_returns_false(self):
        import vibespatial.raster.memory as mem
        from vibespatial.raster.memory import _oom_callback

        mem._oom_retry_count = 0
        mem._oom_last_retry_time = 0.0

        # Exhaust retries
        for _ in range(3):
            _oom_callback(1024)

        # Fourth call should return False
        assert _oom_callback(1024) is False

    def test_time_based_reset(self):
        import time

        import vibespatial.raster.memory as mem
        from vibespatial.raster.memory import _oom_callback

        mem._oom_retry_count = 3  # exhausted
        mem._oom_last_retry_time = time.monotonic() - 2.0  # >1s ago

        # Should reset and allow retry
        assert _oom_callback(1024) is True


# ---------------------------------------------------------------------------
# Integration: deferred init via _ensure_memory_pool
# ---------------------------------------------------------------------------


@requires_gpu
@requires_rmm
class TestDeferredInit:
    """Verify that the pool is configured on first H->D transfer."""

    def test_ensure_memory_pool_on_transfer(self):
        _reset_memory_module()
        import numpy as np

        from vibespatial.raster.buffers import from_numpy
        from vibespatial.raster.memory import _configured

        assert not _configured

        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIBESPATIAL_GPU_OOM_SAFETY", "VIBESPATIAL_GPU_MANAGED_MEMORY")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            raster = from_numpy(np.ones((10, 10), dtype=np.float32))
            # Trigger H->D transfer
            _ = raster.device_data()

        # Import again to get updated values
        from vibespatial.raster import memory as mem

        assert mem._configured is True
        # Auto-detection picks A (plenty of VRAM) or C (shared GPU)
        assert mem._active_tier in ("A", "C")
