"""Tests for NVRTC disk cache integration in vibespatial.raster.

The cache implementation lives in vibespatial.cuda_runtime (parent lib) and
is tested exhaustively there (key composition, read/write, env vars, failure
handling).  These tests verify:
  1. The public API is re-exported from vibespatial.raster.
  2. Raster kernel compilation populates the disk cache (GPU integration).
"""

from __future__ import annotations

import pytest

# Guard: vibespatial >=0.1.4 needed for cache API
_has_cache_api = True
try:
    from vibespatial.cuda_runtime import (
        _disk_cache_enabled,
        _get_cache_dir,
        clear_nvrtc_cache,
        nvrtc_cache_stats,
    )
except ImportError:
    _has_cache_api = False

needs_cache_api = pytest.mark.skipif(
    not _has_cache_api,
    reason="vibespatial >=0.1.4 required (NVRTC cache API)",
)


# ---------------------------------------------------------------------------
# Re-export smoke tests
# ---------------------------------------------------------------------------


@needs_cache_api
def test_clear_nvrtc_cache_reexported():
    from vibespatial.raster import clear_nvrtc_cache as fn

    assert callable(fn)


@needs_cache_api
def test_nvrtc_cache_stats_reexported():
    from vibespatial.raster import nvrtc_cache_stats as fn

    assert callable(fn)


@needs_cache_api
def test_clear_nvrtc_cache_identity():
    """Re-export points to the same function object."""
    from vibespatial.cuda_runtime import clear_nvrtc_cache as original
    from vibespatial.raster import clear_nvrtc_cache as reexport

    assert reexport is original


@needs_cache_api
def test_nvrtc_cache_stats_identity():
    from vibespatial.cuda_runtime import nvrtc_cache_stats as original
    from vibespatial.raster import nvrtc_cache_stats as reexport

    assert reexport is original


# ---------------------------------------------------------------------------
# Public API behavior (no GPU needed)
# ---------------------------------------------------------------------------


@needs_cache_api
def test_clear_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        removed = clear_nvrtc_cache()
        assert removed == 0
    finally:
        _get_cache_dir.cache_clear()


@needs_cache_api
def test_clear_populated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        for name in ("a", "b"):
            (tmp_path / f"{name}.cubin").write_bytes(b"x" * 100)
        removed = clear_nvrtc_cache()
        assert removed == 2
        assert len(list(tmp_path.glob("*.cubin"))) == 0
    finally:
        _get_cache_dir.cache_clear()


@needs_cache_api
def test_cache_stats_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    _get_cache_dir.cache_clear()
    try:
        (tmp_path / "kern.cubin").write_bytes(b"x" * 256)
        stats = nvrtc_cache_stats()
        assert stats["directory"] == str(tmp_path)
        assert stats["file_count"] == 1
        assert stats["total_bytes"] == 256
        assert "enabled" in stats
    finally:
        _get_cache_dir.cache_clear()


# ---------------------------------------------------------------------------
# GPU integration: raster kernel compilation populates disk cache
# ---------------------------------------------------------------------------


@needs_cache_api
@pytest.mark.gpu
def test_raster_kernel_compile_populates_disk_cache(tmp_path, monkeypatch):
    """Compiling a raster NVRTC kernel writes a CUBIN to the disk cache."""
    from vibespatial.cuda_runtime import get_cuda_runtime, make_kernel_cache_key

    try:
        runtime = get_cuda_runtime()
        cc = runtime.compute_capability
    except RuntimeError:
        pytest.skip("CUDA runtime not available")
    if cc == (0, 0):
        pytest.skip("CUDA runtime not available")

    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("VIBESPATIAL_NVRTC_CACHE", "1")
    _get_cache_dir.cache_clear()
    _disk_cache_enabled.cache_clear()

    # Clear the in-memory module cache to force a fresh compile
    cache_key = make_kernel_cache_key("test_raster_cache", _TRIVIAL_KERNEL)
    runtime._module_cache.pop(cache_key, None)

    try:
        runtime.compile_kernels(
            cache_key=cache_key,
            source=_TRIVIAL_KERNEL,
            kernel_names=("test_raster_cache_kernel",),
        )
        cubin_files = [p for p in tmp_path.glob("*.cubin") if "test_raster_cache" in p.name]
        assert len(cubin_files) == 1
        assert cubin_files[0].stat().st_size > 16
    finally:
        runtime._module_cache.pop(cache_key, None)
        _get_cache_dir.cache_clear()
        _disk_cache_enabled.cache_clear()


_TRIVIAL_KERNEL = r"""
extern "C" __global__ void test_raster_cache_kernel(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = (float)idx;
}
"""
