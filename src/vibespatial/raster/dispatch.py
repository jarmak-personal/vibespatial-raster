"""VRAM budget functions for multiband GPU processing.

Provides utilities to query available GPU memory and compute how many raster
bands can be processed in a single GPU pass without exceeding the VRAM budget.

These functions are GPU infrastructure helpers -- they have no CPU fallback
equivalents because they are only meaningful when a CUDA device is present.
When CuPy is unavailable, ``available_vram_bytes()`` returns 0 gracefully.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "available_vram_bytes",
    "max_bands_for_budget",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VRAM_HEADROOM_FRACTION = 0.15
"""Reserve 15 % of effective VRAM as headroom for driver allocations,
fragmentation, and concurrent kernel launches."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_vram_bytes() -> int:
    """Return the effective available VRAM in bytes after headroom.

    The effective available memory accounts for both genuinely free device
    memory *and* blocks held by CuPy's memory pool that can be reused without
    a new ``cudaMalloc``.  A 15 % headroom fraction is subtracted from the sum
    to leave breathing room for the CUDA driver, fragmentation, and any
    concurrent allocations.

    Returns 0 when CuPy is not importable or no CUDA device is available,
    making the function safe to call unconditionally on CPU-only machines.
    """
    try:
        import cupy as cp
    except ImportError:
        return 0

    try:
        free, _ = cp.cuda.runtime.memGetInfo()
        pool_free = cp.get_default_memory_pool().free_bytes()
        effective = free + pool_free
        usable = int(effective * (1.0 - _VRAM_HEADROOM_FRACTION))
        return max(0, usable)
    except Exception:
        # Any CUDA runtime failure (no device, driver mismatch, etc.)
        return 0


def max_bands_for_budget(
    height: int,
    width: int,
    dtype: np.dtype,
    buffers_per_band: int = 2,
    scratch_bytes: int = 0,
) -> int:
    """Compute how many raster bands fit in available VRAM.

    Parameters
    ----------
    height, width:
        Spatial dimensions of each band.
    dtype:
        NumPy dtype of the raster (e.g. ``np.float32``).  Used to determine
        per-element byte width via ``dtype.itemsize``.
    buffers_per_band:
        Number of device buffers required per band (default 2 — one input and
        one output buffer).
    scratch_bytes:
        Additional fixed scratch memory consumed by the operation, subtracted
        from the VRAM budget before dividing by per-band cost.

    Returns
    -------
    int
        Maximum number of bands that fit, but always at least 1 so that a
        single-band fallback is always possible.
    """
    dtype = np.dtype(dtype)
    per_band = height * width * dtype.itemsize * buffers_per_band
    if per_band <= 0:
        return 1

    budget = available_vram_bytes() - scratch_bytes
    if budget <= 0:
        return 1

    return max(1, budget // per_band)
