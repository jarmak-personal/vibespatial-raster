"""VRAM budget functions and band dispatch executors for multiband GPU processing.

Provides utilities to query available GPU memory, compute how many raster
bands can be processed in a single GPU pass, and dispatch per-band operations
across multiband rasters on both GPU and CPU paths.

When CuPy is unavailable, ``available_vram_bytes()`` returns 0 gracefully and
``dispatch_per_band_gpu`` will fail with a clear error at call time.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vibespatial.raster.buffers import OwnedRasterArray

__all__ = [
    "available_vram_bytes",
    "max_bands_for_budget",
    "dispatch_per_band_gpu",
    "dispatch_per_band_cpu",
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

    When RMM is the active allocator (tiers A/B/C), the function queries
    ``rmm.mr.available_device_memory`` which accounts for pool-managed
    blocks.  Otherwise it falls back to the CuPy pool query.

    A 15 % headroom fraction is subtracted from the effective free memory
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
        # Check if RMM is managing the pool (tiers A/B/C).
        from vibespatial.raster.memory import _active_tier, _configured

        if _configured and _active_tier in ("A", "B", "C"):
            import rmm.mr

            free, _total = rmm.mr.available_device_memory()
            usable = int(free * (1.0 - _VRAM_HEADROOM_FRACTION))
            return max(0, usable)

        # Fallback: CuPy pool query (original logic).
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _single_band_view_gpu(
    raster: OwnedRasterArray,
    band_index: int,
) -> OwnedRasterArray:
    """Create a single-band OwnedRasterArray sharing device memory (zero-copy).

    The caller is responsible for ensuring the full raster is already on-device
    before calling this helper. The returned raster wraps a 2D CuPy slice of
    the band -- no new H->D transfer is triggered.

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster that is already DEVICE-resident.
    band_index : int
        0-indexed band to extract.

    Returns
    -------
    OwnedRasterArray
        A single-band raster with ``band_count == 1`` and ``ndim == 2``,
        sharing the device buffer of *raster*.
    """
    from vibespatial.raster.buffers import (
        OwnedRasterArray as _ORA,
    )
    from vibespatial.raster.buffers import (
        RasterDeviceState,
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
    )
    from vibespatial.residency import Residency

    # Zero-copy slice on device -- no H->D transfer
    band_device = raster.device_band(band_index)  # 2D CuPy view

    # Construct a lightweight host placeholder (never read -- device is authoritative)
    host_placeholder = np.empty(
        (band_device.shape[0], band_device.shape[1]),
        dtype=raster.dtype,
    )

    return _ORA(
        data=host_placeholder,
        nodata=raster.nodata,
        dtype=raster.dtype,
        affine=raster.affine,
        crs=raster.crs,
        residency=Residency.DEVICE,
        device_state=RasterDeviceState(data=band_device),
        _host_materialized=False,
        diagnostics=[
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.CREATED,
                detail=f"_single_band_view_gpu band={band_index}",
                residency=Residency.DEVICE,
            )
        ],
    )


def _single_band_view_cpu(
    raster: OwnedRasterArray,
    band_index: int,
) -> OwnedRasterArray:
    """Create a single-band OwnedRasterArray from host data.

    Parameters
    ----------
    raster : OwnedRasterArray
        Multiband raster with host-resident data.
    band_index : int
        0-indexed band to extract.

    Returns
    -------
    OwnedRasterArray
        A single-band raster with ``band_count == 1`` and ``ndim == 2``.
    """
    from vibespatial.raster.buffers import from_numpy

    host = raster.to_numpy()
    if host.ndim == 2:
        if band_index != 0:
            raise IndexError(f"single-band raster, got band_index={band_index}")
        band_data = host
    else:
        if band_index < 0 or band_index >= host.shape[0]:
            raise IndexError(
                f"band_index={band_index} out of range for {host.shape[0]}-band raster"
            )
        band_data = host[band_index]

    return from_numpy(
        band_data,
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )


# ---------------------------------------------------------------------------
# Band dispatch executors
# ---------------------------------------------------------------------------


def dispatch_per_band_gpu(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
    *,
    buffers_per_band: int = 2,
    scratch_bytes: int = 0,
) -> OwnedRasterArray:
    """Apply *op_fn* to each band of *raster* on the GPU, then reassemble.

    For single-band rasters this is a zero-overhead passthrough: *op_fn* is
    called once and its result is returned directly.

    For multiband rasters the full raster is transferred to device once, then
    each band is sliced as a zero-copy 2D view and passed to *op_fn*.  The
    per-band results are assembled via
    :meth:`OwnedRasterArray.from_band_stack`.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single- or multi-band).
    op_fn : Callable[[OwnedRasterArray], OwnedRasterArray]
        Operation to apply per band.  Receives a single-band
        ``OwnedRasterArray`` and must return a single-band
        ``OwnedRasterArray``.
    buffers_per_band : int
        Number of device buffers the operation needs per band (used by
        ``max_bands_for_budget`` for callers that want to pre-plan chunking;
        not consumed directly by this executor).
    scratch_bytes : int
        Fixed scratch memory consumed by the operation (same caveat as
        *buffers_per_band*).

    Returns
    -------
    OwnedRasterArray
        Result raster with the same band count, affine, CRS, and nodata as
        the input (metadata propagation is handled by *from_band_stack*).
    """
    from vibespatial.raster.buffers import (
        OwnedRasterArray as _ORA,
    )
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
    )
    from vibespatial.residency import Residency, TransferTrigger

    t0 = time.perf_counter()

    # -- Single-band fast path: zero overhead --
    if raster.band_count == 1:
        result = op_fn(raster)
        elapsed = time.perf_counter() - t0
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=(f"dispatch_per_band_gpu single-band passthrough elapsed={elapsed:.4f}s"),
                residency=result.residency,
            )
        )
        return result

    # -- Multiband: transfer once, iterate bands --
    raster.move_to(
        Residency.DEVICE,
        trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason="dispatch_per_band_gpu: transfer full multiband raster to device",
    )

    band_results: list[_ORA] = []
    for band_idx in range(raster.band_count):
        band_view = _single_band_view_gpu(raster, band_idx)
        band_result = op_fn(band_view)
        band_results.append(band_result)

    result = _ORA.from_band_stack(band_results, source=raster)
    elapsed = time.perf_counter() - t0
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_per_band_gpu bands={raster.band_count} "
                f"shape=({raster.band_count},{raster.height},{raster.width}) "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=result.residency,
        )
    )
    return result


def dispatch_per_band_cpu(
    raster: OwnedRasterArray,
    op_fn: Callable[[OwnedRasterArray], OwnedRasterArray],
) -> OwnedRasterArray:
    """Apply *op_fn* to each band of *raster* on the CPU, then reassemble.

    For single-band rasters this is a zero-overhead passthrough.

    For multiband rasters each band is sliced from the host numpy array,
    wrapped as a single-band ``OwnedRasterArray``, passed to *op_fn*, and
    the per-band results are assembled via
    :meth:`OwnedRasterArray.from_band_stack`.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster (single- or multi-band).
    op_fn : Callable[[OwnedRasterArray], OwnedRasterArray]
        Operation to apply per band.  Receives a single-band
        ``OwnedRasterArray`` and must return a single-band
        ``OwnedRasterArray``.

    Returns
    -------
    OwnedRasterArray
        Result raster with the same band count, affine, CRS, and nodata as
        the input.
    """
    from vibespatial.raster.buffers import (
        OwnedRasterArray as _ORA,
    )
    from vibespatial.raster.buffers import (
        RasterDiagnosticEvent,
        RasterDiagnosticKind,
    )

    t0 = time.perf_counter()

    # -- Single-band fast path: zero overhead --
    if raster.band_count == 1:
        result = op_fn(raster)
        elapsed = time.perf_counter() - t0
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=(f"dispatch_per_band_cpu single-band passthrough elapsed={elapsed:.4f}s"),
                residency=result.residency,
            )
        )
        return result

    # -- Multiband: iterate bands on host --
    band_results: list[_ORA] = []
    for band_idx in range(raster.band_count):
        band_view = _single_band_view_cpu(raster, band_idx)
        band_result = op_fn(band_view)
        band_results.append(band_result)

    result = _ORA.from_band_stack(band_results, source=raster)
    elapsed = time.perf_counter() - t0
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"dispatch_per_band_cpu bands={raster.band_count} "
                f"shape=({raster.band_count},{raster.height},{raster.width}) "
                f"elapsed={elapsed:.4f}s"
            ),
            residency=result.residency,
        )
    )
    return result
