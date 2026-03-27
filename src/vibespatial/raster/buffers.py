"""Owned raster buffer types for GPU-first raster processing.

Provides OwnedRasterArray -- the canonical device-resident raster representation --
following the same residency, diagnostics, and transfer patterns as
OwnedGeometryArray for vector data.

ADR-0036: Owned Raster Buffer Schema
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.residency import (
    Residency,
    TransferTrigger,
    select_residency_plan,
)
from vibespatial.runtime import RuntimeSelection

if TYPE_CHECKING:
    from pyproj import CRS


# ---------------------------------------------------------------------------
# Diagnostics (mirrors owned_geometry.DiagnosticEvent)
# ---------------------------------------------------------------------------


class RasterDiagnosticKind(StrEnum):
    CREATED = "created"
    TRANSFER = "transfer"
    MATERIALIZATION = "materialization"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class RasterDiagnosticEvent:
    kind: RasterDiagnosticKind
    detail: str
    residency: Residency
    visible_to_user: bool = False
    elapsed_seconds: float = 0.0
    bytes_transferred: int = 0


# ---------------------------------------------------------------------------
# Device state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterDeviceState:
    """GPU-resident mirror of an OwnedRasterArray."""

    data: object  # DeviceArray (CuPy ndarray on device)
    nodata_mask: object | None = None  # DeviceArray, lazily computed


# ---------------------------------------------------------------------------
# Tile chunking spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterTileSpec:
    """Defines how a large raster should be chunked for GPU processing."""

    tile_height: int
    tile_width: int
    overlap: int = 0  # halo pixels for focal/stencil operations

    def tile_count(self, height: int, width: int) -> tuple[int, int]:
        """Return (rows_of_tiles, cols_of_tiles) for the given raster dimensions."""
        effective_h = self.tile_height - 2 * self.overlap
        effective_w = self.tile_width - 2 * self.overlap
        if effective_h <= 0 or effective_w <= 0:
            raise ValueError(
                f"tile dimensions ({self.tile_height}x{self.tile_width}) "
                f"must exceed 2*overlap ({2 * self.overlap})"
            )
        rows = (height + effective_h - 1) // effective_h
        cols = (width + effective_w - 1) // effective_w
        return rows, cols


# ---------------------------------------------------------------------------
# Raster metadata (lightweight, no pixel data)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterMetadata:
    """Raster metadata extracted without reading pixel data."""

    height: int
    width: int
    band_count: int
    dtype: np.dtype
    nodata: float | int | None
    affine: tuple[float, float, float, float, float, float]
    crs: CRS | None
    driver: str = ""

    @property
    def pixel_count(self) -> int:
        return self.height * self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) from the affine transform."""
        a, b, c, d, e, f = self.affine
        corners_x = [
            c,
            c + a * self.width,
            c + b * self.height,
            c + a * self.width + b * self.height,
        ]
        corners_y = [
            f,
            f + d * self.width,
            f + e * self.height,
            f + d * self.width + e * self.height,
        ]
        return (min(corners_x), min(corners_y), max(corners_x), max(corners_y))


# ---------------------------------------------------------------------------
# Raster window (for windowed reads)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RasterWindow:
    """Defines a sub-window into a raster for partial reads."""

    col_off: int
    row_off: int
    width: int
    height: int


# ---------------------------------------------------------------------------
# Interop spec types (ADR-0038)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    """Target grid definition for rasterize operations."""

    affine: tuple[float, float, float, float, float, float]
    width: int
    height: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype("float64"))
    fill_value: float | int = 0

    @staticmethod
    def from_bounds(
        minx: float,
        miny: float,
        maxx: float,
        maxy: float,
        resolution: float,
        dtype: np.dtype | str = "float64",
    ) -> GridSpec:
        """Create a GridSpec from spatial bounds and pixel resolution."""
        width = max(1, int(round((maxx - minx) / resolution)))
        height = max(1, int(round((maxy - miny) / resolution)))
        affine = (resolution, 0.0, minx, 0.0, -resolution, maxy)
        return GridSpec(
            affine=affine,
            width=width,
            height=height,
            dtype=np.dtype(dtype),
        )

    @staticmethod
    def from_raster(raster: OwnedRasterArray) -> GridSpec:
        """Create a GridSpec matching an existing raster's grid."""
        return GridSpec(
            affine=raster.affine,
            width=raster.width,
            height=raster.height,
            dtype=raster.dtype,
        )


class ZonalStatistic(StrEnum):
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STD = "std"
    MEDIAN = "median"


@dataclass(frozen=True)
class ZonalSpec:
    """Configuration for zonal statistics computation."""

    stats: tuple[ZonalStatistic | str, ...] = ("count", "sum", "mean", "min", "max")

    def normalized_stats(self) -> tuple[ZonalStatistic, ...]:
        return tuple(s if isinstance(s, ZonalStatistic) else ZonalStatistic(s) for s in self.stats)


@dataclass(frozen=True)
class PolygonizeSpec:
    """Configuration for raster-to-vector polygonize operations."""

    connectivity: int = 4  # 4 or 8
    value_field: str = "value"
    simplify_tolerance: float | None = None
    max_polygons: int | None = 1_000_000  # explosion guardrail


# ---------------------------------------------------------------------------
# OwnedRasterArray
# ---------------------------------------------------------------------------


@dataclass
class OwnedRasterArray:
    """Owned raster buffer with HOST/DEVICE residency and diagnostic tracking.

    Follows the same residency model as OwnedGeometryArray:
    - Buffers start HOST-resident after creation
    - After first device use, buffers stay DEVICE-resident by default
    - Host materialization is explicit (to_numpy, __repr__)
    - All transfers are tracked via RasterDiagnosticEvent
    """

    data: np.ndarray  # (bands, height, width) or (height, width)
    nodata: float | int | None
    dtype: np.dtype
    affine: tuple[float, float, float, float, float, float]
    crs: CRS | None = None
    residency: Residency = Residency.HOST
    diagnostics: list[RasterDiagnosticEvent] = field(default_factory=list)
    runtime_history: list[RuntimeSelection] = field(default_factory=list)
    device_state: RasterDeviceState | None = None
    _host_materialized: bool = True  # host data is authoritative at creation

    # -- Shape properties --

    @property
    def band_count(self) -> int:
        return self.data.shape[0] if self.data.ndim == 3 else 1

    @property
    def height(self) -> int:
        return self.data.shape[-2]

    @property
    def width(self) -> int:
        return self.data.shape[-1]

    @property
    def pixel_count(self) -> int:
        return self.height * self.width

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    # -- Nodata mask --

    @property
    def nodata_mask(self) -> np.ndarray:
        """Boolean mask where True means nodata (invalid pixel).

        For device-resident rasters whose host data has not been synced,
        this materializes host data first so the mask is computed from
        actual pixel values rather than uninitialized memory.
        """
        # Ensure host data is up-to-date before reading self.data.
        # This is a no-op when _host_materialized is already True (the
        # common case for HOST-resident rasters created via from_numpy).
        self._ensure_host_state()

        if self.nodata is None:
            return np.zeros(self.data.shape, dtype=bool)
        if np.isnan(self.nodata):
            if not np.issubdtype(self.data.dtype, np.floating):
                warnings.warn(
                    f"nodata is NaN but data dtype is {self.data.dtype}; "
                    "NaN cannot exist in integer arrays so mask is all-False",
                    stacklevel=2,
                )
                return np.zeros(self.data.shape, dtype=bool)
            return np.isnan(self.data)
        return self.data == self.nodata

    @property
    def valid_mask(self) -> np.ndarray:
        """Boolean mask where True means valid pixel."""
        return ~self.nodata_mask

    # -- Bounds --

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) from the affine transform."""
        a, b, c, d, e, f = self.affine
        corners_x = [
            c,
            c + a * self.width,
            c + b * self.height,
            c + a * self.width + b * self.height,
        ]
        corners_y = [
            f,
            f + d * self.width,
            f + e * self.height,
            f + d * self.width + e * self.height,
        ]
        return (min(corners_x), min(corners_y), max(corners_x), max(corners_y))

    # -- Metadata --

    @property
    def metadata(self) -> RasterMetadata:
        return RasterMetadata(
            height=self.height,
            width=self.width,
            band_count=self.band_count,
            dtype=self.dtype,
            nodata=self.nodata,
            affine=self.affine,
            crs=self.crs,
        )

    # -- Residency management --

    def move_to(
        self,
        target: Residency,
        *,
        trigger: TransferTrigger = TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
        reason: str = "",
    ) -> None:
        """Move raster data to the target residency."""
        plan = select_residency_plan(
            current=self.residency,
            target=target,
            trigger=trigger,
        )
        if not plan.transfer_required:
            return

        if target is Residency.DEVICE:
            self._ensure_device_state()
        else:
            self._ensure_host_state()

        self.residency = target

    def _ensure_device_state(self) -> RasterDeviceState:
        """Copy host data to device if not already there."""
        if self.device_state is not None:
            return self.device_state

        # Deferred memory pool initialization (ADR-0040): configure the
        # RMM/CuPy pool on the very first H->D transfer so that all
        # subsequent allocations flow through the tiered pool.
        from vibespatial.raster.memory import _ensure_memory_pool

        _ensure_memory_pool()

        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError("CuPy is required for device-resident raster operations") from exc

        t0 = time.perf_counter()
        host_data = np.ascontiguousarray(self.data)
        device_data = cp.asarray(host_data)
        elapsed = time.perf_counter() - t0
        nbytes = host_data.nbytes

        self.device_state = RasterDeviceState(data=device_data)
        self.residency = Residency.DEVICE
        self.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.TRANSFER,
                detail=f"host->device {nbytes:,} bytes ({self.height}x{self.width}x{self.band_count} {self.dtype})",
                residency=Residency.DEVICE,
                visible_to_user=True,
                elapsed_seconds=elapsed,
                bytes_transferred=nbytes,
            )
        )
        return self.device_state

    def _ensure_host_state(self) -> None:
        """Copy device data back to host if needed."""
        if self._host_materialized:
            return
        if self.device_state is None:
            return

        try:
            import cupy as cp
        except ImportError as err:
            raise RuntimeError(
                "Raster has device-resident data that must be transferred to "
                "host, but CuPy is not available. Install CuPy to enable "
                "device-to-host transfers (pip install cupy-cuda12x)."
            ) from err

        t0 = time.perf_counter()
        self.data = cp.asnumpy(self.device_state.data)
        elapsed = time.perf_counter() - t0
        nbytes = self.data.nbytes

        self._host_materialized = True
        self.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.TRANSFER,
                detail=f"device->host {nbytes:,} bytes",
                residency=Residency.HOST,
                visible_to_user=True,
                elapsed_seconds=elapsed,
                bytes_transferred=nbytes,
            )
        )

    # -- Output --

    def to_numpy(self) -> np.ndarray:
        """Return host-resident numpy array, transferring from device if needed."""
        if self.residency is Residency.DEVICE:
            self._ensure_host_state()
        return self.data

    def device_data(self) -> object:
        """Return device-resident CuPy array, transferring from host if needed."""
        state = self._ensure_device_state()
        return state.data

    def device_nodata_mask(self) -> object:
        """Return device-resident nodata mask, computing lazily."""
        state = self._ensure_device_state()
        if state.nodata_mask is not None:
            return state.nodata_mask

        import cupy as cp

        if self.nodata is None:
            mask = cp.zeros(state.data.shape, dtype=cp.bool_)
        elif np.isnan(self.nodata):
            if not np.issubdtype(self.dtype, np.floating):
                warnings.warn(
                    f"nodata is NaN but data dtype is {self.dtype}; "
                    "NaN cannot exist in integer arrays so mask is all-False",
                    stacklevel=2,
                )
                mask = cp.zeros(state.data.shape, dtype=cp.bool_)
            else:
                mask = cp.isnan(state.data)
        else:
            mask = state.data == self.nodata
        self.device_state = RasterDeviceState(data=state.data, nodata_mask=mask)
        return mask

    # -- Band slicing --

    def device_band(self, band_index: int) -> object:
        """Return a 2D CuPy view of a single band (zero-copy slice).

        Parameters
        ----------
        band_index : int
            0-indexed band index.

        Returns
        -------
        object
            CuPy ndarray of shape ``(height, width)`` -- a view into the
            device array, not a copy.

        Raises
        ------
        IndexError
            If *band_index* is out of range for the raster's band count.
        """
        d = self.device_data()
        if d.ndim == 2:
            if band_index != 0:
                raise IndexError(f"single-band raster, got band_index={band_index}")
            return d
        if band_index < 0 or band_index >= d.shape[0]:
            raise IndexError(f"band_index={band_index} out of range for {d.shape[0]}-band raster")
        return d[band_index]

    # -- Band assembly --

    @staticmethod
    def from_band_stack(
        band_results: Sequence[OwnedRasterArray],
        *,
        source: OwnedRasterArray,
    ) -> OwnedRasterArray:
        """Assemble a multiband OwnedRasterArray from per-band results.

        Parameters
        ----------
        band_results : Sequence[OwnedRasterArray]
            One single-band OwnedRasterArray per band, in order.
        source : OwnedRasterArray
            Original raster whose affine, CRS, and nodata are propagated
            to the assembled result.

        Returns
        -------
        OwnedRasterArray
            A new raster with shape ``(len(band_results), H, W)`` (or
            ``(H, W)`` when a single result is passed through unchanged).

        Raises
        ------
        ValueError
            If *band_results* is empty, if dtypes mismatch across bands,
            if nodata values mismatch, or if spatial dimensions differ.
        """
        if not band_results:
            raise ValueError("band_results must not be empty")

        # --- Single-band passthrough (zero overhead) ---
        if len(band_results) == 1:
            result = band_results[0]
            # Propagate source metadata that the caller expects
            result.affine = source.affine
            result.crs = source.crs
            result.nodata = source.nodata
            return result

        # --- Validate consistency across bands ---
        ref_dtype = band_results[0].dtype
        ref_nodata = band_results[0].nodata
        ref_h = band_results[0].height
        ref_w = band_results[0].width

        for i, br in enumerate(band_results[1:], start=1):
            if br.dtype != ref_dtype:
                raise ValueError(f"dtype mismatch: band 0 has {ref_dtype}, band {i} has {br.dtype}")
            # Compare nodata: both None, both NaN, or equal
            if ref_nodata is None and br.nodata is not None:
                raise ValueError(f"nodata mismatch: band 0 has None, band {i} has {br.nodata}")
            if ref_nodata is not None and br.nodata is None:
                raise ValueError(f"nodata mismatch: band 0 has {ref_nodata}, band {i} has None")
            if ref_nodata is not None and br.nodata is not None:
                both_nan = (
                    isinstance(ref_nodata, float)
                    and isinstance(br.nodata, float)
                    and np.isnan(ref_nodata)
                    and np.isnan(br.nodata)
                )
                if not both_nan and ref_nodata != br.nodata:
                    raise ValueError(
                        f"nodata mismatch: band 0 has {ref_nodata}, band {i} has {br.nodata}"
                    )
            if br.height != ref_h or br.width != ref_w:
                raise ValueError(
                    f"shape mismatch: band 0 is ({ref_h}, {ref_w}), "
                    f"band {i} is ({br.height}, {br.width})"
                )

        # --- Determine residency: device if any band is on device ---
        any_on_device = any(br.residency is Residency.DEVICE for br in band_results)

        # --- Merge diagnostics from all bands ---
        merged_diagnostics: list[RasterDiagnosticEvent] = []
        for br in band_results:
            merged_diagnostics.extend(br.diagnostics)

        if any_on_device:
            import cupy as cp

            band_arrays = []
            for br in band_results:
                d = br.device_data()
                # Ensure each result is 2D for stacking
                if d.ndim == 3 and d.shape[0] == 1:
                    d = d[0]
                band_arrays.append(d)
            stacked = cp.stack(band_arrays, axis=0)

            result = from_device(
                stacked,
                nodata=source.nodata,
                affine=source.affine,
                crs=source.crs,
            )
        else:
            band_arrays_np = []
            for br in band_results:
                h = br.to_numpy()
                if h.ndim == 3 and h.shape[0] == 1:
                    h = h[0]
                band_arrays_np.append(h)
            stacked_np = np.stack(band_arrays_np, axis=0)

            result = from_numpy(
                stacked_np,
                nodata=source.nodata,
                affine=source.affine,
                crs=source.crs,
            )

        # Prepend merged diagnostics (before the factory's CREATED event)
        result.diagnostics = merged_diagnostics + result.diagnostics
        return result

    # -- Diagnostics --

    def record_runtime_selection(self, selection: RuntimeSelection) -> None:
        self.runtime_history.append(selection)
        self.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=f"selected={selection.selected} reason={selection.reason}",
                residency=self.residency,
            )
        )

    def diagnostics_report(self) -> dict:
        return {
            "shape": self.shape,
            "dtype": str(self.dtype),
            "band_count": self.band_count,
            "pixel_count": self.pixel_count,
            "nodata": self.nodata,
            "residency": str(self.residency),
            "device_allocated": self.device_state is not None,
            "events": [
                {
                    "kind": str(e.kind),
                    "detail": e.detail,
                    "residency": str(e.residency),
                    "elapsed_seconds": e.elapsed_seconds,
                    "bytes_transferred": e.bytes_transferred,
                }
                for e in self.diagnostics
            ],
        }

    def __repr__(self) -> str:
        return (
            f"OwnedRasterArray("
            f"shape={self.shape}, dtype={self.dtype}, "
            f"nodata={self.nodata}, residency={self.residency}, "
            f"crs={self.crs})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def from_numpy(
    data: np.ndarray,
    *,
    nodata: float | int | None = None,
    affine: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    crs: CRS | None = None,
    residency: Residency = Residency.HOST,
) -> OwnedRasterArray:
    """Create an OwnedRasterArray from a numpy array.

    Parameters
    ----------
    data : np.ndarray
        Pixel data. Shape (height, width) for single-band or
        (bands, height, width) for multi-band.
    nodata : float | int | None
        Nodata sentinel value.
    affine : tuple
        6-element GDAL-style affine: (a, b, c, d, e, f) where
        pixel_x = a * col + b * row + c, pixel_y = d * col + e * row + f.
    crs : CRS | None
        Coordinate reference system.
    residency : Residency
        Initial residency (HOST or DEVICE).
    """
    if data.ndim not in (2, 3):
        raise ValueError(f"raster data must be 2D or 3D, got {data.ndim}D")

    if nodata is not None and np.isnan(nodata) and not np.issubdtype(data.dtype, np.floating):
        raise ValueError(
            f"nodata=NaN is incompatible with integer dtype {data.dtype}; "
            "use a finite sentinel value or cast to a floating-point dtype"
        )

    arr = OwnedRasterArray(
        data=np.ascontiguousarray(data),
        nodata=nodata,
        dtype=data.dtype,
        affine=affine,
        crs=crs,
        residency=Residency.HOST,
        diagnostics=[],
    )
    if residency is Residency.DEVICE:
        arr.move_to(Residency.DEVICE, trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST)
    arr.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.CREATED,
            detail=f"from_numpy shape={data.shape} dtype={data.dtype}",
            residency=residency,
        )
    )
    return arr


def from_device(
    device_data,  # CuPy ndarray or any __cuda_array_interface__ object
    *,
    nodata: float | int | None = None,
    affine: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    crs: CRS | None = None,
) -> OwnedRasterArray:
    """Create an OwnedRasterArray directly from a device array (zero-copy when possible).

    Parameters
    ----------
    device_data
        CuPy ndarray or any object exposing ``__cuda_array_interface__``.
        Wrapped via ``cupy.asarray`` (zero-copy if already CuPy).
    nodata : float | int | None
        Nodata sentinel value.
    affine : tuple
        6-element GDAL-style affine: (a, b, c, d, e, f).
    crs : CRS | None
        Coordinate reference system.
    """
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required to create an OwnedRasterArray from device data"
        ) from exc

    device_arr = cp.asarray(device_data)

    if device_arr.ndim not in (2, 3):
        raise ValueError(f"raster data must be 2D or 3D, got {device_arr.ndim}D")

    host_placeholder = np.empty(device_arr.shape, dtype=device_arr.dtype)

    return OwnedRasterArray(
        data=host_placeholder,
        nodata=nodata,
        dtype=np.dtype(device_arr.dtype),
        affine=affine,
        crs=crs,
        residency=Residency.DEVICE,
        device_state=RasterDeviceState(data=device_arr),
        _host_materialized=False,
        diagnostics=[
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.CREATED,
                detail=f"from_device shape={device_arr.shape} dtype={device_arr.dtype}",
                residency=Residency.DEVICE,
            )
        ],
    )
