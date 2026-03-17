"""Raster IO: GeoTIFF and COG read/write via rasterio or nvImageCodec.

Host-side decode via rasterio, or GPU-native decode via nvImageCodec.
The ``read_raster`` dispatcher tries the GPU path first (when available),
falling back to rasterio transparently.

ADR-0037: Raster IO Support and Read Paths
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    RasterMetadata,
    RasterWindow,
    from_device,
    from_numpy,
)
from vibespatial.residency import Residency, TransferTrigger

if TYPE_CHECKING:
    from pyproj import CRS


def has_rasterio_support() -> bool:
    """Check whether rasterio is available."""
    try:
        return find_spec("rasterio") is not None
    except ModuleNotFoundError:
        return False


def _require_rasterio():
    if not has_rasterio_support():
        raise ImportError(
            "rasterio is required for raster IO. Install it with: uv sync --extra upstream-optional"
        )


def _affine_to_tuple(transform) -> tuple[float, float, float, float, float, float]:
    """Convert a rasterio Affine to a 6-element tuple (a, b, c, d, e, f)."""
    return (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)


def _tuple_to_affine(t: tuple[float, float, float, float, float, float]):
    """Convert a 6-element tuple to a rasterio Affine."""
    import rasterio.transform

    return rasterio.transform.Affine(t[0], t[1], t[2], t[3], t[4], t[5])


def _extract_crs(src) -> CRS | None:
    """Extract a pyproj CRS from a rasterio dataset, or None."""
    if src.crs is None:
        return None
    try:
        from pyproj import CRS

        return CRS.from_user_input(src.crs)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Metadata-only read (no pixel data)
# ---------------------------------------------------------------------------


def read_raster_metadata(path: str | Path) -> RasterMetadata:
    """Read raster metadata without loading pixel data.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF, COG, or other rasterio-supported raster file.

    Returns
    -------
    RasterMetadata
        Shape, dtype, nodata, affine, CRS, and driver information.
    """
    _require_rasterio()
    import rasterio

    with rasterio.open(path) as src:
        nodata = src.nodata
        return RasterMetadata(
            height=src.height,
            width=src.width,
            band_count=src.count,
            dtype=np.dtype(src.dtypes[0]),
            nodata=float(nodata) if nodata is not None else None,
            affine=_affine_to_tuple(src.transform),
            crs=_extract_crs(src),
            driver=src.driver,
        )


# ---------------------------------------------------------------------------
# nvImageCodec availability check
# ---------------------------------------------------------------------------


def has_nvimgcodec_support() -> bool:
    """Check whether nvImageCodec GPU decode is available."""
    try:
        from vibespatial.raster.nvimgcodec_io import has_nvimgcodec_support as _check

        return _check()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# nvImageCodec dispatch helper
# ---------------------------------------------------------------------------


def _try_nvimgcodec_read(path, *, bands, window, overview_level):
    """Attempt nvImageCodec GPU decode. Returns (device_data, metadata) or None."""
    try:
        from vibespatial.raster.nvimgcodec_io import (
            has_nvimgcodec_support as _check,
            nvimgcodec_read,
        )

        if not _check():
            return None
        return nvimgcodec_read(path, bands=bands, window=window, overview_level=overview_level)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# rasterio read helper
# ---------------------------------------------------------------------------


def _read_raster_rasterio(path, *, bands, window, overview_level, residency):
    """Read raster via rasterio (HYBRID path)."""
    _require_rasterio()
    import rasterio
    from rasterio.windows import Window

    open_kwargs = {}
    if overview_level is not None:
        open_kwargs["overview_level"] = overview_level

    with rasterio.open(path, **open_kwargs) as src:
        # Determine bands to read
        if bands is None:
            band_indices = list(range(1, src.count + 1))
        else:
            band_indices = bands

        # Build rasterio Window
        rio_window = None
        if window is not None:
            rio_window = Window(
                col_off=window.col_off,
                row_off=window.row_off,
                width=window.width,
                height=window.height,
            )

        # Read pixel data
        data = src.read(band_indices, window=rio_window)
        # data shape: (bands, height, width)

        # Squeeze single-band to 2D
        if data.shape[0] == 1:
            data = data[0]

        # Extract metadata
        nodata = src.nodata
        if rio_window is not None:
            transform = src.window_transform(rio_window)
        else:
            transform = src.transform

        affine = _affine_to_tuple(transform)
        crs = _extract_crs(src)

    result = from_numpy(
        data,
        nodata=float(nodata) if nodata is not None else None,
        affine=affine,
        crs=crs,
        residency=residency,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"read_raster backend=rasterio path={Path(path).name}",
            residency=result.residency,
            visible_to_user=True,
        )
    )
    return result


# ---------------------------------------------------------------------------
# Full read
# ---------------------------------------------------------------------------


def read_raster(
    path: str | Path,
    *,
    bands: list[int] | None = None,
    window: RasterWindow | None = None,
    overview_level: int | None = None,
    residency: Residency = Residency.HOST,
    decode_backend: str = "auto",
) -> OwnedRasterArray:
    """Read a raster file into an OwnedRasterArray.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF, COG, or other rasterio-supported raster file.
    bands : list[int] or None
        1-based band indices to read. None reads all bands.
    window : RasterWindow or None
        Sub-window to read. None reads the full raster.
    overview_level : int or None
        Overview (pyramid) level to read. None reads full resolution.
    residency : Residency
        Target residency for the output array.
    decode_backend : str
        Decode backend selection: ``"auto"`` (try GPU first, fall back to
        rasterio), ``"nvimgcodec"`` (GPU-only, raises on failure), or
        ``"rasterio"`` (CPU-only, skip GPU path).

    Returns
    -------
    OwnedRasterArray
        The raster data with metadata, in the requested residency.
    """
    path = str(path)  # normalize

    # --- Try GPU-native decode first ---
    if decode_backend in ("auto", "nvimgcodec"):
        gpu_result = _try_nvimgcodec_read(
            path,
            bands=bands,
            window=window,
            overview_level=overview_level,
        )
        if gpu_result is not None:
            device_data, meta = gpu_result
            # Supplement nodata from rasterio if nvimgcodec didn't provide it
            if meta.nodata is None and has_rasterio_support():
                try:
                    rio_meta = read_raster_metadata(path)
                    nodata = rio_meta.nodata
                except Exception:
                    nodata = None
            else:
                nodata = meta.nodata

            result = from_device(
                device_data,
                nodata=nodata,
                affine=meta.affine if meta.affine else (1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
                crs=meta.crs,
            )
            result.diagnostics.append(
                RasterDiagnosticEvent(
                    kind=RasterDiagnosticKind.RUNTIME,
                    detail=f"read_raster backend=nvimgcodec path={Path(path).name}",
                    residency=result.residency,
                    visible_to_user=True,
                )
            )
            # If user wants HOST residency, transfer from device
            if residency is Residency.HOST:
                result.move_to(
                    Residency.HOST,
                    trigger=TransferTrigger.EXPLICIT_RUNTIME_REQUEST,
                    reason="read_raster requested HOST residency",
                )
            return result

        # If nvimgcodec was explicitly requested but failed, raise
        if decode_backend == "nvimgcodec":
            raise RuntimeError(
                f"nvimgcodec decode failed for {path}. "
                "Ensure nvidia-nvimgcodec-cu12[all] is installed and the format is supported."
            )

    # --- Fallback to rasterio ---
    return _read_raster_rasterio(
        path,
        bands=bands,
        window=window,
        overview_level=overview_level,
        residency=residency,
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_raster(
    path: str | Path,
    raster: OwnedRasterArray,
    *,
    driver: str = "GTiff",
    compress: str | None = "deflate",
    tiled: bool = True,
    blockxsize: int = 256,
    blockysize: int = 256,
) -> None:
    """Write an OwnedRasterArray to a raster file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    raster : OwnedRasterArray
        The raster to write.
    driver : str
        GDAL driver name (default "GTiff").
    compress : str or None
        Compression algorithm (default "deflate"). None for no compression.
    tiled : bool
        Whether to write tiled (default True for COG compatibility).
    blockxsize, blockysize : int
        Tile dimensions when tiled=True.
    """
    _require_rasterio()
    import rasterio

    host_data = raster.to_numpy()

    # Ensure 3D (bands, height, width)
    if host_data.ndim == 2:
        host_data = host_data[np.newaxis, :, :]

    band_count = host_data.shape[0]
    height = host_data.shape[1]
    width = host_data.shape[2]

    profile = {
        "driver": driver,
        "dtype": str(raster.dtype),
        "width": width,
        "height": height,
        "count": band_count,
        "transform": _tuple_to_affine(raster.affine),
        "tiled": tiled,
    }

    if raster.nodata is not None:
        profile["nodata"] = raster.nodata
    if raster.crs is not None:
        profile["crs"] = raster.crs.to_epsg() or raster.crs.to_wkt()
    if compress is not None:
        profile["compress"] = compress
    if tiled:
        profile["blockxsize"] = blockxsize
        profile["blockysize"] = blockysize

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(host_data)

    raster.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.MATERIALIZATION,
            detail=f"wrote {height}x{width}x{band_count} to {path} driver={driver}",
            residency=raster.residency,
            visible_to_user=True,
        )
    )
