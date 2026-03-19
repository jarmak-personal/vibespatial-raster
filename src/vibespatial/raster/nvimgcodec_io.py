"""GPU-native raster decode via NVIDIA nvImageCodec.

Decodes GeoTIFF and JPEG2000 files directly to device memory using
nvTIFF and nvJPEG2000 backends. Falls back gracefully when nvImageCodec
is unavailable or the format is unsupported.

Install: pip install nvidia-nvimgcodec-cu12[all]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from vibespatial.raster.buffers import RasterMetadata, RasterWindow

logger = logging.getLogger(__name__)

# Module-level singleton for the decoder (lazy init).
_decoder = None


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def has_nvimgcodec_support() -> bool:
    """Return True if both nvImageCodec and CuPy are importable."""
    try:
        import cupy  # noqa: F401
        import nvidia.nvimgcodec  # noqa: F401

        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Decoder singleton
# ---------------------------------------------------------------------------


def _get_decoder():
    """Return a cached nvImageCodec Decoder with GPU-preferred backends.

    Creates the decoder on first call, then returns the cached instance.
    Returns None if nvImageCodec is not available.
    """
    global _decoder
    if _decoder is not None:
        return _decoder

    try:
        import nvidia.nvimgcodec as nvimgcodec

        _decoder = nvimgcodec.Decoder(
            backends=[
                nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY),
                nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
                nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY),
            ]
        )
        return _decoder
    except Exception:
        logger.debug("Failed to create nvImageCodec Decoder", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------

# Mapping from nvImageCodec dtype string/enum values to numpy dtypes.
_NVIMGCODEC_DTYPE_MAP: dict[str, np.dtype] = {
    "uint8": np.dtype("uint8"),
    "uint16": np.dtype("uint16"),
    "int8": np.dtype("int8"),
    "int16": np.dtype("int16"),
    "int32": np.dtype("int32"),
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
}


def _nvimgcodec_dtype_to_numpy(nvimgcodec_dtype) -> np.dtype | None:
    """Map an nvImageCodec dtype value to a numpy dtype.

    Parameters
    ----------
    nvimgcodec_dtype
        The ``.dtype`` attribute from an nvImageCodec Image or CodeStream.
        May be a string, enum, or numpy-compatible dtype object.

    Returns
    -------
    np.dtype or None
        The corresponding numpy dtype, or None if unmappable.
    """
    # If it already is or can be directly converted to a numpy dtype, use that.
    try:
        return np.dtype(nvimgcodec_dtype)
    except (TypeError, ValueError):
        pass

    # Try string lookup.
    dtype_str = str(nvimgcodec_dtype).lower().strip()
    return _NVIMGCODEC_DTYPE_MAP.get(dtype_str)


# ---------------------------------------------------------------------------
# Geo metadata extraction helper
# ---------------------------------------------------------------------------


def _extract_geo_metadata(code_stream):
    """Attempt to extract geo metadata dict from a CodeStream.

    Returns (affine, crs) tuple via the geokeys parser, or (None, None).
    """
    try:
        geo_metadata = code_stream.metadata
        if not isinstance(geo_metadata, dict) or not geo_metadata:
            return None, None
    except Exception:
        return None, None

    try:
        from vibespatial.raster.geokeys import parse_nvimgcodec_geo_metadata

        return parse_nvimgcodec_geo_metadata(geo_metadata)
    except Exception:
        logger.debug("Failed to parse geo metadata from CodeStream", exc_info=True)
        return None, None


# ---------------------------------------------------------------------------
# Metadata-only read
# ---------------------------------------------------------------------------


def nvimgcodec_read_metadata(path: str | Path) -> RasterMetadata | None:
    """Read raster metadata via nvImageCodec without decoding pixel data.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF, JPEG2000, or other nvImageCodec-supported file.

    Returns
    -------
    RasterMetadata or None
        Extracted metadata, or None if nvImageCodec cannot handle the file.
    """
    try:
        import nvidia.nvimgcodec as nvimgcodec

        path = str(path)
        cs = nvimgcodec.CodeStream(path)

        height = cs.height
        width = cs.width
        num_channels = cs.num_channels
        nv_dtype = cs.dtype

        np_dtype = _nvimgcodec_dtype_to_numpy(nv_dtype)
        if np_dtype is None:
            logger.warning(
                "nvimgcodec_read_metadata: unmappable dtype %r for %s",
                nv_dtype,
                path,
            )
            return None

        affine, crs = _extract_geo_metadata(cs)

        # Use identity affine if geo metadata is absent.
        if affine is None:
            affine = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        return RasterMetadata(
            height=height,
            width=width,
            band_count=num_channels,
            dtype=np_dtype,
            nodata=None,  # nvImageCodec doesn't expose GDAL_NODATA
            affine=affine,
            crs=crs,
            driver="nvimgcodec",
        )
    except Exception:
        logger.debug("nvimgcodec_read_metadata failed for %s", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Full read (decode to device)
# ---------------------------------------------------------------------------


def nvimgcodec_read(
    path: str | Path,
    *,
    bands: list[int] | None = None,
    window: RasterWindow | None = None,
    overview_level: int | None = None,
) -> tuple[object, RasterMetadata] | None:
    """Decode a raster file directly to GPU device memory via nvImageCodec.

    Parameters
    ----------
    path : str or Path
        Path to a GeoTIFF, JPEG2000, or other nvImageCodec-supported file.
    bands : list[int] or None
        1-based band indices to select. None returns all bands.
    window : RasterWindow or None
        Sub-region to decode. None decodes the full raster.
    overview_level : int or None
        If not None, returns None immediately (overviews are not supported
        by nvImageCodec; the caller should fall back to rasterio).

    Returns
    -------
    tuple[object, RasterMetadata] or None
        ``(cupy_array, metadata)`` on success, or None on any failure.
        The CuPy array shape is ``(H, W)`` for single-band or
        ``(C, H, W)`` for multi-band, matching the rasterio convention.
    """
    # Overview reads are unsupported -- signal fallback.
    if overview_level is not None:
        return None

    try:
        import cupy as cp
        import nvidia.nvimgcodec as nvimgcodec

        path = str(path)
        decoder = _get_decoder()
        if decoder is None:
            return None

        cs = nvimgcodec.CodeStream(path)

        # ---- Region / window handling ----
        decode_kwargs: dict = {}
        if window is not None:
            region = nvimgcodec.Region(
                start=[window.row_off, window.col_off],
                end=[window.row_off + window.height, window.col_off + window.width],
            )
            decode_kwargs["region"] = region

        # ---- Decode params: preserve native dtype and band count ----
        color_spec = (
            nvimgcodec.ColorSpec.GRAY if cs.num_channels == 1 else nvimgcodec.ColorSpec.SRGB
        )
        decode_kwargs["params"] = nvimgcodec.DecodeParams(
            allow_any_depth=True, color_spec=color_spec
        )

        # ---- Decode to device ----
        nv_img = decoder.decode(cs, **decode_kwargs)

        # ---- Zero-copy wrap as CuPy ----
        arr = cp.asarray(nv_img)

        # ---- Dtype ----
        np_dtype = np.dtype(arr.dtype)

        # ---- Band reorder: nvImageCodec returns (H, W, C) interleaved ----
        if arr.ndim == 3:
            num_channels = arr.shape[2]
            if num_channels == 1:
                # Single channel -- squeeze to (H, W).
                arr = arr[:, :, 0]
            else:
                # Multi-channel -- transpose to (C, H, W).
                arr = cp.transpose(arr, (2, 0, 1))
        # 2D arrays are already (H, W) -- no reorder needed.

        # ---- Band selection (bands param is 1-indexed) ----
        if bands is not None and arr.ndim == 3:
            band_indices = [b - 1 for b in bands]
            arr = arr[band_indices]
            # Squeeze if selecting a single band.
            if arr.shape[0] == 1:
                arr = arr[0]

        # ---- Determine final shape info ----
        if arr.ndim == 3:
            band_count = arr.shape[0]
            out_height = arr.shape[1]
            out_width = arr.shape[2]
        else:
            band_count = 1
            out_height = arr.shape[0]
            out_width = arr.shape[1]

        # ---- Geo metadata ----
        affine, crs = _extract_geo_metadata(cs)
        if affine is None:
            affine = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)

        # ---- Adjust affine for windowed reads ----
        if window is not None:
            a, b, c, d, e, f = affine
            new_c = c + a * window.col_off + b * window.row_off
            new_f = f + d * window.col_off + e * window.row_off
            affine = (a, b, new_c, d, e, new_f)

        metadata = RasterMetadata(
            height=out_height,
            width=out_width,
            band_count=band_count,
            dtype=np_dtype,
            nodata=None,
            affine=affine,
            crs=crs,
            driver="nvimgcodec",
        )

        return (arr, metadata)

    except Exception:
        logger.warning("nvimgcodec_read failed for %s", path, exc_info=True)
        return None
