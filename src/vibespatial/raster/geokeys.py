"""GeoTIFF GeoKey parsing for nvImageCodec geo metadata.

Converts raw GeoKey dicts (MODEL_PIXEL_SCALE, MODEL_TIEPOINT, EPSG codes)
into the affine 6-tuple and pyproj.CRS that vibespatial-raster uses.

No dependency on nvImageCodec, CuPy, or rasterio.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyproj import CRS


def geokeys_to_affine(
    pixel_scale: list[float] | None = None,
    tiepoint: list[float] | None = None,
    model_transformation: list[float] | None = None,
) -> tuple[float, float, float, float, float, float] | None:
    """Convert GeoTIFF geo-referencing tags to a GDAL-style affine 6-tuple.

    The affine tuple ``(a, b, c, d, e, f)`` maps pixel coordinates to world
    coordinates:

        world_x = a * col + b * row + c
        world_y = d * col + e * row + f

    Parameters
    ----------
    pixel_scale : list[float] or None
        ``MODEL_PIXEL_SCALE`` — ``[scaleX, scaleY, scaleZ]``.
    tiepoint : list[float] or None
        ``MODEL_TIEPOINT`` — ``[i, j, k, x, y, z]``.
    model_transformation : list[float] or None
        ``MODEL_TRANSFORMATION`` — 16-element row-major 4x4 matrix.

    Returns
    -------
    tuple[float, float, float, float, float, float] or None
        ``(a, b, c, d, e, f)`` affine, or ``None`` if insufficient data.
    """
    # MODEL_TRANSFORMATION takes precedence (already a full affine matrix).
    if model_transformation is not None and len(model_transformation) == 16:
        mat = model_transformation
        a = mat[0]
        b = mat[1]
        c = mat[3]
        d = mat[4]
        e = mat[5]
        f = mat[7]
        return (a, b, c, d, e, f)

    # Common case: PixelScale + Tiepoint (north-up raster).
    if pixel_scale is not None and tiepoint is not None:
        if len(pixel_scale) < 2 or len(tiepoint) < 6:
            return None
        scale_x, scale_y = pixel_scale[0], pixel_scale[1]
        i, j = tiepoint[0], tiepoint[1]
        x, y = tiepoint[3], tiepoint[4]
        a = scale_x
        b = 0.0
        c = x - i * scale_x
        d = 0.0
        e = -scale_y
        f = y + j * scale_y
        return (a, b, c, d, e, f)

    return None


def geokeys_to_crs(
    epsg_code: int | None,
    *,
    model_type: int | None = None,
) -> CRS | None:
    """Build a ``pyproj.CRS`` from a GeoTIFF EPSG code.

    Parameters
    ----------
    epsg_code : int or None
        EPSG code (e.g. 4326, 32633). ``None`` means no CRS.
    model_type : int or None
        ``GT_MODEL_TYPE`` (1 = Projected, 2 = Geographic). Currently
        informational only; the EPSG code already encodes this distinction.

    Returns
    -------
    CRS or None
        A ``pyproj.CRS`` object, or ``None`` if the code is missing or
        unrecognised.
    """
    if epsg_code is None:
        return None
    try:
        from pyproj import CRS

        return CRS.from_epsg(epsg_code)
    except Exception:
        return None


def parse_nvimgcodec_geo_metadata(
    geo_metadata: dict,
) -> tuple[tuple[float, float, float, float, float, float] | None, CRS | None]:
    """Parse an nvImageCodec geo-metadata dict into an affine tuple and CRS.

    This is the main entry point for converting the raw dict that
    ``nvImageCodec`` attaches to decoded GeoTIFF images.

    Parameters
    ----------
    geo_metadata : dict
        Raw geo-metadata dict with keys such as ``MODEL_PIXEL_SCALE``,
        ``MODEL_TIEPOINT``, ``MODEL_TRANSFORMATION``, ``PROJECTED_CRS``,
        ``GEODETIC_CRS``, and ``GT_MODEL_TYPE``.

    Returns
    -------
    tuple[tuple | None, CRS | None]
        ``(affine, crs)`` — either value may be ``None`` when the
        corresponding metadata is missing or unparseable.
    """
    pixel_scale = geo_metadata.get("MODEL_PIXEL_SCALE")
    tiepoint = geo_metadata.get("MODEL_TIEPOINT")
    model_transformation = geo_metadata.get("MODEL_TRANSFORMATION")

    affine = geokeys_to_affine(
        pixel_scale=pixel_scale,
        tiepoint=tiepoint,
        model_transformation=model_transformation,
    )

    # Prefer PROJECTED_CRS when both are present.
    epsg_code = geo_metadata.get("PROJECTED_CRS") or geo_metadata.get("GEODETIC_CRS")
    model_type = geo_metadata.get("GT_MODEL_TYPE")

    crs = geokeys_to_crs(epsg_code, model_type=model_type)

    return (affine, crs)
