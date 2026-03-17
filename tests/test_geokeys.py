"""Tests for GeoTIFF GeoKey parsing (pure CPU, no GPU dependency)."""

from __future__ import annotations

import pytest

from vibespatial.raster.geokeys import (
    geokeys_to_affine,
    geokeys_to_crs,
    parse_nvimgcodec_geo_metadata,
)


# ---------------------------------------------------------------------------
# geokeys_to_affine — PixelScale + Tiepoint
# ---------------------------------------------------------------------------


class TestGeokeysToAffine:
    def test_north_up_standard(self):
        affine = geokeys_to_affine(
            pixel_scale=[1, 1, 0],
            tiepoint=[0, 0, 0, 100, 200, 0],
        )
        assert affine == (1, 0, 100, 0, -1, 200)

    def test_nonzero_tiepoint_offset(self):
        # c = x - i*scaleX = 500 - 10*2 = 480
        # f = y + j*scaleY = 600 + 20*3 = 660
        affine = geokeys_to_affine(
            pixel_scale=[2, 3, 0],
            tiepoint=[10, 20, 0, 500, 600, 0],
        )
        assert affine == (2, 0, 480, 0, -3, 660)

    def test_fine_resolution(self):
        affine = geokeys_to_affine(
            pixel_scale=[0.001, 0.001, 0],
            tiepoint=[0, 0, 0, -35.5, 39.2, 0],
        )
        assert affine is not None
        a, b, c, d, e, f = affine
        assert a == pytest.approx(0.001)
        assert b == 0.0
        assert c == pytest.approx(-35.5)
        assert d == 0.0
        assert e == pytest.approx(-0.001)
        assert f == pytest.approx(39.2)

    def test_model_transformation_matrix(self):
        # 4x4 row-major: [a, b, 0, c, d, e, 0, f, 0, 0, 1, 0, 0, 0, 0, 1]
        mat = [
            10.0,
            0.5,
            0.0,
            100.0,
            0.3,
            -10.0,
            0.0,
            200.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        affine = geokeys_to_affine(model_transformation=mat)
        assert affine == (10.0, 0.5, 100.0, 0.3, -10.0, 200.0)

    def test_missing_scale_returns_none(self):
        assert geokeys_to_affine(tiepoint=[0, 0, 0, 1, 2, 0]) is None

    def test_missing_tiepoint_returns_none(self):
        assert geokeys_to_affine(pixel_scale=[1, 1, 0]) is None

    def test_model_transformation_overrides_scale_tiepoint(self):
        mat = [
            5.0,
            0.0,
            0.0,
            50.0,
            0.0,
            -5.0,
            0.0,
            60.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]
        affine = geokeys_to_affine(
            pixel_scale=[1, 1, 0],
            tiepoint=[0, 0, 0, 100, 200, 0],
            model_transformation=mat,
        )
        # model_transformation wins
        assert affine == (5.0, 0.0, 50.0, 0.0, -5.0, 60.0)


# ---------------------------------------------------------------------------
# geokeys_to_crs
# ---------------------------------------------------------------------------


class TestGeokeysToCrs:
    def test_epsg_4326_to_crs(self):
        crs = geokeys_to_crs(4326)
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_epsg_32633_to_crs(self):
        crs = geokeys_to_crs(32633)
        assert crs is not None
        assert crs.to_epsg() == 32633

    def test_unknown_epsg_returns_none(self):
        crs = geokeys_to_crs(99999)
        assert crs is None

    def test_none_epsg_returns_none(self):
        crs = geokeys_to_crs(None)
        assert crs is None


# ---------------------------------------------------------------------------
# parse_nvimgcodec_geo_metadata
# ---------------------------------------------------------------------------


class TestParseNvimgcodecGeoMetadata:
    def test_parse_full_metadata(self):
        geo = {
            "MODEL_PIXEL_SCALE": [0.00259611, 0.0022483, 0.0],
            "MODEL_TIEPOINT": [0.0, 0.0, 0.0, -35.489, 39.1935, 0.0],
            "GT_MODEL_TYPE": 2,
            "GT_RASTER_TYPE": 1,
            "GEODETIC_CRS": 4326,
            "GEODETIC_CITATION": "WGS 84",
        }
        affine, crs = parse_nvimgcodec_geo_metadata(geo)
        assert affine is not None
        a, b, c, d, e, f = affine
        assert a == pytest.approx(0.00259611)
        assert b == 0.0
        assert c == pytest.approx(-35.489)
        assert d == 0.0
        assert e == pytest.approx(-0.0022483)
        assert f == pytest.approx(39.1935)
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_parse_minimal_metadata(self):
        geo = {"GEODETIC_CRS": 4326}
        affine, crs = parse_nvimgcodec_geo_metadata(geo)
        assert affine is None
        assert crs is not None
        assert crs.to_epsg() == 4326

    def test_parse_empty_metadata(self):
        affine, crs = parse_nvimgcodec_geo_metadata({})
        assert affine is None
        assert crs is None

    def test_projected_crs_preferred(self):
        geo = {
            "GEODETIC_CRS": 4326,
            "PROJECTED_CRS": 32633,
        }
        _, crs = parse_nvimgcodec_geo_metadata(geo)
        assert crs is not None
        assert crs.to_epsg() == 32633
