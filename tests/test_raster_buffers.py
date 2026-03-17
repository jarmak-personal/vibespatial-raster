"""Tests for OwnedRasterArray, raster buffer schema, and interop specs."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import (
    GridSpec,
    PolygonizeSpec,
    RasterDiagnosticKind,
    RasterMetadata,
    RasterTileSpec,
    ZonalSpec,
    ZonalStatistic,
    from_device,
    from_numpy,
)
from vibespatial.residency import Residency

try:
    import cupy  # noqa: F401

    HAS_GPU = True
    HAS_CUPY = True
except ImportError:
    HAS_GPU = False
    HAS_CUPY = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


# ---------------------------------------------------------------------------
# OwnedRasterArray creation
# ---------------------------------------------------------------------------


class TestFromNumpy:
    def test_single_band_2d(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        raster = from_numpy(data, nodata=-1.0)
        assert raster.height == 3
        assert raster.width == 4
        assert raster.band_count == 1
        assert raster.pixel_count == 12
        assert raster.dtype == np.float32
        assert raster.nodata == -1.0
        assert raster.residency is Residency.HOST
        assert raster.shape == (3, 4)

    def test_multi_band_3d(self):
        data = np.zeros((3, 100, 200), dtype=np.uint8)
        raster = from_numpy(data)
        assert raster.band_count == 3
        assert raster.height == 100
        assert raster.width == 200
        assert raster.pixel_count == 20_000
        assert raster.shape == (3, 100, 200)

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            from_numpy(np.zeros(10))

    def test_rejects_4d(self):
        with pytest.raises(ValueError, match="2D or 3D"):
            from_numpy(np.zeros((2, 3, 4, 5)))

    def test_creation_diagnostic(self):
        data = np.zeros((10, 10), dtype=np.float64)
        raster = from_numpy(data)
        assert len(raster.diagnostics) == 1
        assert raster.diagnostics[0].kind == RasterDiagnosticKind.CREATED

    def test_default_affine(self):
        raster = from_numpy(np.zeros((5, 5)))
        assert raster.affine == (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


# ---------------------------------------------------------------------------
# Nodata mask
# ---------------------------------------------------------------------------


class TestNodataMask:
    def test_no_nodata(self):
        raster = from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        mask = raster.nodata_mask
        assert not mask.any()
        assert raster.valid_mask.all()

    def test_integer_nodata(self):
        data = np.array([[1, -9999], [3, -9999]], dtype=np.int32)
        raster = from_numpy(data, nodata=-9999)
        mask = raster.nodata_mask
        assert mask[0, 1]
        assert mask[1, 1]
        assert not mask[0, 0]
        assert not mask[1, 0]

    def test_nan_nodata(self):
        data = np.array([[1.0, np.nan], [3.0, np.nan]])
        raster = from_numpy(data, nodata=np.nan)
        mask = raster.nodata_mask
        assert mask[0, 1]
        assert mask[1, 1]
        assert not mask[0, 0]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestBounds:
    def test_identity_affine(self):
        raster = from_numpy(
            np.zeros((10, 20)),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 10.0),
        )
        minx, miny, maxx, maxy = raster.bounds
        assert minx == 0.0
        assert maxx == 20.0
        assert miny == 0.0
        assert maxy == 10.0

    def test_offset_affine(self):
        raster = from_numpy(
            np.zeros((100, 200)),
            affine=(0.5, 0.0, 10.0, 0.0, -0.5, 60.0),
        )
        minx, miny, maxx, maxy = raster.bounds
        assert minx == 10.0
        assert maxx == 110.0
        assert miny == 10.0
        assert maxy == 60.0


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_metadata_extraction(self):
        data = np.zeros((3, 50, 100), dtype=np.float32)
        raster = from_numpy(data, nodata=-1.0, affine=(0.1, 0.0, 5.0, 0.0, -0.1, 10.0))
        meta = raster.metadata
        assert isinstance(meta, RasterMetadata)
        assert meta.height == 50
        assert meta.width == 100
        assert meta.band_count == 3
        assert meta.dtype == np.float32
        assert meta.nodata == -1.0
        assert meta.pixel_count == 5000


# ---------------------------------------------------------------------------
# Residency
# ---------------------------------------------------------------------------


class TestResidency:
    def test_starts_host(self):
        raster = from_numpy(np.zeros((10, 10)))
        assert raster.residency is Residency.HOST
        assert raster.device_state is None

    def test_to_numpy_on_host(self):
        data = np.arange(6).reshape(2, 3).astype(np.float64)
        raster = from_numpy(data)
        result = raster.to_numpy()
        np.testing.assert_array_equal(result, data)

    @gpu
    def test_move_to_device(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        raster = from_numpy(data)
        raster.move_to(Residency.DEVICE)
        assert raster.residency is Residency.DEVICE
        assert raster.device_state is not None

    @gpu
    def test_device_roundtrip(self):
        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        raster = from_numpy(data)
        raster.move_to(Residency.DEVICE)
        raster._host_materialized = False
        result = raster.to_numpy()
        np.testing.assert_array_equal(result, data)

    @gpu
    def test_transfer_diagnostics(self):
        raster = from_numpy(np.zeros((10, 10), dtype=np.float32))
        raster.move_to(Residency.DEVICE)
        transfer_events = [e for e in raster.diagnostics if e.kind == RasterDiagnosticKind.TRANSFER]
        assert len(transfer_events) == 1
        assert transfer_events[0].bytes_transferred == 10 * 10 * 4
        assert transfer_events[0].visible_to_user

    @gpu
    def test_device_nodata_mask(self):
        data = np.array([[1.0, -9999.0], [3.0, -9999.0]], dtype=np.float32)
        raster = from_numpy(data, nodata=-9999.0)
        mask = raster.device_nodata_mask()
        import cupy as cp

        host_mask = cp.asnumpy(mask)
        assert host_mask[0, 1]
        assert host_mask[1, 1]
        assert not host_mask[0, 0]


# ---------------------------------------------------------------------------
# TileSpec
# ---------------------------------------------------------------------------


class TestRasterTileSpec:
    def test_tile_count_exact(self):
        spec = RasterTileSpec(tile_height=256, tile_width=256)
        rows, cols = spec.tile_count(512, 1024)
        assert rows == 2
        assert cols == 4

    def test_tile_count_remainder(self):
        spec = RasterTileSpec(tile_height=256, tile_width=256)
        rows, cols = spec.tile_count(300, 500)
        assert rows == 2
        assert cols == 2

    def test_tile_count_with_overlap(self):
        spec = RasterTileSpec(tile_height=256, tile_width=256, overlap=16)
        # Effective: 224x224
        rows, cols = spec.tile_count(448, 448)
        assert rows == 2
        assert cols == 2

    def test_rejects_bad_overlap(self):
        spec = RasterTileSpec(tile_height=32, tile_width=32, overlap=16)
        with pytest.raises(ValueError, match="must exceed"):
            spec.tile_count(100, 100)


# ---------------------------------------------------------------------------
# GridSpec
# ---------------------------------------------------------------------------


class TestGridSpec:
    def test_from_bounds(self):
        gs = GridSpec.from_bounds(0.0, 0.0, 100.0, 50.0, resolution=0.5)
        assert gs.width == 200
        assert gs.height == 100
        assert gs.affine[0] == 0.5  # pixel width
        assert gs.affine[4] == -0.5  # pixel height (negative = north-up)
        assert gs.dtype == np.float64

    def test_from_raster(self):
        raster = from_numpy(
            np.zeros((10, 20), dtype=np.int16),
            affine=(0.5, 0.0, 10.0, 0.0, -0.5, 60.0),
        )
        gs = GridSpec.from_raster(raster)
        assert gs.width == 20
        assert gs.height == 10
        assert gs.affine == raster.affine
        assert gs.dtype == np.int16


# ---------------------------------------------------------------------------
# ZonalSpec
# ---------------------------------------------------------------------------


class TestZonalSpec:
    def test_default_stats(self):
        spec = ZonalSpec()
        normalized = spec.normalized_stats()
        assert ZonalStatistic.COUNT in normalized
        assert ZonalStatistic.MEAN in normalized

    def test_custom_stats(self):
        spec = ZonalSpec(stats=("min", "max", "median"))
        normalized = spec.normalized_stats()
        assert len(normalized) == 3
        assert ZonalStatistic.MEDIAN in normalized


# ---------------------------------------------------------------------------
# PolygonizeSpec
# ---------------------------------------------------------------------------


class TestPolygonizeSpec:
    def test_defaults(self):
        spec = PolygonizeSpec()
        assert spec.connectivity == 4
        assert spec.value_field == "value"
        assert spec.max_polygons == 1_000_000

    def test_custom(self):
        spec = PolygonizeSpec(connectivity=8, simplify_tolerance=0.5)
        assert spec.connectivity == 8
        assert spec.simplify_tolerance == 0.5


# ---------------------------------------------------------------------------
# Diagnostics report
# ---------------------------------------------------------------------------


class TestDiagnosticsReport:
    def test_report_structure(self):
        raster = from_numpy(np.zeros((5, 5), dtype=np.float32), nodata=-1.0)
        report = raster.diagnostics_report()
        assert report["shape"] == (5, 5)
        assert report["dtype"] == "float32"
        assert report["nodata"] == -1.0
        assert report["residency"] == "host"
        assert not report["device_allocated"]
        assert len(report["events"]) == 1

    def test_repr(self):
        raster = from_numpy(np.zeros((5, 5), dtype=np.float32))
        r = repr(raster)
        assert "OwnedRasterArray" in r
        assert "float32" in r


# ---------------------------------------------------------------------------
# from_device factory
# ---------------------------------------------------------------------------


class TestFromDevice:
    @requires_gpu
    def test_from_device_basic(self):
        import cupy as cp

        device_arr = cp.ones((4, 5), dtype=np.float32)
        raster = from_device(device_arr)
        assert raster.shape == (4, 5)
        assert raster.dtype == np.float32
        assert raster.residency is Residency.DEVICE

    @requires_gpu
    def test_from_device_to_numpy(self):
        import cupy as cp

        device_arr = cp.arange(20, dtype=np.float64).reshape(4, 5)
        raster = from_device(device_arr)
        result = raster.to_numpy()
        expected = cp.asnumpy(device_arr)
        np.testing.assert_array_equal(result, expected)

    @requires_gpu
    def test_from_device_3d(self):
        import cupy as cp

        device_arr = cp.zeros((3, 4, 5), dtype=np.float32)
        raster = from_device(device_arr)
        assert raster.band_count == 3
        assert raster.height == 4
        assert raster.width == 5
        assert raster.shape == (3, 4, 5)

    @requires_gpu
    def test_from_device_preserves_metadata(self):
        import cupy as cp
        from pyproj import CRS

        device_arr = cp.ones((10, 20), dtype=np.float32)
        crs = CRS.from_epsg(4326)
        affine = (0.5, 0.0, 10.0, 0.0, -0.5, 60.0)
        raster = from_device(device_arr, nodata=-9999.0, affine=affine, crs=crs)
        assert raster.nodata == -9999.0
        assert raster.affine == affine
        assert raster.crs == crs

    @requires_gpu
    def test_from_device_no_host_copy(self):
        import cupy as cp

        device_arr = cp.ones((4, 5), dtype=np.float32)
        raster = from_device(device_arr)
        assert raster._host_materialized is False
        raster.to_numpy()
        assert raster._host_materialized is True

    @requires_gpu
    def test_from_device_rejects_1d(self):
        import cupy as cp

        with pytest.raises(ValueError, match="2D or 3D"):
            from_device(cp.zeros(10))

    @requires_gpu
    def test_from_device_rejects_4d(self):
        import cupy as cp

        with pytest.raises(ValueError, match="2D or 3D"):
            from_device(cp.zeros((2, 3, 4, 5)))

    @requires_gpu
    def test_from_device_diagnostic(self):
        import cupy as cp

        device_arr = cp.ones((4, 5), dtype=np.float32)
        raster = from_device(device_arr)
        assert len(raster.diagnostics) == 1
        event = raster.diagnostics[0]
        assert event.kind == RasterDiagnosticKind.CREATED
        assert "from_device" in event.detail
        assert event.residency is Residency.DEVICE
