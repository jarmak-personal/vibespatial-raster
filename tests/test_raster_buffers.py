"""Tests for OwnedRasterArray, raster buffer schema, and interop specs."""

from __future__ import annotations

import unittest.mock

import numpy as np
import pytest

from vibespatial.raster.buffers import (
    GridSpec,
    OwnedRasterArray,
    PolygonizeSpec,
    RasterDeviceState,
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
# Nodata mask on device-resident rasters (bug #2 regression tests)
# ---------------------------------------------------------------------------


class TestNodataMaskDeviceResident:
    """Verify nodata_mask materializes host data before computing the mask.

    Rasters created via from_device() have uninitialized host memory
    (np.empty). Without the fix in nodata_mask, accessing the property
    returns a garbage mask derived from that uninitialized memory.
    """

    @requires_gpu
    def test_integer_nodata_from_device(self):
        """nodata_mask on device-resident raster with integer nodata."""
        import cupy as cp

        data_np = np.array([[1, -9999], [3, -9999]], dtype=np.int32)
        device_arr = cp.asarray(data_np)
        raster = from_device(device_arr, nodata=-9999)

        # Raster is device-resident with uninitialized host placeholder
        assert raster.residency is Residency.DEVICE
        assert raster._host_materialized is False

        mask = raster.nodata_mask
        expected = np.array([[False, True], [False, True]])
        np.testing.assert_array_equal(mask, expected)

    @requires_gpu
    def test_float_nodata_from_device(self):
        """nodata_mask on device-resident raster with float nodata sentinel."""
        import cupy as cp

        data_np = np.array([[1.0, -9999.0], [3.0, -9999.0]], dtype=np.float32)
        device_arr = cp.asarray(data_np)
        raster = from_device(device_arr, nodata=-9999.0)

        assert raster._host_materialized is False

        mask = raster.nodata_mask
        assert mask[0, 1]
        assert mask[1, 1]
        assert not mask[0, 0]
        assert not mask[1, 0]

    @requires_gpu
    def test_nan_nodata_from_device(self):
        """nodata_mask on device-resident raster with NaN nodata."""
        import cupy as cp

        data_np = np.array([[1.0, np.nan], [np.nan, 4.0]], dtype=np.float64)
        device_arr = cp.asarray(data_np)
        raster = from_device(device_arr, nodata=np.nan)

        assert raster._host_materialized is False

        mask = raster.nodata_mask
        expected = np.array([[False, True], [True, False]])
        np.testing.assert_array_equal(mask, expected)

    @requires_gpu
    def test_no_nodata_from_device(self):
        """nodata_mask on device-resident raster with nodata=None returns all-False."""
        import cupy as cp

        device_arr = cp.ones((3, 4), dtype=np.float32)
        raster = from_device(device_arr)

        assert raster._host_materialized is False

        mask = raster.nodata_mask
        assert not mask.any()
        assert mask.shape == (3, 4)

    @requires_gpu
    def test_valid_mask_from_device(self):
        """valid_mask delegates to nodata_mask, so it also must be correct."""
        import cupy as cp

        data_np = np.array([[1, -9999, 3], [-9999, 5, -9999]], dtype=np.int32)
        device_arr = cp.asarray(data_np)
        raster = from_device(device_arr, nodata=-9999)

        assert raster._host_materialized is False

        valid = raster.valid_mask
        expected = np.array([[True, False, True], [False, True, False]])
        np.testing.assert_array_equal(valid, expected)

    @requires_gpu
    def test_nodata_mask_does_not_change_residency(self):
        """Accessing nodata_mask should not flip residency to HOST."""
        import cupy as cp

        device_arr = cp.ones((4, 5), dtype=np.float32)
        raster = from_device(device_arr, nodata=-1.0)

        _ = raster.nodata_mask
        # _ensure_host_state syncs host data but should not change
        # the declared residency (the raster is still conceptually
        # device-resident -- the host sync is an internal detail).
        assert raster.residency is Residency.DEVICE

    @requires_gpu
    def test_nodata_mask_host_materialized_after_access(self):
        """After nodata_mask access, host data should be materialized."""
        import cupy as cp

        data_np = np.array([[10, 20], [30, 40]], dtype=np.int16)
        device_arr = cp.asarray(data_np)
        raster = from_device(device_arr, nodata=20)

        assert raster._host_materialized is False
        _ = raster.nodata_mask
        assert raster._host_materialized is True
        # Subsequent to_numpy should return correct data without another transfer
        np.testing.assert_array_equal(raster.to_numpy(), data_np)


# ---------------------------------------------------------------------------
# _ensure_host_state raises when CuPy unavailable (bug #3 regression tests)
# ---------------------------------------------------------------------------


class TestEnsureHostStateCupyUnavailable:
    """Verify _ensure_host_state raises RuntimeError when device_state exists
    but CuPy cannot be imported.

    Before the fix, _ensure_host_state silently returned without materializing
    host data, causing to_numpy() and nodata_mask to return garbage from the
    uninitialized np.empty() placeholder created by from_device().
    """

    @staticmethod
    def _make_device_resident_raster() -> OwnedRasterArray:
        """Build an OwnedRasterArray that mimics from_device() state.

        We construct it manually so these tests run WITHOUT CuPy installed.
        The key properties:
        - device_state is not None (a mock standing in for CuPy data)
        - _host_materialized is False
        - data is np.empty() (uninitialized placeholder)
        """
        placeholder = np.empty((4, 5), dtype=np.float32)
        mock_device_data = unittest.mock.MagicMock()
        return OwnedRasterArray(
            data=placeholder,
            nodata=-9999.0,
            dtype=np.dtype(np.float32),
            affine=(1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
            crs=None,
            residency=Residency.DEVICE,
            device_state=RasterDeviceState(data=mock_device_data),
            _host_materialized=False,
            diagnostics=[],
        )

    def test_ensure_host_state_raises_when_cupy_unavailable(self):
        """_ensure_host_state must raise RuntimeError, not silently return."""
        raster = self._make_device_resident_raster()

        with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
            with pytest.raises(RuntimeError, match="CuPy is not available"):
                raster._ensure_host_state()

    def test_to_numpy_raises_when_cupy_unavailable(self):
        """to_numpy delegates to _ensure_host_state, so it must also raise."""
        raster = self._make_device_resident_raster()

        with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
            with pytest.raises(RuntimeError, match="CuPy is not available"):
                raster.to_numpy()

    def test_nodata_mask_raises_when_cupy_unavailable(self):
        """nodata_mask calls _ensure_host_state, so it must also raise."""
        raster = self._make_device_resident_raster()

        with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
            with pytest.raises(RuntimeError, match="CuPy is not available"):
                _ = raster.nodata_mask

    def test_host_materialized_raster_unaffected(self):
        """If _host_materialized is True, _ensure_host_state short-circuits.

        Even with CuPy mocked as unavailable, no error should be raised
        because the host data is already authoritative.
        """
        raster = from_numpy(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            nodata=-1.0,
        )
        assert raster._host_materialized is True

        with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
            # Should NOT raise -- host data is already valid
            raster._ensure_host_state()
            result = raster.to_numpy()
            np.testing.assert_array_equal(result, np.arange(20, dtype=np.float32).reshape(4, 5))

    def test_no_device_state_unaffected(self):
        """If device_state is None, _ensure_host_state short-circuits.

        This covers the path where a HOST-resident raster never went to device.
        """
        raster = from_numpy(np.zeros((3, 3), dtype=np.float64))
        raster._host_materialized = False  # force non-materialized for coverage
        raster.device_state = None

        with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
            # Should NOT raise -- no device_state means nothing to transfer
            raster._ensure_host_state()

    def test_error_message_includes_install_hint(self):
        """Error message should include an actionable install hint."""
        raster = self._make_device_resident_raster()

        with unittest.mock.patch.dict("sys.modules", {"cupy": None}):
            with pytest.raises(RuntimeError, match="pip install cupy"):
                raster._ensure_host_state()


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


# ---------------------------------------------------------------------------
# device_band
# ---------------------------------------------------------------------------


class TestDeviceBand:
    """Tests for OwnedRasterArray.device_band() zero-copy band slicing."""

    @requires_gpu
    def test_device_band_single_band(self):
        """Index 0 returns the 2D array; index 1 raises IndexError."""
        import cupy as cp

        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        raster = from_numpy(data)

        band0 = raster.device_band(0)
        assert band0.shape == (3, 4)
        np.testing.assert_array_equal(cp.asnumpy(band0), data)

        with pytest.raises(IndexError, match="single-band raster"):
            raster.device_band(1)

    @requires_gpu
    def test_device_band_multiband(self):
        """Each band returns a correct (H, W) view from a (B, H, W) raster."""
        import cupy as cp

        rng = np.random.default_rng(42)
        data = rng.random((3, 10, 20), dtype=np.float64)
        raster = from_numpy(data)

        for b in range(3):
            band = raster.device_band(b)
            assert band.shape == (10, 20)
            np.testing.assert_array_equal(cp.asnumpy(band), data[b])

    @requires_gpu
    def test_device_band_out_of_range_raises(self):
        """Negative and too-large indices raise IndexError."""
        data = np.zeros((3, 4, 5), dtype=np.float32)
        raster = from_numpy(data)

        with pytest.raises(IndexError, match="out of range"):
            raster.device_band(3)

        with pytest.raises(IndexError, match="out of range"):
            raster.device_band(-1)

    @requires_gpu
    def test_device_band_is_zero_copy(self):
        """The returned band view shares memory with the full device array."""
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        raster = from_numpy(data)

        full = raster.device_data()
        band0 = raster.device_band(0)
        band1 = raster.device_band(1)

        # Views share the same base memory (CuPy .base or .data.ptr check)
        assert band0.data.ptr == full.data.ptr
        assert band1.data.ptr == full[1].data.ptr


# ---------------------------------------------------------------------------
# device_band -- CPU-only tests (no GPU required)
# ---------------------------------------------------------------------------


class TestDeviceBandCPUOnly:
    """Tests for device_band that work without a GPU by mocking device_data."""

    def test_device_band_single_band_index_0(self):
        """Index 0 on a 2D mock returns the array unchanged."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        raster = from_numpy(data)

        # Mock device_data to return a numpy array (standing in for CuPy)
        raster.device_data = lambda: data

        result = raster.device_band(0)
        assert result is data

    def test_device_band_single_band_index_1_raises(self):
        """Index 1 on a single-band raster raises IndexError."""
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        raster = from_numpy(data)
        raster.device_data = lambda: data

        with pytest.raises(IndexError, match="single-band raster"):
            raster.device_band(1)

    def test_device_band_multiband_returns_correct_shape(self):
        """Multi-band raster returns (H, W) slices."""
        data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        raster = from_numpy(data)
        raster.device_data = lambda: data

        for b in range(3):
            band = raster.device_band(b)
            assert band.shape == (4, 5)
            np.testing.assert_array_equal(band, data[b])

    def test_device_band_multiband_out_of_range(self):
        """Out-of-range indices on multi-band raster raise IndexError."""
        data = np.zeros((2, 3, 4), dtype=np.float32)
        raster = from_numpy(data)
        raster.device_data = lambda: data

        with pytest.raises(IndexError, match="out of range"):
            raster.device_band(2)

        with pytest.raises(IndexError, match="out of range"):
            raster.device_band(-1)


# ---------------------------------------------------------------------------
# from_band_stack
# ---------------------------------------------------------------------------


class TestFromBandStack:
    """Tests for OwnedRasterArray.from_band_stack() band assembly."""

    def _make_source(
        self,
        *,
        height: int = 10,
        width: int = 20,
        bands: int = 3,
        dtype=np.float32,
        nodata: float | int | None = -9999.0,
    ) -> OwnedRasterArray:
        """Create a source raster with known metadata."""
        shape = (bands, height, width) if bands > 1 else (height, width)
        data = np.zeros(shape, dtype=dtype)
        return from_numpy(
            data,
            nodata=nodata,
            affine=(0.5, 0.0, 10.0, 0.0, -0.5, 60.0),
        )

    def _make_band_result(
        self,
        *,
        height: int = 10,
        width: int = 20,
        dtype=np.float32,
        nodata: float | int | None = -9999.0,
        fill: float = 1.0,
    ) -> OwnedRasterArray:
        """Create a single-band result raster."""
        data = np.full((height, width), fill, dtype=dtype)
        return from_numpy(data, nodata=nodata)

    def test_from_band_stack_basic(self):
        """3 single-band results assemble into a (3, H, W) raster."""
        source = self._make_source()
        bands = [self._make_band_result(fill=float(i + 1)) for i in range(3)]

        result = OwnedRasterArray.from_band_stack(bands, source=source)

        assert result.band_count == 3
        assert result.height == 10
        assert result.width == 20
        assert result.dtype == np.float32
        assert result.nodata == source.nodata
        assert result.affine == source.affine
        assert result.crs == source.crs
        assert result.shape == (3, 10, 20)

        out = result.to_numpy()
        np.testing.assert_array_equal(out[0], np.full((10, 20), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(out[1], np.full((10, 20), 2.0, dtype=np.float32))
        np.testing.assert_array_equal(out[2], np.full((10, 20), 3.0, dtype=np.float32))

    def test_from_band_stack_dtype_mismatch_raises(self):
        """Mixed float32/float64 raises ValueError."""
        source = self._make_source()
        band_f32 = self._make_band_result(dtype=np.float32)
        band_f64 = self._make_band_result(dtype=np.float64)

        with pytest.raises(ValueError, match="dtype mismatch"):
            OwnedRasterArray.from_band_stack([band_f32, band_f64], source=source)

    def test_from_band_stack_nodata_mismatch_raises(self):
        """Different nodata values raise ValueError."""
        source = self._make_source()
        band_a = self._make_band_result(nodata=-9999.0)
        band_b = self._make_band_result(nodata=-1.0)

        with pytest.raises(ValueError, match="nodata mismatch"):
            OwnedRasterArray.from_band_stack([band_a, band_b], source=source)

    def test_from_band_stack_nodata_none_vs_value_raises(self):
        """None nodata vs non-None nodata raises ValueError."""
        source = self._make_source()
        band_none = self._make_band_result(nodata=None)
        band_val = self._make_band_result(nodata=-9999.0)

        with pytest.raises(ValueError, match="nodata mismatch"):
            OwnedRasterArray.from_band_stack([band_none, band_val], source=source)

        with pytest.raises(ValueError, match="nodata mismatch"):
            OwnedRasterArray.from_band_stack([band_val, band_none], source=source)

    def test_from_band_stack_shape_mismatch_raises(self):
        """Different spatial dimensions raise ValueError."""
        source = self._make_source()
        band_a = self._make_band_result(height=10, width=20)
        band_b = self._make_band_result(height=10, width=30)

        with pytest.raises(ValueError, match="shape mismatch"):
            OwnedRasterArray.from_band_stack([band_a, band_b], source=source)

    def test_from_band_stack_single_band_passthrough(self):
        """Single result is returned without stacking (zero overhead)."""
        source = self._make_source()
        band = self._make_band_result(fill=42.0)

        result = OwnedRasterArray.from_band_stack([band], source=source)

        # Should be the same object, not a copy
        assert result is band
        # Metadata propagated from source
        assert result.affine == source.affine
        assert result.crs == source.crs
        assert result.nodata == source.nodata
        # Still single-band
        assert result.band_count == 1
        assert result.shape == (10, 20)

    def test_from_band_stack_empty_raises(self):
        """Empty band_results raises ValueError."""
        source = self._make_source()

        with pytest.raises(ValueError, match="must not be empty"):
            OwnedRasterArray.from_band_stack([], source=source)

    def test_from_band_stack_preserves_dtype(self):
        """Output dtype matches input band dtype."""
        source = self._make_source(dtype=np.int16, nodata=-1)
        bands = [self._make_band_result(dtype=np.int16, nodata=-1, fill=float(i)) for i in range(2)]

        result = OwnedRasterArray.from_band_stack(bands, source=source)
        assert result.dtype == np.int16

    def test_from_band_stack_merges_diagnostics(self):
        """Diagnostics from all band results are merged into the output."""
        source = self._make_source()
        bands = [self._make_band_result(fill=float(i + 1)) for i in range(3)]

        result = OwnedRasterArray.from_band_stack(bands, source=source)

        # Each band has a CREATED event from from_numpy, plus the result's
        # own CREATED event from from_numpy
        created_events = [e for e in result.diagnostics if e.kind == RasterDiagnosticKind.CREATED]
        # 3 from bands + 1 from the final from_numpy = 4
        assert len(created_events) == 4

    def test_from_band_stack_nan_nodata_accepted(self):
        """Two bands with NaN nodata do not raise (NaN == NaN is handled)."""
        source = self._make_source(nodata=np.nan, dtype=np.float64)
        band_a = self._make_band_result(dtype=np.float64, nodata=np.nan, fill=1.0)
        band_b = self._make_band_result(dtype=np.float64, nodata=np.nan, fill=2.0)

        # Should not raise
        result = OwnedRasterArray.from_band_stack([band_a, band_b], source=source)
        assert result.band_count == 2

    @requires_gpu
    def test_from_band_stack_device_resident(self):
        """Band stacking works when band results are on device."""
        import cupy as cp

        source = self._make_source()
        bands = []
        for i in range(3):
            d = cp.full((10, 20), float(i + 1), dtype=np.float32)
            bands.append(from_device(d, nodata=-9999.0))

        result = OwnedRasterArray.from_band_stack(bands, source=source)
        assert result.band_count == 3
        assert result.residency is Residency.DEVICE

        out = result.to_numpy()
        np.testing.assert_array_equal(out[0], np.full((10, 20), 1.0, dtype=np.float32))
        np.testing.assert_array_equal(out[2], np.full((10, 20), 3.0, dtype=np.float32))
