"""Tests for raster IO: read/write GeoTIFF via rasterio."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.io import has_nvimgcodec_support, has_rasterio_support

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")
requires_nvimgcodec = pytest.mark.skipif(
    not has_nvimgcodec_support() or not HAS_GPU,
    reason="nvImageCodec or CuPy not available",
)

pytestmark = pytest.mark.skipif(
    not has_rasterio_support(),
    reason="rasterio not installed",
)


@pytest.fixture
def tmp_geotiff(tmp_path):
    """Create a small GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "test.tif"
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    transform = from_bounds(0.0, 0.0, 5.0, 4.0, 5, 4)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        width=5,
        height=4,
        count=1,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return path, data, transform


@pytest.fixture
def tmp_multiband_geotiff(tmp_path):
    """Create a multi-band GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "multiband.tif"
    data = np.random.default_rng(42).random((3, 10, 15)).astype(np.float64)
    transform = from_bounds(100.0, 200.0, 115.0, 210.0, 15, 10)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float64",
        width=15,
        height=10,
        count=3,
        transform=transform,
    ) as dst:
        dst.write(data)

    return path, data, transform


class TestHasRasterioSupport:
    def test_available(self):
        assert has_rasterio_support()


class TestReadRasterMetadata:
    def test_metadata(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster_metadata

        path, data, transform = tmp_geotiff
        meta = read_raster_metadata(path)
        assert meta.height == 4
        assert meta.width == 5
        assert meta.band_count == 1
        assert meta.dtype == np.float32
        assert meta.nodata == -9999.0
        assert meta.driver == "GTiff"
        assert meta.pixel_count == 20

    def test_multiband_metadata(self, tmp_multiband_geotiff):
        from vibespatial.raster.io import read_raster_metadata

        path, data, transform = tmp_multiband_geotiff
        meta = read_raster_metadata(path)
        assert meta.band_count == 3
        assert meta.height == 10
        assert meta.width == 15


class TestReadRaster:
    def test_full_read(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, expected_data, _ = tmp_geotiff
        raster = read_raster(path)
        assert raster.residency is Residency.HOST
        assert raster.height == 4
        assert raster.width == 5
        assert raster.band_count == 1
        assert raster.dtype == np.float32
        assert raster.nodata == -9999.0
        np.testing.assert_array_equal(raster.to_numpy(), expected_data)

    def test_multiband_read(self, tmp_multiband_geotiff):
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_multiband_geotiff
        raster = read_raster(path)
        assert raster.band_count == 3
        assert raster.shape == (3, 10, 15)
        np.testing.assert_array_almost_equal(raster.to_numpy(), expected_data)

    def test_select_bands(self, tmp_multiband_geotiff):
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_multiband_geotiff
        raster = read_raster(path, bands=[2])
        assert raster.band_count == 1
        assert raster.shape == (10, 15)  # single band squeezed to 2D
        np.testing.assert_array_almost_equal(raster.to_numpy(), expected_data[1])

    def test_windowed_read(self, tmp_geotiff):
        from vibespatial.raster.buffers import RasterWindow
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_geotiff
        window = RasterWindow(col_off=1, row_off=1, width=3, height=2)
        raster = read_raster(path, window=window)
        assert raster.height == 2
        assert raster.width == 3
        np.testing.assert_array_equal(raster.to_numpy(), expected_data[1:3, 1:4])

    def test_affine_preserved(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster

        path, _, transform = tmp_geotiff
        raster = read_raster(path)
        assert raster.affine[0] == pytest.approx(transform.a)
        assert raster.affine[4] == pytest.approx(transform.e)

    @gpu
    def test_read_to_device(self, tmp_geotiff):
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, _, _ = tmp_geotiff
        raster = read_raster(path, residency=Residency.DEVICE)
        assert raster.residency is Residency.DEVICE
        assert raster.device_state is not None


class TestWriteRaster:
    def test_roundtrip(self, tmp_path, tmp_geotiff):
        from vibespatial.raster.io import read_raster, write_raster

        src_path, expected_data, _ = tmp_geotiff
        raster = read_raster(src_path)

        out_path = tmp_path / "output.tif"
        write_raster(out_path, raster)

        roundtrip = read_raster(out_path)
        np.testing.assert_array_equal(roundtrip.to_numpy(), expected_data)
        assert roundtrip.nodata == raster.nodata
        assert roundtrip.dtype == raster.dtype

    def test_write_records_diagnostic(self, tmp_path, tmp_geotiff):
        from vibespatial.raster.buffers import RasterDiagnosticKind
        from vibespatial.raster.io import read_raster, write_raster

        src_path, _, _ = tmp_geotiff
        raster = read_raster(src_path)
        initial_count = len(raster.diagnostics)

        out_path = tmp_path / "output.tif"
        write_raster(out_path, raster)

        assert len(raster.diagnostics) == initial_count + 1
        assert raster.diagnostics[-1].kind == RasterDiagnosticKind.MATERIALIZATION


class TestDecodeBackendDispatch:
    """Tests for the decode_backend parameter and GPU/rasterio dispatch."""

    def test_read_raster_explicit_rasterio_backend(self, tmp_geotiff):
        """decode_backend='rasterio' works and skips nvimgcodec entirely."""
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, expected_data, _ = tmp_geotiff
        raster = read_raster(path, decode_backend="rasterio")
        assert raster.residency is Residency.HOST
        assert raster.height == 4
        assert raster.width == 5
        np.testing.assert_array_equal(raster.to_numpy(), expected_data)

    def test_read_raster_auto_falls_back_to_rasterio(self, tmp_geotiff):
        """With nvimgcodec unavailable, auto backend still works via rasterio."""
        from vibespatial.raster.io import read_raster

        path, expected_data, _ = tmp_geotiff
        # auto is the default -- on CI without nvimgcodec this must still work
        raster = read_raster(path, decode_backend="auto")
        np.testing.assert_array_equal(raster.to_numpy(), expected_data)

    def test_read_raster_backend_diagnostic(self, tmp_geotiff):
        """Verify diagnostic event says 'backend=rasterio'."""
        from vibespatial.raster.buffers import RasterDiagnosticKind
        from vibespatial.raster.io import read_raster

        path, _, _ = tmp_geotiff
        raster = read_raster(path, decode_backend="rasterio")
        runtime_events = [e for e in raster.diagnostics if e.kind == RasterDiagnosticKind.RUNTIME]
        assert len(runtime_events) >= 1
        assert "backend=rasterio" in runtime_events[-1].detail

    def test_has_nvimgcodec_support_returns_bool(self):
        """Verify the public API function returns a bool."""
        result = has_nvimgcodec_support()
        assert isinstance(result, bool)


class TestNvimgcodecBackend:
    """GPU-dependent tests for nvImageCodec decode path."""

    @requires_nvimgcodec
    def test_read_raster_nvimgcodec_device(self, tmp_geotiff):
        """Verify device-resident result when nvimgcodec available."""
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, _, _ = tmp_geotiff
        raster = read_raster(path, residency=Residency.DEVICE, decode_backend="nvimgcodec")
        assert raster.residency is Residency.DEVICE
        assert raster.device_state is not None
        runtime_events = [e for e in raster.diagnostics if "backend=nvimgcodec" in e.detail]
        assert len(runtime_events) >= 1

    @requires_nvimgcodec
    def test_read_raster_nvimgcodec_to_host(self, tmp_geotiff):
        """Verify HOST residency with nvimgcodec (auto D->H transfer)."""
        from vibespatial.raster.io import read_raster
        from vibespatial.residency import Residency

        path, expected_data, _ = tmp_geotiff
        raster = read_raster(path, residency=Residency.HOST, decode_backend="nvimgcodec")
        assert raster.residency is Residency.HOST
        # Data should be accessible as numpy
        host_data = raster.to_numpy()
        assert host_data.shape == expected_data.shape
