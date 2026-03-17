"""Tests for nvImageCodec GPU-native raster IO."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from vibespatial.raster.nvimgcodec_io import has_nvimgcodec_support

    HAS_NVIMGCODEC = has_nvimgcodec_support()
except ImportError:
    HAS_NVIMGCODEC = False

try:
    from vibespatial.raster.io import has_rasterio_support

    HAS_RASTERIO = has_rasterio_support()
except ImportError:
    HAS_RASTERIO = False

requires_nvimgcodec = pytest.mark.skipif(not HAS_NVIMGCODEC, reason="nvImageCodec not available")

requires_nvimgcodec_and_rasterio = pytest.mark.skipif(
    not (HAS_NVIMGCODEC and HAS_RASTERIO),
    reason="nvImageCodec and/or rasterio not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_geotiff_single(tmp_path):
    """Create a single-band GeoTIFF for nvimgcodec testing."""
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    path = tmp_path / "single_band.tif"
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
        crs=CRS.from_epsg(4326),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    return path, data, transform


@pytest.fixture
def tmp_geotiff_multi(tmp_path):
    """Create a 3-band GeoTIFF for nvimgcodec testing."""
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    path = tmp_path / "multi_band.tif"
    data = np.random.default_rng(42).random((3, 10, 15)).astype(np.float32)
    transform = from_bounds(100.0, 200.0, 115.0, 210.0, 15, 10)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        width=15,
        height=10,
        count=3,
        transform=transform,
        crs=CRS.from_epsg(32633),
    ) as dst:
        dst.write(data)

    return path, data, transform


# ---------------------------------------------------------------------------
# Basic tests (always run -- no nvimgcodec needed)
# ---------------------------------------------------------------------------


class TestHasNvimgcodecSupport:
    def test_returns_bool(self):
        from vibespatial.raster.nvimgcodec_io import has_nvimgcodec_support

        result = has_nvimgcodec_support()
        assert isinstance(result, bool)


class TestUnsupportedInputs:
    @requires_nvimgcodec
    def test_unsupported_format_returns_none(self, tmp_path):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        txt_file = tmp_path / "not_a_raster.txt"
        txt_file.write_text("hello world")
        result = nvimgcodec_read(str(txt_file))
        assert result is None

    @requires_nvimgcodec
    def test_overview_returns_none(self, tmp_path):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        # overview_level != None should immediately return None
        result = nvimgcodec_read(str(tmp_path / "any_file.tif"), overview_level=1)
        assert result is None


# ---------------------------------------------------------------------------
# Tests that require nvimgcodec + rasterio (to create test fixtures)
# ---------------------------------------------------------------------------


class TestNvimgcodecReadMetadata:
    @requires_nvimgcodec_and_rasterio
    def test_read_metadata_geotiff(self, tmp_geotiff_single):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read_metadata

        path, data, _ = tmp_geotiff_single
        meta = nvimgcodec_read_metadata(path)
        assert meta is not None
        assert meta.height == 4
        assert meta.width == 5
        assert meta.dtype == np.dtype("float32")
        assert meta.driver == "nvimgcodec"

    @requires_nvimgcodec_and_rasterio
    def test_metadata_affine_present(self, tmp_geotiff_single):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read_metadata

        path, _, _ = tmp_geotiff_single
        meta = nvimgcodec_read_metadata(path)
        assert meta is not None
        assert meta.affine is not None
        assert len(meta.affine) == 6


class TestNvimgcodecRead:
    @requires_nvimgcodec_and_rasterio
    def test_read_to_device(self, tmp_geotiff_single):
        import cupy as cp

        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        path, _, _ = tmp_geotiff_single
        result = nvimgcodec_read(path)
        assert result is not None
        arr, meta = result
        # Verify the result is a CuPy array on the GPU.
        assert isinstance(arr, cp.ndarray)

    @requires_nvimgcodec_and_rasterio
    def test_read_shape_single_band(self, tmp_geotiff_single):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        path, data, _ = tmp_geotiff_single
        result = nvimgcodec_read(path)
        assert result is not None
        arr, meta = result
        # Single-band should be squeezed to 2D (H, W).
        assert arr.ndim == 2
        assert arr.shape == (4, 5)
        assert meta.band_count == 1

    @requires_nvimgcodec_and_rasterio
    def test_read_shape_multi_band(self, tmp_geotiff_multi):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        path, data, _ = tmp_geotiff_multi
        result = nvimgcodec_read(path)
        assert result is not None
        arr, meta = result
        # Multi-band should be (C, H, W).
        assert arr.ndim == 3
        assert arr.shape == (3, 10, 15)
        assert meta.band_count == 3

    @requires_nvimgcodec_and_rasterio
    def test_read_windowed(self, tmp_geotiff_single):
        from vibespatial.raster.buffers import RasterWindow
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        path, _, _ = tmp_geotiff_single
        window = RasterWindow(col_off=1, row_off=1, width=3, height=2)
        result = nvimgcodec_read(path, window=window)
        assert result is not None
        arr, meta = result
        assert meta.height == 2
        assert meta.width == 3

    @requires_nvimgcodec_and_rasterio
    def test_read_band_selection(self, tmp_geotiff_multi):
        from vibespatial.raster.nvimgcodec_io import nvimgcodec_read

        path, data, _ = tmp_geotiff_multi
        # Select band 2 (1-indexed).
        result = nvimgcodec_read(path, bands=[2])
        assert result is not None
        arr, meta = result
        # Selecting a single band should squeeze to 2D.
        assert arr.ndim == 2
        assert meta.band_count == 1
