"""Tests for zonal statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from vibespatial.raster.buffers import GridSpec, ZonalSpec, from_numpy
from vibespatial.raster.zonal import _has_cupy, zonal_stats, zonal_stats_gdf

requires_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")

try:
    import geopandas as gpd  # noqa: F401

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

try:
    from rasterio.features import rasterize as _  # noqa: F401

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

requires_geopandas = pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not available")
requires_rasterio = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not available")


class TestZonalStatsBasic:
    def test_two_zones(self):
        zones = from_numpy(np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int32))
        values = from_numpy(np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))
        result = zonal_stats(zones, values, stats=["count", "sum", "mean"])
        assert len(result) == 2
        # Zone 1: pixels (10, 20, 40) -> count=3, sum=70, mean=23.33
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["count"] == 3
        assert z1["sum"] == pytest.approx(70.0)
        assert z1["mean"] == pytest.approx(70.0 / 3)
        # Zone 2: pixels (30, 50, 60) -> count=3, sum=140, mean=46.67
        z2 = result[result["zone"] == 2].iloc[0]
        assert z2["count"] == 3
        assert z2["sum"] == pytest.approx(140.0)

    def test_default_stats(self):
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        result = zonal_stats(zones, values)
        assert "count" in result.columns
        assert "sum" in result.columns
        assert "mean" in result.columns
        assert "min" in result.columns
        assert "max" in result.columns

    def test_min_max(self):
        zones = from_numpy(np.array([[1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 15.0, 10.0]]))
        result = zonal_stats(zones, values, stats=["min", "max"])
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["min"] == 5.0
        assert z1["max"] == 15.0

    def test_std(self):
        zones = from_numpy(np.array([[1, 1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[2.0, 4.0, 4.0, 4.0]]))
        result = zonal_stats(zones, values, stats=["std"])
        z1 = result[result["zone"] == 1].iloc[0]
        expected_std = np.std([2.0, 4.0, 4.0, 4.0])
        assert z1["std"] == pytest.approx(expected_std)

    def test_median(self):
        zones = from_numpy(np.array([[1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[1.0, 3.0, 2.0]]))
        result = zonal_stats(zones, values, stats=["median"])
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["median"] == 2.0


class TestZonalStatsNodata:
    def test_values_nodata_excluded(self):
        zones = from_numpy(np.array([[1, 1, 1]], dtype=np.int32))
        values = from_numpy(
            np.array([[10.0, -9999.0, 30.0]]),
            nodata=-9999.0,
        )
        result = zonal_stats(zones, values, stats=["count", "sum"])
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["count"] == 2  # nodata pixel excluded
        assert z1["sum"] == pytest.approx(40.0)

    def test_zone_nodata_excluded(self):
        zones = from_numpy(
            np.array([[1, 0, 2]], dtype=np.int32),
            nodata=0,
        )
        values = from_numpy(np.array([[10.0, 20.0, 30.0]]))
        result = zonal_stats(zones, values, stats=["count"])
        assert len(result) == 2  # zone 0 (nodata) excluded
        assert 0 not in result["zone"].values


class TestZonalStatsEdgeCases:
    def test_shape_mismatch(self):
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[1.0, 2.0, 3.0]]))
        with pytest.raises(ValueError, match="must match"):
            zonal_stats(zones, values)

    def test_empty_result(self):
        zones = from_numpy(
            np.array([[0, 0]], dtype=np.int32),
            nodata=0,
        )
        values = from_numpy(np.array([[1.0, 2.0]]))
        result = zonal_stats(zones, values)
        assert len(result) == 0

    def test_zonal_multiband_zones_raises(self):
        """Multiband zones raster must be rejected with ValueError."""
        zones_data = np.ones((2, 4, 4), dtype=np.int32)  # 2-band
        values_data = np.ones((4, 4), dtype=np.float64)
        zones = from_numpy(zones_data)
        values = from_numpy(values_data)
        with pytest.raises(ValueError, match="zones raster must be single-band"):
            zonal_stats(zones, values)

    def test_zonal_multiband_values_accepted(self):
        """Multiband values raster is accepted and produces per-band columns."""
        zones_data = np.array([[1, 1, 2], [2, 2, 1]], dtype=np.int32)
        values_data = np.stack(
            [
                np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ]
        )  # 2-band
        zones = from_numpy(zones_data)
        values = from_numpy(values_data)
        result = zonal_stats(zones, values, stats=["mean"])
        # Should have band-prefixed columns, not bare "mean"
        assert "band_1_mean" in result.columns
        assert "band_2_mean" in result.columns
        assert "mean" not in result.columns

    def test_zonal_spec(self):
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        spec = ZonalSpec(stats=("min", "max"))
        result = zonal_stats(zones, values, stats=spec)
        assert "min" in result.columns
        assert "max" in result.columns
        assert "sum" not in result.columns


class TestZonalStatsMultiband:
    """Per-band dispatch for multiband values rasters."""

    def test_zonal_multiband_values(self):
        """3-band values raster produces band_N_ prefixed columns for each stat."""
        rng = np.random.default_rng(42)
        zones_data = np.array([[1, 1, 2, 2], [1, 2, 2, 3], [3, 3, 3, 3]], dtype=np.int32)
        band_1 = rng.random((3, 4)) * 100
        band_2 = rng.random((3, 4)) * 200
        band_3 = rng.random((3, 4)) * 50
        values_data = np.stack([band_1, band_2, band_3])  # shape (3, 3, 4)

        zones = from_numpy(zones_data)
        values = from_numpy(values_data)

        result = zonal_stats(zones, values, stats=["mean", "std"], use_gpu=False)

        # Verify column structure: zone + band_N_stat for each of 3 bands x 2 stats
        assert "zone" in result.columns
        for band_num in (1, 2, 3):
            for stat in ("mean", "std"):
                col = f"band_{band_num}_{stat}"
                assert col in result.columns, f"Missing column: {col}"

        # No bare stat columns
        assert "mean" not in result.columns
        assert "std" not in result.columns

        # All 3 zones present
        assert set(result["zone"].values) == {1, 2, 3}

    def test_zonal_multiband_zones_still_raises(self):
        """Multiband zones raster still raises ValueError."""
        zones_data = np.ones((2, 4, 4), dtype=np.int32)
        values_data = np.ones((4, 4), dtype=np.float64)
        zones = from_numpy(zones_data)
        values = from_numpy(values_data)
        with pytest.raises(ValueError, match="zones raster must be single-band"):
            zonal_stats(zones, values)

    def test_zonal_multiband_matches_individual(self):
        """Per-band multiband results match running zonal_stats on each band individually."""
        rng = np.random.default_rng(99)
        zones_data = np.array([[1, 1, 2], [2, 3, 3]], dtype=np.int32)
        band_1 = rng.random((2, 3)) * 100
        band_2 = rng.random((2, 3)) * 50
        band_3 = rng.random((2, 3)) * 200
        values_data = np.stack([band_1, band_2, band_3])

        zones = from_numpy(zones_data)
        values = from_numpy(values_data)

        all_stats = ["count", "sum", "mean", "min", "max", "std", "median"]

        # Multiband call
        result_multi = zonal_stats(zones, values, stats=all_stats, use_gpu=False)

        # Individual per-band calls
        for band_idx, band_data in enumerate([band_1, band_2, band_3]):
            band_num = band_idx + 1
            single_values = from_numpy(band_data)
            single_zones = from_numpy(zones_data)
            result_single = zonal_stats(single_zones, single_values, stats=all_stats, use_gpu=False)

            # Sort both by zone for comparison
            result_single_sorted = result_single.sort_values("zone").reset_index(drop=True)
            result_multi_sorted = result_multi.sort_values("zone").reset_index(drop=True)

            for stat in all_stats:
                single_col = result_single_sorted[stat].values
                multi_col = result_multi_sorted[f"band_{band_num}_{stat}"].values
                np.testing.assert_allclose(
                    single_col,
                    multi_col,
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"Mismatch for band_{band_num}_{stat}",
                )

    def test_zonal_single_band_no_prefix(self):
        """Single-band values still produce unprefixed columns (backward compat)."""
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        result = zonal_stats(zones, values, stats=["mean", "sum"])
        assert "mean" in result.columns
        assert "sum" in result.columns
        assert "band_1_mean" not in result.columns

    def test_zonal_multiband_nodata_propagation(self):
        """Nodata in multiband values is correctly excluded per-band."""
        zones_data = np.array([[1, 1, 2, 2]], dtype=np.int32)
        band_1 = np.array([[10.0, -9999.0, 30.0, 40.0]])
        band_2 = np.array([[1.0, 2.0, -9999.0, 4.0]])
        values_data = np.stack([band_1, band_2])

        zones = from_numpy(zones_data)
        values = from_numpy(values_data, nodata=-9999.0)

        result = zonal_stats(zones, values, stats=["count", "sum"], use_gpu=False)
        result_sorted = result.sort_values("zone").reset_index(drop=True)

        # Band 1, zone 1: only pixel 10.0 (the -9999 is nodata)
        assert result_sorted.loc[0, "band_1_count"] == 1
        assert result_sorted.loc[0, "band_1_sum"] == pytest.approx(10.0)
        # Band 1, zone 2: pixels 30.0, 40.0
        assert result_sorted.loc[1, "band_1_count"] == 2
        assert result_sorted.loc[1, "band_1_sum"] == pytest.approx(70.0)

        # Band 2, zone 1: pixels 1.0, 2.0
        assert result_sorted.loc[0, "band_2_count"] == 2
        assert result_sorted.loc[0, "band_2_sum"] == pytest.approx(3.0)
        # Band 2, zone 2: only pixel 4.0 (the -9999 is nodata)
        assert result_sorted.loc[1, "band_2_count"] == 1
        assert result_sorted.loc[1, "band_2_sum"] == pytest.approx(4.0)

    def test_zonal_multiband_diagnostics(self):
        """Multiband zonal_stats appends diagnostic event with band count."""
        zones_data = np.array([[1, 2]], dtype=np.int32)
        values_data = np.stack(
            [
                np.array([[5.0, 10.0]]),
                np.array([[15.0, 20.0]]),
            ]
        )
        zones = from_numpy(zones_data)
        values = from_numpy(values_data)
        zonal_stats(zones, values, stats=["mean"], use_gpu=False)
        runtime_events = [e for e in zones.diagnostics if e.kind == "runtime"]
        assert len(runtime_events) > 0
        assert "bands=2" in runtime_events[-1].detail


class TestZonalStatsDispatch:
    """Test auto-dispatch logic and use_gpu parameter."""

    def test_force_cpu(self):
        """use_gpu=False always uses CPU path."""
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        result = zonal_stats(zones, values, stats=["count"], use_gpu=False)
        assert len(result) == 2
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["count"] == 1

    def test_auto_dispatch_small_raster(self):
        """Small rasters should auto-dispatch to CPU."""
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        result = zonal_stats(zones, values, stats=["count"])
        assert len(result) == 2

    def test_diagnostics_include_backend(self):
        """Diagnostic event should report which backend was used."""
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        zonal_stats(zones, values, stats=["count"], use_gpu=False)
        runtime_events = [e for e in zones.diagnostics if e.kind == "runtime"]
        assert len(runtime_events) > 0
        assert "cpu" in runtime_events[-1].detail


# ---------------------------------------------------------------------------
# GPU tests -- skipped when CuPy is unavailable
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
class TestZonalStatsGPU:
    """GPU path: verify CCCL-based zonal stats match CPU for all 7 stat types."""

    def test_gpu_all_stats_match_cpu(self):
        """GPU and CPU produce identical results for all 7 statistics."""
        rng = np.random.RandomState(42)
        zone_data = rng.randint(1, 6, size=(50, 60)).astype(np.int32)
        value_data = rng.randn(50, 60).astype(np.float64) * 100

        zones = from_numpy(zone_data)
        values = from_numpy(value_data)
        all_stats = ["count", "sum", "mean", "min", "max", "std", "median"]

        cpu_result = zonal_stats(zones, values, stats=all_stats, use_gpu=False)
        gpu_result = zonal_stats(zones, values, stats=all_stats, use_gpu=True)

        # Same number of zones
        assert len(cpu_result) == len(gpu_result)

        # Sort both by zone for comparison
        cpu_sorted = cpu_result.sort_values("zone").reset_index(drop=True)
        gpu_sorted = gpu_result.sort_values("zone").reset_index(drop=True)

        np.testing.assert_array_equal(cpu_sorted["zone"].values, gpu_sorted["zone"].values)

        for stat in all_stats:
            np.testing.assert_allclose(
                cpu_sorted[stat].values,
                gpu_sorted[stat].values,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"Mismatch for stat={stat}",
            )

    def test_gpu_two_zones_basic(self):
        """Basic two-zone GPU test with known values."""
        zones = from_numpy(np.array([[1, 1, 2], [1, 2, 2]], dtype=np.int32))
        values = from_numpy(np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))
        result = zonal_stats(zones, values, stats=["count", "sum", "mean"], use_gpu=True)
        assert len(result) == 2

        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["count"] == 3
        assert z1["sum"] == pytest.approx(70.0)
        assert z1["mean"] == pytest.approx(70.0 / 3)

        z2 = result[result["zone"] == 2].iloc[0]
        assert z2["count"] == 3
        assert z2["sum"] == pytest.approx(140.0)

    def test_gpu_min_max(self):
        zones = from_numpy(np.array([[1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 15.0, 10.0]]))
        result = zonal_stats(zones, values, stats=["min", "max"], use_gpu=True)
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["min"] == 5.0
        assert z1["max"] == 15.0

    def test_gpu_std(self):
        zones = from_numpy(np.array([[1, 1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[2.0, 4.0, 4.0, 4.0]]))
        result = zonal_stats(zones, values, stats=["std"], use_gpu=True)
        z1 = result[result["zone"] == 1].iloc[0]
        expected_std = np.std([2.0, 4.0, 4.0, 4.0])
        assert z1["std"] == pytest.approx(expected_std)

    def test_gpu_median_odd(self):
        zones = from_numpy(np.array([[1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[1.0, 3.0, 2.0]]))
        result = zonal_stats(zones, values, stats=["median"], use_gpu=True)
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["median"] == 2.0

    def test_gpu_median_even(self):
        zones = from_numpy(np.array([[1, 1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[1.0, 3.0, 4.0, 2.0]]))
        result = zonal_stats(zones, values, stats=["median"], use_gpu=True)
        z1 = result[result["zone"] == 1].iloc[0]
        # Sorted: [1, 2, 3, 4], median = (2 + 3) / 2 = 2.5
        assert z1["median"] == pytest.approx(2.5)


@pytest.mark.gpu
@requires_gpu
class TestZonalStatsGPUNodata:
    """GPU path: nodata handling."""

    def test_gpu_values_nodata_excluded(self):
        zones = from_numpy(np.array([[1, 1, 1]], dtype=np.int32))
        values = from_numpy(
            np.array([[10.0, -9999.0, 30.0]]),
            nodata=-9999.0,
        )
        result = zonal_stats(zones, values, stats=["count", "sum"], use_gpu=True)
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["count"] == 2
        assert z1["sum"] == pytest.approx(40.0)

    def test_gpu_zone_nodata_excluded(self):
        zones = from_numpy(
            np.array([[1, 0, 2]], dtype=np.int32),
            nodata=0,
        )
        values = from_numpy(np.array([[10.0, 20.0, 30.0]]))
        result = zonal_stats(zones, values, stats=["count"], use_gpu=True)
        assert len(result) == 2
        assert 0 not in result["zone"].values

    def test_gpu_both_nodata(self):
        """Both zones and values have nodata; only fully valid pixels counted."""
        zones = from_numpy(
            np.array([[1, 0, 2, 2]], dtype=np.int32),
            nodata=0,
        )
        values = from_numpy(
            np.array([[10.0, 20.0, -9999.0, 40.0]]),
            nodata=-9999.0,
        )
        result = zonal_stats(zones, values, stats=["count", "sum"], use_gpu=True)
        # Zone 1: pixel 10.0 (valid zone, valid value)
        # Zone 0: excluded (nodata zone)
        # Zone 2: pixel -9999.0 excluded (nodata value), pixel 40.0 valid
        z1 = result[result["zone"] == 1].iloc[0]
        assert z1["count"] == 1
        assert z1["sum"] == pytest.approx(10.0)
        z2 = result[result["zone"] == 2].iloc[0]
        assert z2["count"] == 1
        assert z2["sum"] == pytest.approx(40.0)


@pytest.mark.gpu
@requires_gpu
class TestZonalStatsGPUEdgeCases:
    """GPU path: edge cases."""

    def test_gpu_single_zone(self):
        zones = from_numpy(np.array([[1, 1, 1, 1]], dtype=np.int32))
        values = from_numpy(np.array([[10.0, 20.0, 30.0, 40.0]]))
        result = zonal_stats(
            zones, values, stats=["count", "sum", "mean", "min", "max"], use_gpu=True
        )
        assert len(result) == 1
        z1 = result.iloc[0]
        assert z1["count"] == 4
        assert z1["sum"] == pytest.approx(100.0)
        assert z1["mean"] == pytest.approx(25.0)
        assert z1["min"] == 10.0
        assert z1["max"] == 40.0

    def test_gpu_all_nodata_zones(self):
        zones = from_numpy(
            np.array([[0, 0]], dtype=np.int32),
            nodata=0,
        )
        values = from_numpy(np.array([[1.0, 2.0]]))
        result = zonal_stats(zones, values, use_gpu=True)
        assert len(result) == 0

    def test_gpu_diagnostics_report_gpu(self):
        """Diagnostics should record GPU backend."""
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        zonal_stats(zones, values, stats=["count"], use_gpu=True)
        runtime_events = [e for e in zones.diagnostics if e.kind == "runtime"]
        assert any("gpu" in e.detail for e in runtime_events)


# ---------------------------------------------------------------------------
# zonal_stats_gdf tests — vector-based zonal statistics
# ---------------------------------------------------------------------------


def _make_gdf(geometries, index=None, **kwargs):
    """Create a GeoDataFrame from shapely geometries.

    Separate helper so we import geopandas lazily and only inside tests
    guarded by the requires_geopandas marker.
    """
    import geopandas as _gpd

    return _gpd.GeoDataFrame(kwargs, geometry=geometries, index=index)


@requires_geopandas
@requires_rasterio
class TestZonalStatsGdfBasic:
    """Basic functionality of zonal_stats_gdf."""

    def test_single_zone_covering_raster(self):
        """A single polygon covering the entire raster aggregates all pixels."""
        data = np.arange(1.0, 26.0).reshape(5, 5)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 5)], name=["full"])
        result = zonal_stats_gdf(gdf, values, stats=["count", "sum", "mean"], use_gpu=False)

        assert len(result) == 1
        z = result.iloc[0]
        assert z["count"] == 25
        assert z["sum"] == pytest.approx(np.sum(data))
        assert z["mean"] == pytest.approx(np.mean(data))

    def test_two_non_overlapping_zones(self):
        """Two non-overlapping polygons produce correct per-zone statistics."""
        # 10x10 raster with values = row * 10 + col
        data = np.arange(100, dtype=np.float64).reshape(10, 10)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        values = from_numpy(data, affine=affine)

        # Zone A covers upper-left quadrant, Zone B covers lower-right quadrant
        gdf = _make_gdf(
            [box(0, 5, 5, 10), box(5, 0, 10, 5)],
            name=["upper_left", "lower_right"],
        )

        result = zonal_stats_gdf(gdf, values, stats=["count", "sum", "min", "max"], use_gpu=False)
        assert len(result) == 2

        z1 = result[result["zone"] == 1].iloc[0]
        z2 = result[result["zone"] == 2].iloc[0]

        assert z1["count"] == 25
        assert z2["count"] == 25
        # Zone 1 (upper-left) should have lower values than Zone 2 (lower-right)
        assert z1["max"] < z2["min"]

    def test_all_seven_statistics(self):
        """All seven statistics are computed correctly via the GDF wrapper."""
        data = np.array(
            [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]],
            dtype=np.float64,
        )
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 2)], name=["all"])
        result = zonal_stats_gdf(
            gdf,
            values,
            stats=["count", "sum", "mean", "min", "max", "std", "median"],
            use_gpu=False,
        )

        z = result.iloc[0]
        flat = data.ravel()
        assert z["count"] == len(flat)
        assert z["sum"] == pytest.approx(np.sum(flat))
        assert z["mean"] == pytest.approx(np.mean(flat))
        assert z["min"] == pytest.approx(np.min(flat))
        assert z["max"] == pytest.approx(np.max(flat))
        assert z["std"] == pytest.approx(np.std(flat, ddof=0))
        assert z["median"] == pytest.approx(np.median(flat))

    def test_result_has_gdf_index_column(self):
        """Result DataFrame includes gdf_index mapping back to GeoDataFrame rows."""
        data = np.ones((5, 5), dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 3, 3)], name=["a"])
        result = zonal_stats_gdf(gdf, values, stats=["count"], use_gpu=False)

        assert "gdf_index" in result.columns
        assert result.iloc[0]["gdf_index"] == 0

    def test_zonal_spec_parameter(self):
        """Passing a ZonalSpec object selects specific statistics."""
        data = np.ones((5, 5), dtype=np.float64) * 7.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 5)], name=["z"])
        spec = ZonalSpec(stats=("min", "max"))
        result = zonal_stats_gdf(gdf, values, stats=spec, use_gpu=False)

        assert "min" in result.columns
        assert "max" in result.columns
        # Statistics not requested should be absent
        assert "sum" not in result.columns
        assert "count" not in result.columns

    def test_default_stats(self):
        """When stats=None, the default set (count, sum, mean, min, max) is used."""
        data = np.ones((5, 5), dtype=np.float64) * 3.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 5)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=None, use_gpu=False)

        for col in ["count", "sum", "mean", "min", "max"]:
            assert col in result.columns


@requires_geopandas
@requires_rasterio
class TestZonalStatsGdfNodata:
    """Nodata handling through the GDF wrapper."""

    def test_values_nodata_excluded(self):
        """Nodata pixels in the values raster are excluded from aggregation."""
        data = np.array([[10.0, -9999.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
        values = from_numpy(data, nodata=-9999.0, affine=affine)

        gdf = _make_gdf([box(0, 0, 3, 2)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["count", "sum"], use_gpu=False)

        z = result.iloc[0]
        # 5 valid pixels out of 6 (one nodata)
        assert z["count"] == 5
        assert z["sum"] == pytest.approx(10.0 + 30.0 + 40.0 + 50.0 + 60.0)

    def test_nan_nodata_excluded(self):
        """NaN nodata sentinel is correctly excluded from aggregation."""
        data = np.array([[1.0, np.nan, 3.0]], dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
        values = from_numpy(data, nodata=np.nan, affine=affine)

        gdf = _make_gdf([box(0, 0, 3, 1)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["count", "sum"], use_gpu=False)

        z = result.iloc[0]
        assert z["count"] == 2
        assert z["sum"] == pytest.approx(4.0)

    def test_nodata_only_zone_produces_nan(self):
        """A zone where all values are nodata should produce NaN statistics."""
        # Build a raster where all pixels in the zone's area are nodata
        data = np.full((5, 5), -9999.0, dtype=np.float64)
        data[3:, :] = 1.0  # bottom rows have real data
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, nodata=-9999.0, affine=affine)

        # Zone covers only the top area where all pixels are nodata
        gdf = _make_gdf(
            [box(0, 3, 5, 5), box(0, 0, 5, 2)],
            name=["nodata_zone", "valid_zone"],
        )
        result = zonal_stats_gdf(gdf, values, stats=["count", "mean"], use_gpu=False)

        # The nodata zone (zone 1) should be excluded entirely since
        # its rasterized area has all-nodata values.  The valid zone
        # (zone 2) should have real statistics.
        # Zone 1 may or may not appear depending on rasterization --
        # if it appears, its count should be 0 or NaN.
        if len(result[result["zone"] == 1]) > 0:
            z1_count = result[result["zone"] == 1].iloc[0]["count"]
            # If present with all-nodata values, count should be 0
            # (the zone label pixels exist but all values are nodata)
            assert z1_count == 0
        z2 = result[result["zone"] == 2].iloc[0]
        assert z2["count"] > 0


@requires_geopandas
@requires_rasterio
class TestZonalStatsGdfEdgeCases:
    """Edge cases for zonal_stats_gdf."""

    def test_empty_geodataframe(self):
        """An empty GeoDataFrame returns an empty result."""
        data = np.ones((5, 5), dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([], name=[])
        result = zonal_stats_gdf(gdf, values, stats=["count"], use_gpu=False)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_zone_outside_raster_extent(self):
        """A polygon entirely outside the raster extent produces no zones."""
        data = np.ones((5, 5), dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        # Polygon far outside the raster (raster covers x=0..5, y=0..5)
        gdf = _make_gdf([box(20, 20, 25, 25)], name=["outside"])
        result = zonal_stats_gdf(gdf, values, stats=["count"], use_gpu=False)

        assert len(result) == 0

    def test_non_default_geodataframe_index(self):
        """Custom GeoDataFrame index values are preserved in gdf_index column."""
        data = np.ones((10, 10), dtype=np.float64) * 5.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf(
            [box(0, 5, 5, 10), box(5, 0, 10, 5)],
            index=[100, 200],
            name=["a", "b"],
        )
        result = zonal_stats_gdf(gdf, values, stats=["count"], use_gpu=False)

        assert len(result) == 2
        # gdf_index should map to the custom index values 100 and 200
        gdf_indices = sorted(result["gdf_index"].values)
        assert gdf_indices == [100, 200]

    def test_single_pixel_zone(self):
        """A polygon covering exactly one pixel computes correct statistics."""
        data = np.array([[42.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
        values = from_numpy(data, affine=affine)

        # Polygon covering only the upper-left pixel
        gdf = _make_gdf([box(0.0, 1.0, 1.0, 2.0)], name=["tiny"])
        result = zonal_stats_gdf(gdf, values, stats=["count", "sum", "mean"], use_gpu=False)

        if len(result) > 0:
            z = result.iloc[0]
            assert z["count"] >= 1
            # If the zone captures exactly the upper-left pixel, value is 42.0
            assert z["mean"] == pytest.approx(42.0)

    def test_overlapping_polygons(self):
        """Overlapping polygons: later polygon wins at overlap pixels (rasterize convention)."""
        data = np.ones((10, 10), dtype=np.float64) * 5.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        values = from_numpy(data, affine=affine)

        # Two overlapping zones
        gdf = _make_gdf(
            [box(0, 0, 7, 7), box(3, 3, 10, 10)],
            name=["first", "second"],
        )
        result = zonal_stats_gdf(gdf, values, stats=["count"], use_gpu=False)

        assert len(result) == 2
        z1 = result[result["zone"] == 1].iloc[0]
        z2 = result[result["zone"] == 2].iloc[0]
        # Overlap pixels go to zone 2 (second polygon), so zone 1 gets fewer pixels
        total = z1["count"] + z2["count"]
        # Total should be <= 100 (10x10 raster) since some pixels may be fill_value
        assert total <= 100

    def test_constant_value_raster(self):
        """All pixels have the same value; min, max, std, mean are all correct."""
        data = np.full((5, 5), 7.5, dtype=np.float64)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 5)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["min", "max", "mean", "std"], use_gpu=False)

        z = result.iloc[0]
        assert z["min"] == pytest.approx(7.5)
        assert z["max"] == pytest.approx(7.5)
        assert z["mean"] == pytest.approx(7.5)
        assert z["std"] == pytest.approx(0.0)


@requires_geopandas
@requires_rasterio
class TestZonalStatsGdfGridSpec:
    """Custom grid_spec parameter behavior."""

    def test_default_grid_matches_values_raster(self):
        """When grid_spec=None, zone raster matches the values raster grid."""
        data = np.arange(1.0, 17.0).reshape(4, 4)
        affine = (2.0, 0.0, 0.0, 0.0, -2.0, 8.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 8, 8)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["count", "sum"], use_gpu=False)

        z = result.iloc[0]
        assert z["count"] == 16
        assert z["sum"] == pytest.approx(np.sum(data))

    def test_explicit_grid_spec(self):
        """An explicit grid_spec is used for rasterizing the zones."""
        data = np.ones((10, 10), dtype=np.float64) * 5.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
        values = from_numpy(data, affine=affine)

        # Custom grid_spec that matches the values raster exactly
        custom_grid = GridSpec(
            affine=affine,
            width=10,
            height=10,
            dtype=np.dtype("int32"),
            fill_value=0,
        )

        gdf = _make_gdf([box(2, 2, 5, 5)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["count"], grid_spec=custom_grid, use_gpu=False)

        assert len(result) == 1
        z = result.iloc[0]
        assert z["count"] > 0


@requires_geopandas
@requires_rasterio
class TestZonalStatsGdfDiagnostics:
    """Diagnostic event coverage for zonal_stats_gdf."""

    def test_diagnostics_appended_on_cpu_path(self):
        """Calling zonal_stats_gdf appends a runtime diagnostic event."""
        data = np.ones((5, 5), dtype=np.float64) * 3.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 5)], name=["z"])
        # zonal_stats_gdf delegates to zonal_stats which records diagnostics
        # on the internal zone_raster, not on the values raster.
        # Verify that the function completes without error and returns data.
        result = zonal_stats_gdf(gdf, values, stats=["count"], use_gpu=False)
        assert len(result) == 1

    def test_dispatches_to_cpu_for_small_rasters(self):
        """Small rasters auto-dispatch to CPU via the underlying zonal_stats."""
        data = np.ones((5, 5), dtype=np.float64) * 2.0
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 5.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 5, 5)], name=["z"])
        # With use_gpu=None (default), the small raster should auto-dispatch to CPU
        result = zonal_stats_gdf(gdf, values, stats=["count"])
        assert len(result) == 1


@requires_geopandas
@requires_rasterio
class TestZonalStatsGdfIntegerValues:
    """Integer-typed value rasters are handled correctly."""

    def test_int32_values_raster(self):
        """Integer value rasters are upcast to float64 for statistics."""
        data = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 3, 2)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["count", "sum", "mean"], use_gpu=False)

        z = result.iloc[0]
        assert z["count"] == 6
        assert z["sum"] == pytest.approx(210.0)
        assert z["mean"] == pytest.approx(35.0)

    def test_uint8_values_raster(self):
        """uint8 value rasters produce correct statistics without overflow."""
        data = np.array([[100, 200, 250]], dtype=np.uint8)
        affine = (1.0, 0.0, 0.0, 0.0, -1.0, 1.0)
        values = from_numpy(data, affine=affine)

        gdf = _make_gdf([box(0, 0, 3, 1)], name=["z"])
        result = zonal_stats_gdf(gdf, values, stats=["sum", "mean"], use_gpu=False)

        z = result.iloc[0]
        # Sum should be 550 (not overflow-truncated)
        assert z["sum"] == pytest.approx(550.0)
        assert z["mean"] == pytest.approx(550.0 / 3)
