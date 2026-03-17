"""Tests for zonal statistics."""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import ZonalSpec, from_numpy
from vibespatial.raster.zonal import _has_cupy, zonal_stats

requires_gpu = pytest.mark.skipif(not _has_cupy(), reason="CuPy not available")


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

    def test_zonal_spec(self):
        zones = from_numpy(np.array([[1, 2]], dtype=np.int32))
        values = from_numpy(np.array([[5.0, 10.0]]))
        spec = ZonalSpec(stats=("min", "max"))
        result = zonal_stats(zones, values, stats=spec)
        assert "min" in result.columns
        assert "max" in result.columns
        assert "sum" not in result.columns


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
