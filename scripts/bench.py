#!/usr/bin/env python3
"""Bench: test, profile, and benchmark vibespatial.raster operations.

Usage:
    uv run python scripts/bench.py test
    uv run python scripts/bench.py test -f raster_add
    uv run python scripts/bench.py profile
    uv run python scripts/bench.py benchmark -n 10
    uv run python scripts/bench.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import threading
import time
import traceback as tb_mod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_SAMPLE = _REPO / "SAMPLE_DATA" / "NE1_HR_LC_SR_W_DR.tif"
_OUTPUT_ROOT = Path(__file__).resolve().parent / "bench_output"

_DEFAULT_WARMUP = 2
_DEFAULT_ITERATIONS = 5

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------


def _has_cupy() -> bool:
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        return False


def _has_rasterio() -> bool:
    try:
        import rasterio  # noqa: F401

        return True
    except Exception:
        return False


def _has_pynvml() -> bool:
    try:
        import pynvml  # noqa: F401

        return True
    except Exception:
        return False


# Cache probes so they only run once
_CUPY: bool | None = None
_RASTERIO: bool | None = None
_PYNVML: bool | None = None


def has_cupy() -> bool:
    global _CUPY
    if _CUPY is None:
        _CUPY = _has_cupy()
    return _CUPY


def has_rasterio() -> bool:
    global _RASTERIO
    if _RASTERIO is None:
        _RASTERIO = _has_rasterio()
    return _RASTERIO


def has_pynvml() -> bool:
    global _PYNVML
    if _PYNVML is None:
        _PYNVML = _has_pynvml()
    return _PYNVML


# ---------------------------------------------------------------------------
# GPU monitor — polls NVML at ~1 ms in a background thread
# ---------------------------------------------------------------------------


@dataclass
class GpuSample:
    t_ms: float
    gpu_pct: int
    mem_pct: int
    mem_used_mb: float


class GpuMonitor:
    """Background NVML sampler at ~1 ms intervals."""

    def __init__(self):
        self._handle: Any = None
        self._samples: list[GpuSample] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._ok = False
        self.gpu_name = "N/A"
        self.gpu_mem_total_mb = 0.0

    def initialize(self):
        if not has_pynvml():
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.gpu_name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.gpu_mem_total_mb = mem.total / 1024**2
            self._ok = True
        except Exception:
            pass

    def shutdown(self):
        if self._ok:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def start(self):
        if not self._ok:
            return
        self._samples = []
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> list[GpuSample]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        return list(self._samples)

    def _poll(self):
        import pynvml

        t0 = time.perf_counter()
        while self._running:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._samples.append(
                    GpuSample(
                        t_ms=round(elapsed_ms, 3),
                        gpu_pct=util.gpu,
                        mem_pct=util.memory,
                        mem_used_mb=round(mem.used / 1024**2, 1),
                    )
                )
            except Exception:
                pass
            time.sleep(0.001)


# ---------------------------------------------------------------------------
# Operation registry
# ---------------------------------------------------------------------------


@dataclass
class OpDef:
    name: str
    category: str
    run: Callable[[Any], Any]
    baseline: Callable[[Any], Any] | None = None
    check: Callable[[Any, Any], None] | None = None
    requires_cupy: bool = False
    requires_rasterio: bool = False


REGISTRY: dict[str, OpDef] = {}


def op(
    name: str,
    *,
    category: str,
    requires_cupy: bool = False,
    requires_rasterio: bool = False,
):
    """Register a bench operation.

    The decorated function must return ``(run, baseline, check)`` callables.
    ``baseline`` and ``check`` may be ``None``.
    """

    def decorator(func: Callable) -> Callable:
        run_fn, baseline_fn, check_fn = func()
        REGISTRY[name] = OpDef(
            name=name,
            category=category,
            run=run_fn,
            baseline=baseline_fn,
            check=check_fn,
            requires_cupy=requires_cupy,
            requires_rasterio=requires_rasterio,
        )
        return func

    return decorator


# ---------------------------------------------------------------------------
# Test context — lazily prepares inputs from the sample GeoTIFF
# ---------------------------------------------------------------------------


class TestContext:
    """Lazy-cached test data derived from the sample GeoTIFF."""

    def __init__(self, sample_path: Path):
        self.sample_path = sample_path
        self._cache: dict[str, Any] = {}

    def _get(self, key: str, builder: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = builder()
        return self._cache[key]

    def warmup(self):
        """Pre-compute derived data so op timing is clean."""
        for name in (
            "raster",
            "metadata",
            "single_band",
            "second_raster",
            "binary_mask",
            "labeled",
            "zones",
            "grid_spec",
        ):
            try:
                getattr(self, name)
            except Exception as exc:
                print(f"  {_YELLOW}Warning: ctx.{name}: {exc}{_RESET}")

    def evict_device(self):
        """Re-create all cached rasters as HOST-resident to free GPU memory."""
        import gc

        host_snapshots: dict[str, dict] = {}
        for key, val in list(self._cache.items()):
            if hasattr(val, "to_numpy") and hasattr(val, "affine"):
                host_snapshots[key] = {
                    "data": val.to_numpy().copy(),
                    "nodata": val.nodata,
                    "affine": val.affine,
                    "crs": val.crs,
                }
        for key in host_snapshots:
            del self._cache[key]
        gc.collect()
        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            from vibespatial.raster.memory import free_pool_memory

            free_pool_memory()
        except Exception:
            pass
        from vibespatial.raster import from_numpy

        for key, info in host_snapshots.items():
            self._cache[key] = from_numpy(
                info["data"], nodata=info["nodata"], affine=info["affine"], crs=info["crs"]
            )

    # -- core properties ---------------------------------------------------

    @property
    def raster(self):
        def build():
            from vibespatial.raster import read_raster

            return read_raster(str(self.sample_path))

        return self._get("raster", build)

    @property
    def metadata(self):
        def build():
            from vibespatial.raster import read_raster_metadata

            return read_raster_metadata(str(self.sample_path))

        return self._get("metadata", build)

    @property
    def single_band(self):
        """First band as float32 OwnedRasterArray."""

        def build():
            from vibespatial.raster import from_numpy

            data = self.raster.to_numpy()
            band = data[0] if data.ndim == 3 else data
            return from_numpy(
                band.astype(np.float32),
                nodata=np.float32("nan"),
                affine=self.raster.affine,
                crs=self.raster.crs,
            )

        return self._get("single_band", build)

    @property
    def second_raster(self):
        """Shifted copy for binary algebra ops."""

        def build():
            from vibespatial.raster import from_numpy

            data = self.single_band.to_numpy()
            shifted = np.roll(data, 100, axis=1).copy()
            return from_numpy(
                shifted,
                nodata=np.float32("nan"),
                affine=self.single_band.affine,
                crs=self.single_band.crs,
            )

        return self._get("second_raster", build)

    @property
    def binary_mask(self):
        """Binary mask (uint8, 0/1) from median threshold."""

        def build():
            from vibespatial.raster import from_numpy

            data = self.single_band.to_numpy()
            mask = (data > np.nanmedian(data)).astype(np.uint8)
            return from_numpy(
                mask,
                nodata=None,
                affine=self.single_band.affine,
                crs=self.single_band.crs,
            )

        return self._get("binary_mask", build)

    @property
    def labeled(self):
        """CCL labels (CPU-computed for setup)."""

        def build():
            from vibespatial.raster import label_connected_components

            return label_connected_components(self.binary_mask, connectivity=4, use_gpu=False)

        return self._get("labeled", build)

    @property
    def zones(self):
        """Classified zones (~10 classes, int32, 1-indexed)."""

        def build():
            from vibespatial.raster import from_numpy

            data = self.single_band.to_numpy()
            vmin, vmax = float(np.nanmin(data)), float(np.nanmax(data))
            bins = np.linspace(vmin, vmax, 11)[1:-1]
            z = np.digitize(data, bins).astype(np.int32) + 1
            return from_numpy(
                z,
                nodata=0,
                affine=self.single_band.affine,
                crs=self.single_band.crs,
            )

        return self._get("zones", build)

    @property
    def geometries_and_values(self):
        """(list[Polygon], np.ndarray) for rasterize tests."""

        def build():
            from shapely.geometry import box

            bnd = self.raster.bounds
            nx, ny = 20, 10
            dx, dy = (bnd[2] - bnd[0]) / nx, (bnd[3] - bnd[1]) / ny
            geoms, vals = [], []
            for i in range(nx):
                for j in range(ny):
                    x0, y0 = bnd[0] + i * dx, bnd[1] + j * dy
                    geoms.append(box(x0, y0, x0 + dx * 0.9, y0 + dy * 0.9))
                    vals.append(float(i * ny + j + 1))
            return geoms, np.array(vals, dtype=np.float64)

        return self._get("geometries_and_values", build)

    @property
    def grid_spec(self):
        def build():
            from vibespatial.raster import GridSpec

            return GridSpec.from_raster(self.single_band)

        return self._get("grid_spec", build)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _chk_raster(result, ctx):
    """Assert result is a valid OwnedRasterArray with non-degenerate data."""
    assert hasattr(result, "to_numpy"), f"Expected OwnedRasterArray, got {type(result).__name__}"
    d = result.to_numpy()
    assert d.size > 0, "Empty output"
    if np.issubdtype(d.dtype, np.floating):
        assert not np.all(np.isnan(d)), "All NaN output"


def _chk_df(result, _ctx):
    """Assert result is a non-empty DataFrame."""
    import pandas as pd

    assert isinstance(result, pd.DataFrame), f"Expected DataFrame, got {type(result).__name__}"
    assert not result.empty, "Empty DataFrame"


def _chk_polygons(result, _ctx):
    """Assert polygonize result is (geoms, values) with content."""
    assert isinstance(result, tuple) and len(result) == 2, (
        f"Expected (geoms, values) tuple, got {type(result)}"
    )
    assert len(result[0]) > 0, "No polygons produced"


# ===========================================================================
# Operations — IO
# ===========================================================================


@op("read_raster", category="io", requires_rasterio=True)
def _op_read_raster():
    def run(ctx):
        from vibespatial.raster import read_raster

        return read_raster(str(ctx.sample_path))

    def baseline(ctx):
        import rasterio

        with rasterio.open(ctx.sample_path) as ds:
            return ds.read()

    return run, baseline, _chk_raster


@op("write_raster", category="io", requires_rasterio=True)
def _op_write_raster():
    def run(ctx):
        from vibespatial.raster import write_raster

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = f.name
        try:
            write_raster(tmp, ctx.raster)
        finally:
            os.unlink(tmp)

    def baseline(ctx):
        import rasterio
        from rasterio.transform import Affine

        data = ctx.raster.to_numpy()
        if data.ndim == 2:
            data = data[np.newaxis]
        a = ctx.raster.affine
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            tmp = f.name
        try:
            with rasterio.open(
                tmp,
                "w",
                driver="GTiff",
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                transform=Affine(a[0], a[1], a[2], a[3], a[4], a[5]),
            ) as ds:
                ds.write(data)
        finally:
            os.unlink(tmp)

    return run, baseline, None


@op("read_raster_metadata", category="io", requires_rasterio=True)
def _op_read_metadata():
    def run(ctx):
        from vibespatial.raster import read_raster_metadata

        m = read_raster_metadata(str(ctx.sample_path))
        return m

    def baseline(ctx):
        import rasterio

        with rasterio.open(ctx.sample_path) as ds:
            return {"height": ds.height, "width": ds.width, "count": ds.count}

    def check(result, _ctx):
        assert result.width > 0 and result.height > 0

    return run, baseline, check


# ===========================================================================
# Operations — Algebra (local)
# ===========================================================================


@op("raster_add", category="algebra", requires_cupy=True)
def _op_add():
    def run(ctx):
        from vibespatial.raster import raster_add

        return raster_add(ctx.single_band, ctx.second_raster)

    def baseline(ctx):
        return np.add(ctx.single_band.to_numpy(), ctx.second_raster.to_numpy())

    return run, baseline, _chk_raster


@op("raster_subtract", category="algebra", requires_cupy=True)
def _op_sub():
    def run(ctx):
        from vibespatial.raster import raster_subtract

        return raster_subtract(ctx.single_band, ctx.second_raster)

    def baseline(ctx):
        return np.subtract(ctx.single_band.to_numpy(), ctx.second_raster.to_numpy())

    return run, baseline, _chk_raster


@op("raster_multiply", category="algebra", requires_cupy=True)
def _op_mul():
    def run(ctx):
        from vibespatial.raster import raster_multiply

        return raster_multiply(ctx.single_band, ctx.second_raster)

    def baseline(ctx):
        return np.multiply(ctx.single_band.to_numpy(), ctx.second_raster.to_numpy())

    return run, baseline, _chk_raster


@op("raster_divide", category="algebra", requires_cupy=True)
def _op_div():
    def run(ctx):
        from vibespatial.raster import raster_divide

        return raster_divide(ctx.single_band, ctx.second_raster)

    def baseline(ctx):
        a = ctx.single_band.to_numpy()
        b = ctx.second_raster.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.true_divide(a, b)

    return run, baseline, _chk_raster


@op("raster_apply", category="algebra", requires_cupy=True)
def _op_apply():
    def run(ctx):
        from vibespatial.raster import raster_apply

        return raster_apply(ctx.single_band, lambda x: x * 2.0)

    def baseline(ctx):
        return ctx.single_band.to_numpy() * 2.0

    return run, baseline, _chk_raster


@op("raster_where", category="algebra", requires_cupy=True)
def _op_where():
    def run(ctx):
        from vibespatial.raster import raster_where

        return raster_where(ctx.binary_mask, ctx.single_band, 0.0)

    def baseline(ctx):
        cond = ctx.binary_mask.to_numpy().astype(bool)
        return np.where(cond, ctx.single_band.to_numpy(), 0.0)

    return run, baseline, _chk_raster


@op("raster_classify", category="algebra", requires_cupy=True)
def _op_classify():
    def run(ctx):
        from vibespatial.raster import raster_classify

        d = ctx.single_band.to_numpy()
        bins = list(np.linspace(float(np.nanmin(d)), float(np.nanmax(d)), 6)[1:-1])
        labels = list(range(len(bins) + 1))
        return raster_classify(ctx.single_band, bins, labels)

    def baseline(ctx):
        d = ctx.single_band.to_numpy()
        bins = np.linspace(np.nanmin(d), np.nanmax(d), 6)[1:-1]
        return np.digitize(d, bins)

    return run, baseline, _chk_raster


# ===========================================================================
# Operations — Algebra (focal)
# ===========================================================================


@op("raster_convolve", category="focal", requires_cupy=True)
def _op_convolve():
    def run(ctx):
        from vibespatial.raster import raster_convolve

        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        return raster_convolve(ctx.single_band, kernel)

    def baseline(ctx):
        from scipy.ndimage import convolve

        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        return convolve(ctx.single_band.to_numpy(), kernel)

    return run, baseline, _chk_raster


@op("raster_gaussian_filter", category="focal", requires_cupy=True)
def _op_gaussian():
    def run(ctx):
        from vibespatial.raster import raster_gaussian_filter

        return raster_gaussian_filter(ctx.single_band, sigma=1.0)

    def baseline(ctx):
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(ctx.single_band.to_numpy(), sigma=1.0)

    return run, baseline, _chk_raster


@op("raster_slope", category="focal", requires_cupy=True)
def _op_slope():
    def run(ctx):
        from vibespatial.raster import raster_slope

        return raster_slope(ctx.single_band)

    def baseline(ctx):
        d = ctx.single_band.to_numpy()
        p = np.pad(d, 1, mode="edge")
        dz_dx = (
            (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        ) / 8.0
        dz_dy = (
            (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        ) / 8.0
        return np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    return run, baseline, _chk_raster


@op("raster_aspect", category="focal", requires_cupy=True)
def _op_aspect():
    def run(ctx):
        from vibespatial.raster import raster_aspect

        return raster_aspect(ctx.single_band)

    def baseline(ctx):
        d = ctx.single_band.to_numpy()
        p = np.pad(d, 1, mode="edge")
        dz_dx = (
            (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
        ) / 8.0
        dz_dy = (
            (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
            - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
        ) / 8.0
        return (90 - np.degrees(np.arctan2(-dz_dy, dz_dx))) % 360

    return run, baseline, _chk_raster


# ===========================================================================
# Operations — Label & morphology
# ===========================================================================


@op("label_connected_components", category="label")
def _op_label():
    def run(ctx):
        from vibespatial.raster import label_connected_components

        return label_connected_components(ctx.binary_mask, connectivity=4)

    def baseline(ctx):
        from scipy.ndimage import label as scipy_label

        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        labeled, _ = scipy_label(ctx.binary_mask.to_numpy(), structure=struct)
        return labeled

    def check(result, _ctx):
        d = result.to_numpy()
        assert d.max() > 0, "Should find at least one component"

    return run, baseline, check


@op("morphology_erode", category="label")
def _op_erode():
    def run(ctx):
        from vibespatial.raster import raster_morphology

        return raster_morphology(ctx.binary_mask, "erode", connectivity=4)

    def baseline(ctx):
        from scipy.ndimage import binary_erosion

        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        return binary_erosion(ctx.binary_mask.to_numpy(), structure=struct).astype(np.uint8)

    return run, baseline, _chk_raster


@op("morphology_dilate", category="label")
def _op_dilate():
    def run(ctx):
        from vibespatial.raster import raster_morphology

        return raster_morphology(ctx.binary_mask, "dilate", connectivity=4)

    def baseline(ctx):
        from scipy.ndimage import binary_dilation

        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        return binary_dilation(ctx.binary_mask.to_numpy(), structure=struct).astype(np.uint8)

    return run, baseline, _chk_raster


@op("sieve_filter", category="label")
def _op_sieve():
    def run(ctx):
        from vibespatial.raster import sieve_filter

        return sieve_filter(ctx.labeled, min_size=100, connectivity=4)

    return run, None, _chk_raster


# ===========================================================================
# Operations — Zonal
# ===========================================================================


@op("zonal_stats", category="zonal")
def _op_zonal():
    def run(ctx):
        from vibespatial.raster import zonal_stats

        return zonal_stats(ctx.zones, ctx.single_band, ["count", "sum", "mean", "min", "max"])

    def baseline(ctx):
        import pandas as pd

        z = ctx.zones.to_numpy().ravel()
        v = ctx.single_band.to_numpy().ravel()
        valid = ~np.isnan(v) & (z > 0)
        df = pd.DataFrame({"zone": z[valid], "value": v[valid]})
        return df.groupby("zone")["value"].agg(["count", "sum", "mean", "min", "max"])

    return run, baseline, _chk_df


# ===========================================================================
# Operations — Polygonize
# ===========================================================================


@op("polygonize_owned", category="polygonize")
def _op_polygonize():
    def run(ctx):
        from vibespatial.raster import polygonize_owned

        return polygonize_owned(ctx.labeled, connectivity=4, max_polygons=10_000)

    def baseline(ctx):
        from rasterio.features import shapes
        from rasterio.transform import Affine

        data = ctx.labeled.to_numpy().astype(np.int32)
        a = ctx.labeled.affine
        return list(shapes(data, transform=Affine(a[0], a[1], a[2], a[3], a[4], a[5])))

    return run, baseline, _chk_polygons


# ===========================================================================
# Operations — Rasterize
# ===========================================================================


@op("rasterize_owned", category="rasterize")
def _op_rasterize():
    def run(ctx):
        from vibespatial.raster import rasterize_owned

        geoms, vals = ctx.geometries_and_values
        return rasterize_owned(geoms, vals, ctx.grid_spec)

    def baseline(ctx):
        from rasterio.features import rasterize
        from rasterio.transform import Affine

        geoms, vals = ctx.geometries_and_values
        gs = ctx.grid_spec
        a = gs.affine
        return rasterize(
            list(zip(geoms, vals)),
            out_shape=(gs.height, gs.width),
            transform=Affine(a[0], a[1], a[2], a[3], a[4], a[5]),
            fill=0,
            dtype=gs.dtype,
        )

    return run, baseline, _chk_raster


# ===========================================================================
# Result types
# ===========================================================================


@dataclass
class OpResult:
    name: str
    category: str
    status: str  # pass | fail | skip | error
    time_s: float = 0.0
    error: str | None = None
    error_tb: str | None = None
    # benchmark
    baseline_time_s: float | None = None
    speedup: float | None = None
    op_times: list[float] = field(default_factory=list)
    baseline_times: list[float] = field(default_factory=list)
    # profile
    gpu_trace: list[dict] | None = None
    gpu_peak_pct: float | None = None
    gpu_avg_pct: float | None = None
    gpu_peak_mem_mb: float | None = None


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def _free_gpu_memory(ctx: TestContext | None = None):
    """Release cached CuPy GPU memory between operations."""
    if ctx is not None:
        ctx.evict_device()
        return
    import gc

    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    try:
        from vibespatial.raster.memory import free_pool_memory

        free_pool_memory()
    except Exception:
        pass


def _select(function_filter: str | None) -> list[OpDef]:
    """Select ops by exact name or category prefix."""
    if function_filter is None:
        return list(REGISTRY.values())
    # exact name
    if function_filter in REGISTRY:
        return [REGISTRY[function_filter]]
    # category match
    matches = [o for o in REGISTRY.values() if o.category == function_filter]
    if matches:
        return matches
    print(f"Unknown operation or category: {function_filter}", file=sys.stderr)
    print(f"Available: {', '.join(REGISTRY.keys())}", file=sys.stderr)
    sys.exit(1)


def _skip_reason(o: OpDef) -> str | None:
    if o.requires_cupy and not has_cupy():
        return "CuPy not available"
    if o.requires_rasterio and not has_rasterio():
        return "rasterio not available"
    return None


def run_tests(ops: list[OpDef], ctx: TestContext) -> list[OpResult]:
    results: list[OpResult] = []
    for o in ops:
        reason = _skip_reason(o)
        if reason:
            r = OpResult(o.name, o.category, "skip", error=reason)
            results.append(r)
            _print_result(r)
            continue
        t0 = time.perf_counter()
        try:
            result = o.run(ctx)
            elapsed = time.perf_counter() - t0
            if o.check:
                o.check(result, ctx)
            r = OpResult(o.name, o.category, "pass", time_s=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            r = OpResult(
                o.name,
                o.category,
                "error",
                time_s=elapsed,
                error=f"{type(exc).__name__}: {exc}",
                error_tb=tb_mod.format_exc(),
            )
        results.append(r)
        _print_result(r)
        _free_gpu_memory(ctx)
    return results


def run_profiles(ops: list[OpDef], ctx: TestContext, gpu: GpuMonitor) -> list[OpResult]:
    results: list[OpResult] = []
    for o in ops:
        reason = _skip_reason(o)
        if reason:
            r = OpResult(o.name, o.category, "skip", error=reason)
            results.append(r)
            _print_result(r, mode="profile")
            continue
        gpu.start()
        t0 = time.perf_counter()
        try:
            o.run(ctx)
            elapsed = time.perf_counter() - t0
            trace = gpu.stop()
            trace_dicts = [
                {
                    "t_ms": s.t_ms,
                    "gpu_pct": s.gpu_pct,
                    "mem_pct": s.mem_pct,
                    "mem_used_mb": s.mem_used_mb,
                }
                for s in trace
            ]
            gpu_pcts = [s.gpu_pct for s in trace]
            mem_mbs = [s.mem_used_mb for s in trace]
            r = OpResult(
                o.name,
                o.category,
                "pass",
                time_s=elapsed,
                gpu_trace=trace_dicts,
                gpu_peak_pct=max(gpu_pcts) if gpu_pcts else None,
                gpu_avg_pct=(round(statistics.mean(gpu_pcts), 1) if gpu_pcts else None),
                gpu_peak_mem_mb=max(mem_mbs) if mem_mbs else None,
            )
        except Exception as exc:
            gpu.stop()
            elapsed = time.perf_counter() - t0
            r = OpResult(
                o.name,
                o.category,
                "error",
                time_s=elapsed,
                error=f"{type(exc).__name__}: {exc}",
                error_tb=tb_mod.format_exc(),
            )
        results.append(r)
        _print_result(r, mode="profile")
        _free_gpu_memory(ctx)
    return results


def run_benchmarks(
    ops: list[OpDef],
    ctx: TestContext,
    *,
    warmup: int = _DEFAULT_WARMUP,
    iterations: int = _DEFAULT_ITERATIONS,
) -> list[OpResult]:
    results: list[OpResult] = []
    for o in ops:
        reason = _skip_reason(o)
        if reason:
            r = OpResult(o.name, o.category, "skip", error=reason)
            results.append(r)
            _print_result(r, mode="benchmark")
            continue
        # Ours
        op_times: list[float] = []
        error = None
        try:
            for i in range(warmup + iterations):
                t0 = time.perf_counter()
                o.run(ctx)
                elapsed = time.perf_counter() - t0
                if i >= warmup:
                    op_times.append(elapsed)
                _free_gpu_memory(ctx)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _free_gpu_memory(ctx)
        # Baseline
        bl_times: list[float] = []
        if o.baseline and not error:
            try:
                for i in range(warmup + iterations):
                    t0 = time.perf_counter()
                    o.baseline(ctx)
                    elapsed = time.perf_counter() - t0
                    if i >= warmup:
                        bl_times.append(elapsed)
            except Exception:
                pass
        our_med = statistics.median(op_times) if op_times else 0.0
        bl_med = statistics.median(bl_times) if bl_times else None
        speedup = (bl_med / our_med) if (bl_med and our_med > 0) else None
        r = OpResult(
            o.name,
            o.category,
            "error" if error else "pass",
            time_s=our_med,
            error=error,
            baseline_time_s=bl_med,
            speedup=round(speedup, 2) if speedup else None,
            op_times=op_times,
            baseline_times=bl_times,
        )
        results.append(r)
        _print_result(r, mode="benchmark")
        _free_gpu_memory(ctx)
    return results


# ---------------------------------------------------------------------------
# Reporting — terminal
# ---------------------------------------------------------------------------


def _fmt(s: float) -> str:
    if s == 0:
        return "    —"
    if s < 0.001:
        return f"{s * 1e6:.0f}µs"
    if s < 1.0:
        return f"{s * 1000:.1f}ms"
    return f"{s:.3f}s"


def _print_result(r: OpResult, *, mode: str = "test"):
    if r.status == "pass":
        icon = f"{_GREEN}PASS{_RESET}"
    elif r.status == "skip":
        icon = f"{_YELLOW}SKIP{_RESET}"
    else:
        icon = f"{_RED}FAIL{_RESET}"

    line = f"  [{icon}] {r.name:<34}"

    if mode == "benchmark" and r.status == "pass":
        bl = _fmt(r.baseline_time_s) if r.baseline_time_s else "—"
        sp = f"{r.speedup:.1f}x" if r.speedup else "—"
        line += f" {_fmt(r.time_s):>9}  baseline {bl:>9}  {sp:>7}"
    elif mode == "profile" and r.status == "pass":
        gp = f"{r.gpu_peak_pct:.0f}%" if r.gpu_peak_pct is not None else "—"
        ga = f"{r.gpu_avg_pct:.0f}%" if r.gpu_avg_pct is not None else "—"
        gm = f"{r.gpu_peak_mem_mb:.0f}MB" if r.gpu_peak_mem_mb is not None else "—"
        line += f" {_fmt(r.time_s):>9}  gpu {gp:>4}/{ga:>4}  mem {gm:>8}"
    elif r.status == "pass":
        line += f" {_fmt(r.time_s):>9}"
    elif r.error:
        line += f" {_DIM}{r.error[:60]}{_RESET}"

    print(line)


def _print_header(meta: dict):
    print()
    print(f"  {_BOLD}vibespatial.raster bench{_RESET}")
    print(f"  {'═' * 56}")
    print(f"  Sample : {meta['sample_name']}")
    print(f"  Shape  : {meta['shape_str']}  │  {meta['dtype']}  │  {meta['size_mb']:.1f} MB")
    if meta.get("crs"):
        print(f"  CRS    : {meta['crs']}")
    gpu = meta.get("gpu_name", "N/A")
    if gpu != "N/A":
        print(f"  GPU    : {gpu}  │  {meta['gpu_mem_total_mb']:.0f} MB")
    else:
        print("  GPU    : not available")
    print(f"  CuPy   : {'yes' if meta['has_cupy'] else 'no'}")
    print()


def _print_summary(results: list[OpResult], mode: str):
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status in ("fail", "error"))
    skipped = sum(1 for r in results if r.status == "skip")
    parts = [f"{passed} passed"]
    if failed:
        parts.append(f"{_RED}{failed} failed{_RESET}")
    if skipped:
        parts.append(f"{_YELLOW}{skipped} skipped{_RESET}")
    print(f"\n  {mode.upper()} complete: {', '.join(parts)}")


# ---------------------------------------------------------------------------
# Reporting — file output
# ---------------------------------------------------------------------------


def _serialize(results: list[OpResult], mode: str, meta: dict) -> dict:
    return {
        "meta": meta,
        "mode": mode,
        "results": [
            {
                k: v
                for k, v in {
                    "name": r.name,
                    "category": r.category,
                    "status": r.status,
                    "time_s": round(r.time_s, 6),
                    "error": r.error,
                    "baseline_time_s": (
                        round(r.baseline_time_s, 6) if r.baseline_time_s is not None else None
                    ),
                    "speedup": r.speedup,
                    "op_times_s": ([round(t, 6) for t in r.op_times] if r.op_times else None),
                    "baseline_times_s": (
                        [round(t, 6) for t in r.baseline_times] if r.baseline_times else None
                    ),
                    "gpu_peak_pct": r.gpu_peak_pct,
                    "gpu_avg_pct": r.gpu_avg_pct,
                    "gpu_peak_mem_mb": r.gpu_peak_mem_mb,
                }.items()
                if v is not None
            }
            for r in results
        ],
    }


def _save_reports(results: list[OpResult], mode: str, meta: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    (out_dir / f"{mode}.json").write_text(
        json.dumps(_serialize(results, mode, meta), indent=2) + "\n"
    )
    # GPU traces
    if mode == "profile":
        traces_dir = out_dir / "gpu_traces"
        traces_dir.mkdir(exist_ok=True)
        for r in results:
            if r.gpu_trace:
                (traces_dir / f"{r.name}.json").write_text(json.dumps(r.gpu_trace) + "\n")
    # Human text
    lines = [f"vibespatial.raster bench — {mode}", "=" * 60, ""]
    lines.append(f"Timestamp : {meta['timestamp']}")
    lines.append(f"Sample    : {meta['sample_name']}")
    lines.append(f"Shape     : {meta['shape_str']}  {meta['dtype']}  {meta['size_mb']:.1f} MB")
    lines.append(f"GPU       : {meta.get('gpu_name', 'N/A')}")
    lines.append(f"CuPy      : {'yes' if meta['has_cupy'] else 'no'}")
    lines.append("")
    for r in results:
        if r.status == "pass":
            if mode == "benchmark":
                bl = _fmt(r.baseline_time_s) if r.baseline_time_s else "—"
                sp = f"{r.speedup:.1f}x" if r.speedup else "—"
                lines.append(f"PASS  {r.name:<34} {_fmt(r.time_s):>9}  baseline {bl:>9}  {sp:>7}")
            elif mode == "profile":
                gp = f"{r.gpu_peak_pct:.0f}%" if r.gpu_peak_pct is not None else "—"
                ga = f"{r.gpu_avg_pct:.0f}%" if r.gpu_avg_pct is not None else "—"
                gm = f"{r.gpu_peak_mem_mb:.0f}MB" if r.gpu_peak_mem_mb is not None else "—"
                lines.append(
                    f"PASS  {r.name:<34} {_fmt(r.time_s):>9}  gpu {gp:>4}/{ga:>4}  mem {gm:>8}"
                )
            else:
                lines.append(f"PASS  {r.name:<34} {_fmt(r.time_s):>9}")
        elif r.status == "skip":
            lines.append(f"SKIP  {r.name:<34} {r.error}")
        else:
            lines.append(f"FAIL  {r.name:<34} {r.error}")
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status in ("fail", "error"))
    skipped = sum(1 for r in results if r.status == "skip")
    lines.append("")
    lines.append(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")
    (out_dir / f"{mode}.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _list_ops():
    cats: dict[str, list[OpDef]] = {}
    for o in REGISTRY.values():
        cats.setdefault(o.category, []).append(o)
    for cat, ops in cats.items():
        print(f"\n  {_BOLD}{cat}{_RESET}")
        for o in ops:
            tags = []
            if o.requires_cupy:
                tags.append("gpu")
            if o.baseline:
                tags.append("baseline")
            tag_str = f"  [{', '.join(tags)}]" if tags else ""
            print(f"    {o.name:<34}{tag_str}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench",
        description="Test, profile, and benchmark vibespatial.raster operations.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("test", "profile", "benchmark"),
        help="Execution mode",
    )
    parser.add_argument("-f", "--function", help="Run a single operation or category")
    parser.add_argument(
        "--sample",
        type=Path,
        default=_DEFAULT_SAMPLE,
        help="Path to sample GeoTIFF",
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=_DEFAULT_ITERATIONS,
        help=f"Benchmark iterations (default: {_DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=_DEFAULT_WARMUP,
        help=f"Benchmark warmup iterations (default: {_DEFAULT_WARMUP})",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("--list", action="store_true", help="List available operations")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Suppress terminal output (JSON only)",
    )

    args = parser.parse_args(argv)

    if args.list:
        _list_ops()
        return 0

    if not args.mode:
        parser.error("mode is required (test | profile | benchmark)")

    if not args.sample.exists():
        print(f"  Error: sample not found: {args.sample}", file=sys.stderr)
        return 1

    # Setup
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = args.output or (_OUTPUT_ROOT / stamp)

    gpu = GpuMonitor()
    gpu.initialize()

    ctx = TestContext(args.sample)
    if not args.json_only:
        print("  Preparing sample data ...")
    ctx.warmup()

    # Metadata
    r = ctx.raster
    shape = list(r.shape)
    meta = {
        "timestamp": datetime.now(UTC).isoformat(),
        "sample_name": args.sample.name,
        "shape": shape,
        "shape_str": " x ".join(str(s) for s in shape),
        "dtype": str(r.dtype),
        "size_mb": round(r.to_numpy().nbytes / 1024**2, 2),
        "crs": str(r.crs) if r.crs else None,
        "gpu_name": gpu.gpu_name,
        "gpu_mem_total_mb": round(gpu.gpu_mem_total_mb, 0),
        "has_cupy": has_cupy(),
        "has_pynvml": has_pynvml(),
        "python": sys.version.split()[0],
    }

    if not args.json_only:
        _print_header(meta)

    ops = _select(args.function)

    # Run
    if args.mode == "test":
        results = run_tests(ops, ctx)
    elif args.mode == "profile":
        results = run_profiles(ops, ctx, gpu)
    elif args.mode == "benchmark":
        results = run_benchmarks(ops, ctx, warmup=args.warmup, iterations=args.iterations)
    else:
        return 1

    # Save
    _save_reports(results, args.mode, meta, out_dir)

    # Latest symlink
    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    latest = _OUTPUT_ROOT / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(out_dir.name)
    except OSError:
        pass

    if not args.json_only:
        _print_summary(results, args.mode)
        print(f"  Saved to {out_dir.relative_to(_REPO)}\n")

    gpu.shutdown()
    return 1 if any(r.status in ("fail", "error") for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
