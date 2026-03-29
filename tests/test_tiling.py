"""Tests for the tiling execution engine (vibeSpatial-fx3.2, fx3.3, fx3.4).

Covers:
- WHOLE fast-path (op_fn called directly, no tiling overhead)
- TILED unary: small raster with small tile size, correct stitching
- TILED binary: same for binary operations
- Edge tile handling: raster not divisible by tile size
- Multiband tiling: 3D (bands, H, W) rasters tile on spatial dims
- Affine adjustment: each tile gets correct shifted affine
- Nodata propagation: nodata sentinel preserved through tiling
- Identity operation: tiling with identity op_fn produces exact input
- Metadata preservation: CRS, nodata, dtype preserved through tiling
- Diagnostic events: RUNTIME events appended for tiled dispatches
- Malformed plan: TILED with tile_shape=None raises ValueError
- Binary spatial mismatch: a and b with different shapes raises ValueError
- Phase 3 accumulator tiling: map-reduce for histogram/zonal-style ops

All tests use explicit RasterPlan construction -- no GPU dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticKind,
    RasterPlan,
    TilingStrategy,
    from_numpy,
)
from vibespatial.raster.tiling import (
    _adjust_affine,
    _tile_bounds,
    dispatch_tiled,
    dispatch_tiled_accumulator,
    dispatch_tiled_binary,
    dispatch_tiled_halo,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TEST_AFFINE: tuple[float, float, float, float, float, float] = (
    10.0,
    0.0,
    500.0,
    0.0,
    -10.0,
    1000.0,
)
"""(a=10, b=0, c=500, d=0, e=-10, f=1000): 10m resolution, north-up."""

_WHOLE_PLAN = RasterPlan(
    strategy=TilingStrategy.WHOLE,
    tile_shape=None,
    halo=0,
    n_tiles=0,
    estimated_vram_per_tile=0,
)

_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_raster(
    height: int = 64,
    width: int = 64,
    *,
    bands: int = 1,
    dtype: np.dtype | type = np.float32,
    nodata: float | int | None = None,
    affine: tuple[float, float, float, float, float, float] = _TEST_AFFINE,
    fill: float | None = None,
) -> OwnedRasterArray:
    """Create a synthetic HOST-resident OwnedRasterArray for testing."""
    dtype = np.dtype(dtype)
    shape = (bands, height, width) if bands > 1 else (height, width)
    if fill is not None:
        data = np.full(shape, fill, dtype=dtype)
    else:
        if np.issubdtype(dtype, np.floating):
            data = _RNG.random(shape).astype(dtype)
        else:
            data = _RNG.integers(0, 255, size=shape, dtype=dtype)
    return from_numpy(data, nodata=nodata, affine=affine, crs=None)


def _make_tiled_plan(
    tile_h: int = 32,
    tile_w: int = 32,
) -> RasterPlan:
    """Create a TILED RasterPlan with given tile dimensions."""
    return RasterPlan(
        strategy=TilingStrategy.TILED,
        tile_shape=(tile_h, tile_w),
        halo=0,
        n_tiles=0,  # n_tiles is informational; tiling computes it from shape
        estimated_vram_per_tile=0,
    )


# ---------------------------------------------------------------------------
# _tile_bounds helper
# ---------------------------------------------------------------------------


class TestTileBounds:
    def test_first_tile(self):
        rs, re, cs, ce = _tile_bounds(0, 0, 32, 32, 64, 64)
        assert (rs, re, cs, ce) == (0, 32, 0, 32)

    def test_last_tile_exact(self):
        """Raster exactly divisible by tile size."""
        rs, re, cs, ce = _tile_bounds(1, 1, 32, 32, 64, 64)
        assert (rs, re, cs, ce) == (32, 64, 32, 64)

    def test_last_tile_clamped(self):
        """Raster not divisible by tile: last tile is smaller."""
        rs, re, cs, ce = _tile_bounds(1, 1, 32, 32, 50, 50)
        assert (rs, re) == (32, 50)
        assert (cs, ce) == (32, 50)

    def test_single_tile_covers_raster(self):
        rs, re, cs, ce = _tile_bounds(0, 0, 100, 100, 50, 50)
        assert (rs, re, cs, ce) == (0, 50, 0, 50)

    def test_asymmetric_tiles(self):
        rs, re, cs, ce = _tile_bounds(0, 1, 16, 24, 64, 64)
        assert (rs, re) == (0, 16)
        assert (cs, ce) == (24, 48)


# ---------------------------------------------------------------------------
# _adjust_affine helper
# ---------------------------------------------------------------------------


class TestAdjustAffine:
    def test_no_offset(self):
        result = _adjust_affine(_TEST_AFFINE, 0, 0)
        assert result == _TEST_AFFINE

    def test_col_offset(self):
        """Shifting by col_offset adjusts x-origin by col_offset * a."""
        a, b, c, d, e, f = _TEST_AFFINE
        result = _adjust_affine(_TEST_AFFINE, row_offset=0, col_offset=10)
        expected_c = c + 10 * a
        assert result == (a, b, expected_c, d, e, f)

    def test_row_offset(self):
        """Shifting by row_offset adjusts y-origin by row_offset * e."""
        a, b, c, d, e, f = _TEST_AFFINE
        result = _adjust_affine(_TEST_AFFINE, row_offset=5, col_offset=0)
        expected_f = f + 5 * e
        assert result == (a, b, c, d, e, expected_f)

    def test_both_offsets(self):
        a, b, c, d, e, f = _TEST_AFFINE
        result = _adjust_affine(_TEST_AFFINE, row_offset=3, col_offset=7)
        expected_c = c + 7 * a + 3 * b
        expected_f = f + 7 * d + 3 * e
        assert result == (a, b, expected_c, d, e, expected_f)

    def test_rotated_affine(self):
        """Non-zero b and d (rotated grid) are handled correctly."""
        rotated = (10.0, 2.0, 500.0, 3.0, -10.0, 1000.0)
        a, b, c, d, e, f = rotated
        result = _adjust_affine(rotated, row_offset=4, col_offset=6)
        expected_c = c + 6 * a + 4 * b
        expected_f = f + 6 * d + 4 * e
        assert result == (a, b, expected_c, d, e, expected_f)


# ---------------------------------------------------------------------------
# dispatch_tiled — WHOLE fast path
# ---------------------------------------------------------------------------


class TestDispatchTiledWhole:
    def test_whole_calls_op_fn_directly(self):
        """WHOLE plan passes the raster through op_fn without tiling."""
        raster = _make_raster(64, 64)
        calls: list[OwnedRasterArray] = []

        def spy_op(r: OwnedRasterArray) -> OwnedRasterArray:
            calls.append(r)
            return r

        result = dispatch_tiled(raster, spy_op, _WHOLE_PLAN)
        assert len(calls) == 1
        assert calls[0] is raster
        assert result is raster

    def test_whole_returns_op_fn_result(self):
        raster = _make_raster(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, double_op, _WHOLE_PLAN)
        np.testing.assert_allclose(result.to_numpy(), raster.to_numpy() * 2)


# ---------------------------------------------------------------------------
# dispatch_tiled — TILED unary
# ---------------------------------------------------------------------------


class TestDispatchTiledUnary:
    def test_identity_op_exact_reconstruction(self):
        """Tiling an identity function reconstructs the original raster."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def identity(r: OwnedRasterArray) -> OwnedRasterArray:
            return r

        result = dispatch_tiled(raster, identity, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_double_op_tiled(self):
        """Doubling values through tiling matches direct doubling."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, double_op, plan)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_tiled_vs_whole_match(self):
        """Tiled and whole paths produce identical results for pointwise op."""
        raster = _make_raster(64, 64, dtype=np.float64)

        def negate_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(-r.to_numpy(), nodata=r.nodata, affine=r.affine, crs=r.crs)

        result_whole = dispatch_tiled(raster, negate_op, _WHOLE_PLAN)
        result_tiled = dispatch_tiled(raster, negate_op, _make_tiled_plan(16, 16))
        np.testing.assert_array_equal(result_tiled.to_numpy(), result_whole.to_numpy())

    def test_tile_count_matches_calls(self):
        """Op function is called exactly once per tile."""
        raster = _make_raster(64, 48)  # 64/32=2 rows, 48/32=2 cols -> 4 tiles
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def counting_op(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return r

        dispatch_tiled(raster, counting_op, plan)
        assert call_count == 4  # 2 x 2 tile grid

    def test_single_pixel_raster(self):
        """1x1 raster with tiling still works."""
        raster = _make_raster(1, 1, dtype=np.float32, fill=42.0)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.to_numpy().item() == pytest.approx(42.0)

    def test_very_small_tiles(self):
        """1x1 tiles process every pixel independently."""
        raster = _make_raster(4, 4, dtype=np.float32)
        plan = _make_tiled_plan(1, 1)
        call_count = 0

        def counting_identity(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return r

        result = dispatch_tiled(raster, counting_identity, plan)
        assert call_count == 16  # 4 x 4
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())


# ---------------------------------------------------------------------------
# Edge tile handling
# ---------------------------------------------------------------------------


class TestEdgeTiles:
    def test_raster_not_divisible_by_tile(self):
        """50x50 raster with 32x32 tiles: edge tiles are 18 pixels wide/tall."""
        raster = _make_raster(50, 50, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)
        tile_shapes: list[tuple[int, ...]] = []

        def shape_spy(r: OwnedRasterArray) -> OwnedRasterArray:
            tile_shapes.append(r.shape)
            return r

        result = dispatch_tiled(raster, shape_spy, plan)

        # 2x2 tile grid
        assert len(tile_shapes) == 4
        assert tile_shapes[0] == (32, 32)  # top-left
        assert tile_shapes[1] == (32, 18)  # top-right (50-32=18)
        assert tile_shapes[2] == (18, 32)  # bottom-left
        assert tile_shapes[3] == (18, 18)  # bottom-right

        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_raster_smaller_than_tile(self):
        """Raster smaller than tile -> one tile covers everything."""
        raster = _make_raster(10, 15, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def counting_identity(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            assert r.height == 10
            assert r.width == 15
            return r

        result = dispatch_tiled(raster, counting_identity, plan)
        assert call_count == 1
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_asymmetric_tile_and_raster(self):
        """Non-square raster with non-square tiles."""
        raster = _make_raster(100, 60, dtype=np.int16)
        plan = _make_tiled_plan(40, 25)  # 3 row tiles, 3 col tiles (100/40, 60/25)
        call_count = 0

        def counting_identity(r: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return r

        result = dispatch_tiled(raster, counting_identity, plan)
        # rows: ceil(100/40)=3, cols: ceil(60/25)=3 -> 9 tiles
        assert call_count == 9
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())


# ---------------------------------------------------------------------------
# Multiband tiling
# ---------------------------------------------------------------------------


class TestMultibandTiling:
    def test_3band_identity(self):
        """3-band raster tiled correctly on spatial dims."""
        raster = _make_raster(64, 64, bands=3, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())
        assert result.band_count == 3
        assert result.shape == (3, 64, 64)

    def test_3band_edge_tiles(self):
        """Multiband raster with edge tiles."""
        raster = _make_raster(50, 50, bands=3, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, double_op, plan)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.shape == (3, 50, 50)

    def test_tile_slices_all_bands(self):
        """Each tile receives all bands for its spatial extent."""
        raster = _make_raster(32, 32, bands=4, dtype=np.uint8)
        plan = _make_tiled_plan(16, 16)

        def check_bands(r: OwnedRasterArray) -> OwnedRasterArray:
            assert r.band_count == 4
            assert r.to_numpy().ndim == 3
            assert r.to_numpy().shape[0] == 4
            return r

        dispatch_tiled(raster, check_bands, plan)


# ---------------------------------------------------------------------------
# Affine adjustment per tile
# ---------------------------------------------------------------------------


class TestAffinePerTile:
    def test_tile_affine_is_shifted(self):
        """Each tile receives an affine shifted to its spatial position."""
        raster = _make_raster(
            64,
            64,
            affine=(10.0, 0.0, 500.0, 0.0, -10.0, 1000.0),
        )
        plan = _make_tiled_plan(32, 32)
        tile_affines: list[tuple[float, ...]] = []

        def affine_spy(r: OwnedRasterArray) -> OwnedRasterArray:
            tile_affines.append(r.affine)
            return r

        dispatch_tiled(raster, affine_spy, plan)

        # 2x2 grid -> 4 tiles
        assert len(tile_affines) == 4

        a, b, c, d, e, f = 10.0, 0.0, 500.0, 0.0, -10.0, 1000.0

        # Tile (0,0): no offset
        assert tile_affines[0] == (a, b, c, d, e, f)
        # Tile (0,1): col_offset=32 -> new_c = 500 + 32*10 = 820
        assert tile_affines[1] == (a, b, c + 32 * a, d, e, f)
        # Tile (1,0): row_offset=32 -> new_f = 1000 + 32*(-10) = 680
        assert tile_affines[2] == (a, b, c, d, e, f + 32 * e)
        # Tile (1,1): both offsets
        assert tile_affines[3] == (a, b, c + 32 * a, d, e, f + 32 * e)

    def test_output_has_original_affine(self):
        """The assembled output raster uses the full (un-shifted) affine."""
        raster = _make_raster(64, 64, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.affine == _TEST_AFFINE


# ---------------------------------------------------------------------------
# Nodata propagation
# ---------------------------------------------------------------------------


class TestNodataPropagation:
    def test_nodata_preserved_through_identity(self):
        """Nodata sentinel is propagated to the output raster."""
        raster = _make_raster(64, 64, dtype=np.float32, nodata=-9999.0)
        # Place nodata in known positions
        data = raster.to_numpy().copy()
        data[0, 0] = -9999.0
        data[31, 31] = -9999.0
        data[63, 63] = -9999.0
        raster = from_numpy(data, nodata=-9999.0, affine=_TEST_AFFINE)

        plan = _make_tiled_plan(32, 32)
        result = dispatch_tiled(raster, lambda r: r, plan)

        assert result.nodata == -9999.0
        result_data = result.to_numpy()
        assert result_data[0, 0] == -9999.0
        assert result_data[31, 31] == -9999.0
        assert result_data[63, 63] == -9999.0

    def test_nodata_none_preserved(self):
        """Raster with nodata=None stays nodata=None after tiling."""
        raster = _make_raster(32, 32, nodata=None)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.nodata is None

    def test_integer_nodata(self):
        """Integer nodata (e.g., 255 for uint8) is preserved."""
        raster = _make_raster(32, 32, dtype=np.uint8, nodata=255)
        data = raster.to_numpy().copy()
        data[0, 0] = 255
        data[15, 15] = 255
        raster = from_numpy(data, nodata=255, affine=_TEST_AFFINE)

        plan = _make_tiled_plan(16, 16)
        result = dispatch_tiled(raster, lambda r: r, plan)

        assert result.nodata == 255
        assert result.to_numpy()[0, 0] == 255
        assert result.to_numpy()[15, 15] == 255

    def test_nan_nodata_float(self):
        """NaN nodata in float rasters is preserved through tiling."""
        data = _RNG.random((32, 32)).astype(np.float32)
        data[5, 5] = np.nan
        data[20, 20] = np.nan
        raster = from_numpy(data, nodata=float("nan"), affine=_TEST_AFFINE)

        plan = _make_tiled_plan(16, 16)
        result = dispatch_tiled(raster, lambda r: r, plan)

        assert result.nodata is not None and np.isnan(result.nodata)
        assert np.isnan(result.to_numpy()[5, 5])
        assert np.isnan(result.to_numpy()[20, 20])


# ---------------------------------------------------------------------------
# Dtype preservation
# ---------------------------------------------------------------------------


class TestDtypePreservation:
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32, np.float64])
    def test_dtype_roundtrip(self, dtype):
        raster = _make_raster(32, 32, dtype=dtype)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_dtype_changing_op_unary(self):
        """Op that changes dtype (float32 -> uint8) works via lazy allocation."""
        raster = _make_raster(32, 32, dtype=np.float32, fill=128.5)
        plan = _make_tiled_plan(16, 16)

        def to_uint8(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy().astype(np.uint8),
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled(raster, to_uint8, plan)
        assert result.dtype == np.dtype(np.uint8)
        np.testing.assert_array_equal(result.to_numpy(), 128)


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


class TestMetadataPreservation:
    def test_affine_preserved(self):
        custom_affine = (5.0, 0.0, 100.0, 0.0, -5.0, 200.0)
        raster = _make_raster(32, 32, affine=custom_affine)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.affine == custom_affine

    def test_nodata_preserved(self):
        raster = _make_raster(32, 32, nodata=-9999.0)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.nodata == -9999.0

    def test_crs_preserved(self):
        """CRS is propagated (None in tests, but slot preserved)."""
        raster = _make_raster(32, 32)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.crs == raster.crs


# ---------------------------------------------------------------------------
# Diagnostic events
# ---------------------------------------------------------------------------


class TestDiagnostics:
    def test_tiled_unary_has_runtime_event(self):
        raster = _make_raster(64, 64)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        runtime_events = [
            ev for ev in result.diagnostics if ev.kind == RasterDiagnosticKind.RUNTIME
        ]
        assert len(runtime_events) >= 1
        evt = runtime_events[-1]
        assert "dispatch_tiled" in evt.detail
        assert "unary" in evt.detail
        assert "tiles=" in evt.detail

    def test_tiled_binary_has_runtime_event(self):
        a = _make_raster(64, 64)
        b = _make_raster(64, 64)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        runtime_events = [
            ev for ev in result.diagnostics if ev.kind == RasterDiagnosticKind.RUNTIME
        ]
        assert len(runtime_events) >= 1
        evt = runtime_events[-1]
        assert "dispatch_tiled_binary" in evt.detail
        assert "tiles=" in evt.detail

    def test_whole_path_no_tiling_diagnostic(self):
        """WHOLE path delegates to op_fn -- no tiling diagnostic appended."""
        raster = _make_raster(32, 32)

        result = dispatch_tiled(raster, lambda r: r, _WHOLE_PLAN)
        # The WHOLE path returns op_fn's result directly; dispatch_tiled
        # does not wrap it with an extra diagnostic.
        tiling_events = [
            ev
            for ev in result.diagnostics
            if ev.kind == RasterDiagnosticKind.RUNTIME and "dispatch_tiled" in ev.detail
        ]
        assert len(tiling_events) == 0


# ---------------------------------------------------------------------------
# dispatch_tiled_binary — WHOLE fast path
# ---------------------------------------------------------------------------


class TestDispatchTiledBinaryWhole:
    def test_whole_calls_op_fn_directly(self):
        a = _make_raster(32, 32, fill=3.0)
        b = _make_raster(32, 32, fill=7.0)
        calls: list[tuple[OwnedRasterArray, OwnedRasterArray]] = []

        def spy_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            calls.append((x, y))
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, spy_op, _WHOLE_PLAN)
        assert len(calls) == 1
        assert calls[0][0] is a
        assert calls[0][1] is b
        np.testing.assert_allclose(result.to_numpy(), 10.0)


# ---------------------------------------------------------------------------
# dispatch_tiled_binary — TILED path
# ---------------------------------------------------------------------------


class TestDispatchTiledBinary:
    def test_add_tiled(self):
        """Binary add through tiling matches direct addition."""
        a = _make_raster(64, 64, dtype=np.float32)
        b = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result_tiled = dispatch_tiled_binary(a, b, add_op, plan)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_allclose(result_tiled.to_numpy(), expected)

    def test_tiled_vs_whole_binary(self):
        """Tiled and whole binary paths produce identical results."""
        a = _make_raster(64, 64, dtype=np.float64)
        b = _make_raster(64, 64, dtype=np.float64)

        def mul_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() * y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result_whole = dispatch_tiled_binary(a, b, mul_op, _WHOLE_PLAN)
        result_tiled = dispatch_tiled_binary(a, b, mul_op, _make_tiled_plan(16, 16))
        np.testing.assert_array_equal(result_tiled.to_numpy(), result_whole.to_numpy())

    def test_binary_tile_count(self):
        """Binary tiling calls op_fn once per tile."""
        a = _make_raster(48, 64)
        b = _make_raster(48, 64)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def counting_add(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            nonlocal call_count
            call_count += 1
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        dispatch_tiled_binary(a, b, counting_add, plan)
        # ceil(48/32)=2 rows, ceil(64/32)=2 cols -> 4 tiles
        assert call_count == 4

    def test_binary_edge_tiles(self):
        """Binary tiling with non-divisible raster dimensions."""
        a = _make_raster(50, 35, dtype=np.float32)
        b = _make_raster(50, 35, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_binary_multiband(self):
        """3-band binary tiling."""
        a = _make_raster(64, 64, bands=3, dtype=np.float32)
        b = _make_raster(64, 64, bands=3, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        expected = a.to_numpy() + b.to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.band_count == 3

    def test_binary_nodata_propagation(self):
        """Nodata from either input is preserved in binary tiling output."""
        data_a = np.ones((32, 32), dtype=np.float32)
        data_b = np.ones((32, 32), dtype=np.float32) * 2
        data_a[0, 0] = -9999.0
        data_b[15, 15] = -9999.0

        a = from_numpy(data_a, nodata=-9999.0, affine=_TEST_AFFINE)
        b = from_numpy(data_b, nodata=-9999.0, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(16, 16)

        def add_with_nodata(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            xd = x.to_numpy()
            yd = y.to_numpy()
            result = xd + yd
            # Propagate nodata: if either is nodata, output is nodata
            mask = (xd == -9999.0) | (yd == -9999.0)
            result[mask] = -9999.0
            return from_numpy(result, nodata=-9999.0, affine=x.affine, crs=x.crs)

        result = dispatch_tiled_binary(a, b, add_with_nodata, plan)
        rd = result.to_numpy()
        assert rd[0, 0] == -9999.0
        assert rd[15, 15] == -9999.0
        # Non-nodata pixels have correct sum
        assert rd[1, 1] == pytest.approx(3.0)

    def test_binary_metadata_preserved(self):
        """Output preserves a's affine and CRS."""
        custom_affine = (5.0, 0.0, 100.0, 0.0, -5.0, 200.0)
        a = _make_raster(32, 32, affine=custom_affine, nodata=-1.0)
        b = _make_raster(32, 32, affine=custom_affine, nodata=-1.0)
        plan = _make_tiled_plan(16, 16)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        result = dispatch_tiled_binary(a, b, add_op, plan)
        assert result.affine == custom_affine
        assert result.crs is None


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_tiled_plan_without_tile_shape_raises(self):
        """TILED strategy with tile_shape=None is a malformed plan."""
        bad_plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=None,
            halo=0,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        raster = _make_raster(32, 32)

        with pytest.raises(ValueError, match="tile_shape is None"):
            dispatch_tiled(raster, lambda r: r, bad_plan)

    def test_binary_tiled_plan_without_tile_shape_raises(self):
        bad_plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=None,
            halo=0,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        a = _make_raster(32, 32)
        b = _make_raster(32, 32)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        with pytest.raises(ValueError, match="tile_shape is None"):
            dispatch_tiled_binary(a, b, add_op, bad_plan)

    def test_device_resident_unary_raises(self):
        """DEVICE-resident input on TILED path raises ValueError."""
        from vibespatial.residency import Residency

        raster = _make_raster(32, 32)
        raster.residency = Residency.DEVICE  # simulate device-resident
        plan = _make_tiled_plan(16, 16)

        with pytest.raises(ValueError, match="HOST-resident"):
            dispatch_tiled(raster, lambda r: r, plan)

    def test_device_resident_binary_raises(self):
        """DEVICE-resident input on binary TILED path raises ValueError."""
        from vibespatial.residency import Residency

        a = _make_raster(32, 32)
        b = _make_raster(32, 32)
        a.residency = Residency.DEVICE  # simulate device-resident
        plan = _make_tiled_plan(16, 16)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        with pytest.raises(ValueError, match="HOST-resident"):
            dispatch_tiled_binary(a, b, add_op, plan)

    def test_binary_spatial_mismatch_raises(self):
        """Mismatched spatial dimensions raise ValueError."""
        a = _make_raster(32, 32)
        b = _make_raster(32, 64)  # different width
        plan = _make_tiled_plan(16, 16)

        def add_op(x: OwnedRasterArray, y: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                x.to_numpy() + y.to_numpy(),
                nodata=x.nodata,
                affine=x.affine,
                crs=x.crs,
            )

        with pytest.raises(ValueError, match="Spatial dimension mismatch"):
            dispatch_tiled_binary(a, b, add_op, plan)


# ---------------------------------------------------------------------------
# Constant-value and all-nodata edge cases
# ---------------------------------------------------------------------------


class TestConstantAndAllNodata:
    def test_constant_value_raster(self):
        raster = _make_raster(64, 64, fill=42.0, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        result = dispatch_tiled(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), 42.0)

    def test_all_nodata_raster(self):
        """Raster where every pixel is nodata."""
        data = np.full((32, 32), -9999.0, dtype=np.float32)
        raster = from_numpy(data, nodata=-9999.0, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(16, 16)

        result = dispatch_tiled(raster, lambda r: r, plan)
        assert result.nodata == -9999.0
        np.testing.assert_array_equal(result.to_numpy(), -9999.0)


# ---------------------------------------------------------------------------
# Lazy import via __init__.py
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_dispatch_tiled_importable(self):
        from vibespatial.raster import dispatch_tiled as dt

        assert callable(dt)

    def test_dispatch_tiled_binary_importable(self):
        from vibespatial.raster import dispatch_tiled_binary as dtb

        assert callable(dtb)

    def test_dispatch_tiled_halo_importable(self):
        from vibespatial.raster import dispatch_tiled_halo as dth

        assert callable(dth)


# ===========================================================================
# Phase 2: dispatch_tiled_halo — halo tiling for stencil operations
# (vibeSpatial-fx3.3)
# ===========================================================================


def _make_halo_plan(
    tile_h: int = 32,
    tile_w: int = 32,
    halo: int = 1,
) -> RasterPlan:
    """Create a TILED RasterPlan with given tile dimensions and halo."""
    return RasterPlan(
        strategy=TilingStrategy.TILED,
        tile_shape=(tile_h, tile_w),
        halo=halo,
        n_tiles=0,
        estimated_vram_per_tile=0,
    )


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — WHOLE fast path
# ---------------------------------------------------------------------------


class TestHaloWholeFastPath:
    def test_whole_calls_op_fn_directly(self):
        """WHOLE plan passes the raster through op_fn without tiling."""
        raster = _make_raster(64, 64)
        calls: list[OwnedRasterArray] = []

        def spy_op(r: OwnedRasterArray) -> OwnedRasterArray:
            calls.append(r)
            return r

        result = dispatch_tiled_halo(raster, spy_op, _WHOLE_PLAN)
        assert len(calls) == 1
        assert calls[0] is raster
        assert result is raster

    def test_whole_returns_op_fn_result(self):
        raster = _make_raster(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled_halo(raster, double_op, _WHOLE_PLAN)
        np.testing.assert_allclose(result.to_numpy(), raster.to_numpy() * 2)


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — identity reconstruction
# ---------------------------------------------------------------------------


class TestHaloIdentityReconstruction:
    def test_identity_with_halo_exact_reconstruction(self):
        """Tiling an identity function with halo reconstructs the original."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_halo_plan(32, 32, halo=2)

        def identity(r: OwnedRasterArray) -> OwnedRasterArray:
            return r

        result = dispatch_tiled_halo(raster, identity, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())
        assert result.shape == raster.shape

    def test_identity_large_halo(self):
        """Halo larger than half the tile still reconstructs correctly."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_halo_plan(16, 16, halo=8)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_identity_halo_1(self):
        """Minimal halo=1 with identity op reconstructs exactly."""
        raster = _make_raster(48, 48, dtype=np.float64)
        plan = _make_halo_plan(16, 16, halo=1)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — edge tile handling
# ---------------------------------------------------------------------------


class TestHaloEdgeTiles:
    def test_non_divisible_raster(self):
        """50x50 raster with 32x32 tiles and halo=2: edge tiles work."""
        raster = _make_raster(50, 50, dtype=np.float32)
        plan = _make_halo_plan(32, 32, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_edge_tiles_have_reduced_halo(self):
        """At raster boundary, physical tile halo is clamped.

        For tile (0,0), the top and left halo is 0 because there are no
        pixels above or to the left of the raster.  The physical tile
        should be larger than the effective tile but only on the bottom
        and right edges.
        """
        raster = _make_raster(50, 50, dtype=np.float32)
        halo = 3
        plan = _make_halo_plan(25, 25, halo=halo)
        physical_shapes: list[tuple[int, ...]] = []

        def shape_spy(r: OwnedRasterArray) -> OwnedRasterArray:
            physical_shapes.append(r.shape)
            return r

        dispatch_tiled_halo(raster, shape_spy, plan)

        # 2x2 tile grid for 50x50 with 25x25 effective tiles
        assert len(physical_shapes) == 4

        # Tile (0,0): top/left halo = 0, bottom/right halo = 3 -> (28, 28)
        assert physical_shapes[0] == (25 + halo, 25 + halo)
        # Tile (0,1): top halo = 0, left halo = 3, bottom halo = 3,
        #             right halo = 0 (clamped at 50) -> (28, 28)
        assert physical_shapes[1] == (25 + halo, 25 + halo)
        # Tile (1,0): top halo = 3, left halo = 0, bottom = 0,
        #             right halo = 3 -> (28, 28)
        assert physical_shapes[2] == (25 + halo, 25 + halo)
        # Tile (1,1): top = 3, left = 3, bottom = 0, right = 0 -> (28, 28)
        assert physical_shapes[3] == (25 + halo, 25 + halo)

    def test_raster_smaller_than_tile(self):
        """Raster smaller than tile -> single tile with reduced halo."""
        raster = _make_raster(10, 15, dtype=np.float32)
        plan = _make_halo_plan(32, 32, halo=4)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_single_pixel_raster_with_halo(self):
        """1x1 raster with halo -- halo is clamped to 0 on all edges."""
        raster = _make_raster(1, 1, dtype=np.float32, fill=99.0)
        plan = _make_halo_plan(32, 32, halo=5)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.to_numpy().item() == pytest.approx(99.0)
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — trim correctness
# ---------------------------------------------------------------------------


class TestHaloTrimCorrectness:
    def test_stencil_op_trims_halo_artifacts(self):
        """A stencil op that writes -1 at edges has those artifacts trimmed.

        The op writes -1 to a 1-pixel border of its input (simulating
        stencil boundary artifacts).  Only the interior should appear in
        the output.  With proper trim, the output equals the identity
        everywhere except at the *raster* boundary where the first/last
        row/col tiles have no halo on that edge.
        """
        raster = _make_raster(64, 64, dtype=np.float32, fill=1.0)
        halo = 2
        plan = _make_halo_plan(32, 32, halo=halo)

        def border_corrupt_op(r: OwnedRasterArray) -> OwnedRasterArray:
            """Identity but writes -1 to a halo-width border."""
            data = r.to_numpy().copy()
            h, w = data.shape[-2], data.shape[-1]
            if data.ndim == 3:
                data[:, :halo, :] = -1.0
                data[:, h - halo :, :] = -1.0
                data[:, :, :halo] = -1.0
                data[:, :, w - halo :] = -1.0
            else:
                data[:halo, :] = -1.0
                data[h - halo :, :] = -1.0
                data[:, :halo] = -1.0
                data[:, w - halo :] = -1.0
            return from_numpy(data, nodata=r.nodata, affine=r.affine, crs=r.crs)

        result = dispatch_tiled_halo(raster, border_corrupt_op, plan)
        rd = result.to_numpy()

        # Interior tiles (not at raster boundary) should be fully clean.
        # The effective interior excludes the first/last 'halo' rows/cols
        # of the raster because those tiles have no halo to trim on the
        # boundary side -- the border_corrupt_op corrupts those pixels.
        inner = rd[halo : 64 - halo, halo : 64 - halo]
        np.testing.assert_array_equal(inner, 1.0)

    def test_physical_tile_larger_than_effective(self):
        """Each physical tile is strictly larger than the effective tile
        (except when the entire raster fits in one tile)."""
        raster = _make_raster(64, 64, dtype=np.float32)
        halo = 3
        plan = _make_halo_plan(32, 32, halo=halo)
        physical_shapes: list[tuple[int, int]] = []
        effective_shapes: list[tuple[int, int]] = []

        _call_idx = [0]

        def shape_tracker(r: OwnedRasterArray) -> OwnedRasterArray:
            h, w = r.height, r.width
            physical_shapes.append((h, w))
            # Compute effective shape from tile grid position
            tile_h, tile_w = 32, 32
            raster_h, raster_w = 64, 64
            tr = _call_idx[0] // 2
            tc = _call_idx[0] % 2
            rs, re, cs, ce = _tile_bounds(tr, tc, tile_h, tile_w, raster_h, raster_w)
            effective_shapes.append((re - rs, ce - cs))
            _call_idx[0] += 1
            return r

        dispatch_tiled_halo(raster, shape_tracker, plan)

        for phys, eff in zip(physical_shapes, effective_shapes):
            assert phys[0] >= eff[0]
            assert phys[1] >= eff[1]
            # At least one dimension should have halo expansion
            assert phys[0] > eff[0] or phys[1] > eff[1]


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — multiband
# ---------------------------------------------------------------------------


class TestHaloMultiband:
    def test_3band_identity_with_halo(self):
        """3-band raster with halo tiles correctly on spatial dims."""
        raster = _make_raster(64, 64, bands=3, dtype=np.float32)
        plan = _make_halo_plan(32, 32, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())
        assert result.band_count == 3
        assert result.shape == (3, 64, 64)

    def test_multiband_edge_tiles_with_halo(self):
        """Multiband raster with non-divisible dims and halo."""
        raster = _make_raster(50, 50, bands=3, dtype=np.float32)
        plan = _make_halo_plan(32, 32, halo=3)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result = dispatch_tiled_halo(raster, double_op, plan)
        expected = raster.to_numpy() * 2
        np.testing.assert_allclose(result.to_numpy(), expected)
        assert result.shape == (3, 50, 50)

    def test_multiband_tile_slices_all_bands(self):
        """Each halo tile receives all bands."""
        raster = _make_raster(32, 32, bands=4, dtype=np.uint8)
        plan = _make_halo_plan(16, 16, halo=2)

        def check_bands(r: OwnedRasterArray) -> OwnedRasterArray:
            assert r.band_count == 4
            assert r.to_numpy().ndim == 3
            assert r.to_numpy().shape[0] == 4
            return r

        dispatch_tiled_halo(raster, check_bands, plan)


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — affine adjustment
# ---------------------------------------------------------------------------


class TestHaloAffineAdjustment:
    def test_tile_affine_shifted_to_physical_position(self):
        """Each tile's affine is shifted to the physical tile origin,
        not the effective origin."""
        affine = (10.0, 0.0, 500.0, 0.0, -10.0, 1000.0)
        raster = _make_raster(64, 64, affine=affine)
        halo = 2
        plan = _make_halo_plan(32, 32, halo=halo)
        tile_affines: list[tuple[float, ...]] = []

        def affine_spy(r: OwnedRasterArray) -> OwnedRasterArray:
            tile_affines.append(r.affine)
            return r

        dispatch_tiled_halo(raster, affine_spy, plan)

        a, b, c, d, e, f = affine

        # 2x2 tile grid -> 4 tiles
        assert len(tile_affines) == 4

        # Tile (0,0): phys_rs=max(0,0-2)=0, phys_cs=max(0,0-2)=0
        assert tile_affines[0] == (a, b, c, d, e, f)

        # Tile (0,1): eff_cs=32, phys_cs=max(0,32-2)=30
        expected_c_01 = c + 30 * a
        assert tile_affines[1] == (a, b, expected_c_01, d, e, f)

        # Tile (1,0): eff_rs=32, phys_rs=max(0,32-2)=30
        expected_f_10 = f + 30 * e
        assert tile_affines[2] == (a, b, c, d, e, expected_f_10)

        # Tile (1,1): phys_rs=30, phys_cs=30
        expected_c_11 = c + 30 * a
        expected_f_11 = f + 30 * e
        assert tile_affines[3] == (a, b, expected_c_11, d, e, expected_f_11)

    def test_output_has_original_affine(self):
        """The assembled output raster uses the full (un-shifted) affine."""
        raster = _make_raster(64, 64, affine=_TEST_AFFINE)
        plan = _make_halo_plan(32, 32, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.affine == _TEST_AFFINE


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — metadata preservation
# ---------------------------------------------------------------------------


class TestHaloMetadataPreservation:
    def test_affine_preserved(self):
        custom_affine = (5.0, 0.0, 100.0, 0.0, -5.0, 200.0)
        raster = _make_raster(32, 32, affine=custom_affine)
        plan = _make_halo_plan(16, 16, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.affine == custom_affine

    def test_nodata_preserved(self):
        raster = _make_raster(32, 32, nodata=-9999.0)
        plan = _make_halo_plan(16, 16, halo=1)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.nodata == -9999.0

    def test_crs_preserved(self):
        raster = _make_raster(32, 32)
        plan = _make_halo_plan(16, 16, halo=1)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.crs == raster.crs

    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32, np.float64])
    def test_dtype_roundtrip(self, dtype):
        raster = _make_raster(32, 32, dtype=dtype)
        plan = _make_halo_plan(16, 16, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(result.to_numpy(), raster.to_numpy())

    def test_nodata_positions_preserved(self):
        """Nodata sentinel values at known positions survive halo tiling."""
        data = _RNG.random((64, 64)).astype(np.float32)
        data[0, 0] = -9999.0
        data[31, 31] = -9999.0
        data[63, 63] = -9999.0
        raster = from_numpy(data, nodata=-9999.0, affine=_TEST_AFFINE)
        plan = _make_halo_plan(32, 32, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        rd = result.to_numpy()
        assert rd[0, 0] == -9999.0
        assert rd[31, 31] == -9999.0
        assert rd[63, 63] == -9999.0
        assert result.nodata == -9999.0

    def test_nan_nodata_preserved(self):
        """NaN nodata in float rasters is preserved through halo tiling."""
        data = _RNG.random((32, 32)).astype(np.float32)
        data[5, 5] = np.nan
        data[20, 20] = np.nan
        raster = from_numpy(data, nodata=float("nan"), affine=_TEST_AFFINE)
        plan = _make_halo_plan(16, 16, halo=1)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.nodata is not None and np.isnan(result.nodata)
        assert np.isnan(result.to_numpy()[5, 5])
        assert np.isnan(result.to_numpy()[20, 20])


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — halo=0 matches dispatch_tiled
# ---------------------------------------------------------------------------


class TestHaloZeroMatchesDispatchTiled:
    def test_halo_zero_matches_dispatch_tiled_identity(self):
        """When halo=0, dispatch_tiled_halo produces the same result as
        dispatch_tiled for an identity op."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan_halo = _make_halo_plan(32, 32, halo=0)
        plan_plain = _make_tiled_plan(32, 32)

        result_halo = dispatch_tiled_halo(raster, lambda r: r, plan_halo)
        result_plain = dispatch_tiled(raster, lambda r: r, plan_plain)
        np.testing.assert_array_equal(result_halo.to_numpy(), result_plain.to_numpy())

    def test_halo_zero_matches_dispatch_tiled_doubling(self):
        """halo=0 with a doubling op matches dispatch_tiled."""
        raster = _make_raster(50, 50, dtype=np.float32)
        plan_halo = _make_halo_plan(32, 32, halo=0)
        plan_plain = _make_tiled_plan(32, 32)

        def double_op(r: OwnedRasterArray) -> OwnedRasterArray:
            return from_numpy(
                r.to_numpy() * 2,
                nodata=r.nodata,
                affine=r.affine,
                crs=r.crs,
            )

        result_halo = dispatch_tiled_halo(raster, double_op, plan_halo)
        result_plain = dispatch_tiled(raster, double_op, plan_plain)
        np.testing.assert_array_equal(result_halo.to_numpy(), result_plain.to_numpy())

    def test_halo_zero_multiband(self):
        """halo=0 multiband matches dispatch_tiled."""
        raster = _make_raster(64, 64, bands=3, dtype=np.float32)
        plan_halo = _make_halo_plan(32, 32, halo=0)
        plan_plain = _make_tiled_plan(32, 32)

        result_halo = dispatch_tiled_halo(raster, lambda r: r, plan_halo)
        result_plain = dispatch_tiled(raster, lambda r: r, plan_plain)
        np.testing.assert_array_equal(result_halo.to_numpy(), result_plain.to_numpy())


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — diagnostics
# ---------------------------------------------------------------------------


class TestHaloDiagnostics:
    def test_tiled_halo_has_runtime_event(self):
        raster = _make_raster(64, 64)
        plan = _make_halo_plan(32, 32, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        runtime_events = [
            ev for ev in result.diagnostics if ev.kind == RasterDiagnosticKind.RUNTIME
        ]
        assert len(runtime_events) >= 1
        evt = runtime_events[-1]
        assert "dispatch_tiled_halo" in evt.detail
        assert "tiles=" in evt.detail
        assert "halo=" in evt.detail

    def test_diagnostic_includes_halo_value(self):
        """The diagnostic detail string contains the actual halo value."""
        raster = _make_raster(32, 32)
        plan = _make_halo_plan(16, 16, halo=5)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        runtime_events = [
            ev for ev in result.diagnostics if ev.kind == RasterDiagnosticKind.RUNTIME
        ]
        assert any("halo=5" in ev.detail for ev in runtime_events)

    def test_whole_path_no_tiling_diagnostic(self):
        """WHOLE path delegates to op_fn -- no tiling diagnostic appended."""
        raster = _make_raster(32, 32)

        result = dispatch_tiled_halo(raster, lambda r: r, _WHOLE_PLAN)
        tiling_events = [
            ev
            for ev in result.diagnostics
            if ev.kind == RasterDiagnosticKind.RUNTIME and "dispatch_tiled_halo" in ev.detail
        ]
        assert len(tiling_events) == 0


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — error handling
# ---------------------------------------------------------------------------


class TestHaloErrorHandling:
    def test_tiled_plan_without_tile_shape_raises(self):
        bad_plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=None,
            halo=2,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        raster = _make_raster(32, 32)

        with pytest.raises(ValueError, match="tile_shape is None"):
            dispatch_tiled_halo(raster, lambda r: r, bad_plan)

    def test_device_resident_raises(self):
        """DEVICE-resident input on TILED path raises ValueError."""
        from vibespatial.residency import Residency

        raster = _make_raster(32, 32)
        raster.residency = Residency.DEVICE
        plan = _make_halo_plan(16, 16, halo=2)

        with pytest.raises(ValueError, match="HOST-resident"):
            dispatch_tiled_halo(raster, lambda r: r, plan)


# ---------------------------------------------------------------------------
# dispatch_tiled_halo — constant-value and all-nodata edge cases
# ---------------------------------------------------------------------------


class TestHaloConstantAndAllNodata:
    def test_constant_value_raster(self):
        raster = _make_raster(64, 64, fill=42.0, dtype=np.float32)
        plan = _make_halo_plan(32, 32, halo=3)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        np.testing.assert_array_equal(result.to_numpy(), 42.0)

    def test_all_nodata_raster(self):
        data = np.full((32, 32), -9999.0, dtype=np.float32)
        raster = from_numpy(data, nodata=-9999.0, affine=_TEST_AFFINE)
        plan = _make_halo_plan(16, 16, halo=2)

        result = dispatch_tiled_halo(raster, lambda r: r, plan)
        assert result.nodata == -9999.0
        np.testing.assert_array_equal(result.to_numpy(), -9999.0)


# ===========================================================================
# Phase 3: dispatch_tiled_accumulator — accumulator tiling for reduce ops
# (vibeSpatial-fx3.4)
# ===========================================================================


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — WHOLE fast path
# ---------------------------------------------------------------------------


class TestAccumulatorWhole:
    def test_whole_calls_tile_fn_once(self):
        """WHOLE plan passes the raster through tile_fn directly, no merge."""
        raster = _make_raster(32, 32, dtype=np.float32)
        calls: list[OwnedRasterArray] = []

        def tile_fn(r: OwnedRasterArray) -> float:
            calls.append(r)
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            raise AssertionError("merge_fn should not be called on WHOLE path")

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, _WHOLE_PLAN)
        assert len(calls) == 1
        assert calls[0] is raster
        assert isinstance(result, float)

    def test_whole_returns_tile_fn_result(self):
        raster = _make_raster(32, 32, dtype=np.float32, fill=2.0)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, _WHOLE_PLAN)
        expected = 2.0 * 32 * 32
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — sum accumulator
# ---------------------------------------------------------------------------


class TestAccumulatorSum:
    def test_tiled_sum_matches_whole(self):
        """tile_fn returns pixel sum, merge_fn adds. Tiled sum matches non-tiled."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        result_tiled = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        result_whole = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, _WHOLE_PLAN)
        assert result_tiled == pytest.approx(result_whole, rel=1e-5)

    def test_tiled_sum_small_tiles(self):
        """16x16 tiles on 64x64 raster (16 tiles)."""
        raster = _make_raster(64, 64, dtype=np.float64, fill=1.0)
        plan = _make_tiled_plan(16, 16)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result == pytest.approx(64 * 64)

    def test_tiled_sum_tile_count(self):
        """tile_fn is called once per tile."""
        raster = _make_raster(64, 48)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def tile_fn(r: OwnedRasterArray) -> float:
            nonlocal call_count
            call_count += 1
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        # ceil(64/32)=2 rows, ceil(48/32)=2 cols -> 4 tiles
        assert call_count == 4


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — min/max accumulator
# ---------------------------------------------------------------------------


class TestAccumulatorMinMax:
    def test_tiled_minmax_matches_whole(self):
        """tile_fn returns (min, max), merge_fn takes overall min/max."""
        raster = _make_raster(64, 64, dtype=np.float32)
        plan = _make_tiled_plan(32, 32)

        def tile_fn(r: OwnedRasterArray) -> tuple[float, float]:
            data = r.to_numpy()
            return (float(data.min()), float(data.max()))

        def merge_fn(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
            return (min(a[0], b[0]), max(a[1], b[1]))

        result_tiled = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        result_whole = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, _WHOLE_PLAN)
        assert result_tiled[0] == pytest.approx(result_whole[0])
        assert result_tiled[1] == pytest.approx(result_whole[1])

    def test_minmax_known_values(self):
        """Known min/max in specific positions are found correctly."""
        data = np.ones((32, 32), dtype=np.float32) * 50.0
        data[5, 5] = -100.0
        data[25, 25] = 999.0
        raster = from_numpy(data, nodata=None, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(16, 16)

        def tile_fn(r: OwnedRasterArray) -> tuple[float, float]:
            d = r.to_numpy()
            return (float(d.min()), float(d.max()))

        def merge_fn(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
            return (min(a[0], b[0]), max(a[1], b[1]))

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result[0] == pytest.approx(-100.0)
        assert result[1] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — histogram accumulator
# ---------------------------------------------------------------------------


class TestAccumulatorHistogram:
    def test_tiled_histogram_matches_whole(self):
        """tile_fn returns (counts, bin_edges) via np.histogram.
        merge_fn sums counts. Tiled result matches non-tiled."""
        raster = _make_raster(64, 64, dtype=np.float32)
        bins = np.linspace(0.0, 1.0, 11)

        def tile_fn(r: OwnedRasterArray) -> tuple[np.ndarray, np.ndarray]:
            counts, edges = np.histogram(r.to_numpy(), bins=bins)
            return (counts, edges)

        def merge_fn(
            a: tuple[np.ndarray, np.ndarray],
            b: tuple[np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a[0] + b[0], a[1])

        plan = _make_tiled_plan(32, 32)
        result_tiled = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        result_whole = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, _WHOLE_PLAN)

        np.testing.assert_array_equal(result_tiled[0], result_whole[0])
        np.testing.assert_array_equal(result_tiled[1], result_whole[1])

    def test_histogram_total_count(self):
        """Total counts across all bins equal the total pixel count."""
        raster = _make_raster(64, 64, dtype=np.float32)
        bins = np.linspace(0.0, 1.0, 11)

        def tile_fn(r: OwnedRasterArray) -> tuple[np.ndarray, np.ndarray]:
            return np.histogram(r.to_numpy(), bins=bins)

        def merge_fn(
            a: tuple[np.ndarray, np.ndarray],
            b: tuple[np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            return (a[0] + b[0], a[1])

        plan = _make_tiled_plan(16, 16)
        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result[0].sum() == 64 * 64


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — dict accumulator (zonal-style)
# ---------------------------------------------------------------------------


class TestAccumulatorDict:
    def test_tiled_dict_matches_whole(self):
        """tile_fn returns {zone: count}, merge_fn merges dicts (summing counts)."""
        # Create a raster with 4 zones (quadrants 0-3)
        data = np.zeros((32, 32), dtype=np.int32)
        data[:16, :16] = 0
        data[:16, 16:] = 1
        data[16:, :16] = 2
        data[16:, 16:] = 3
        raster = from_numpy(data, nodata=None, affine=_TEST_AFFINE)

        def tile_fn(r: OwnedRasterArray) -> dict[int, int]:
            d = r.to_numpy()
            unique, counts = np.unique(d, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}

        def merge_fn(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
            merged = dict(a)
            for k, v in b.items():
                merged[k] = merged.get(k, 0) + v
            return merged

        plan = _make_tiled_plan(16, 16)
        result_tiled = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        result_whole = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, _WHOLE_PLAN)

        assert result_tiled == result_whole
        # Each quadrant is 16x16 = 256 pixels
        assert result_tiled == {0: 256, 1: 256, 2: 256, 3: 256}

    def test_dict_with_overlapping_zones(self):
        """Zones that span multiple tiles are correctly merged."""
        # Single zone value across the entire raster
        data = np.full((32, 32), 7, dtype=np.int32)
        raster = from_numpy(data, nodata=None, affine=_TEST_AFFINE)

        def tile_fn(r: OwnedRasterArray) -> dict[int, int]:
            d = r.to_numpy()
            unique, counts = np.unique(d, return_counts=True)
            return {int(u): int(c) for u, c in zip(unique, counts)}

        def merge_fn(a: dict[int, int], b: dict[int, int]) -> dict[int, int]:
            merged = dict(a)
            for k, v in b.items():
                merged[k] = merged.get(k, 0) + v
            return merged

        plan = _make_tiled_plan(16, 16)
        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result == {7: 32 * 32}


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — edge tiles
# ---------------------------------------------------------------------------


class TestAccumulatorEdgeTiles:
    def test_non_divisible_raster_sum(self):
        """Non-divisible raster dimensions still produce correct sum."""
        raster = _make_raster(50, 35, dtype=np.float32, fill=1.0)
        plan = _make_tiled_plan(32, 32)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result == pytest.approx(50 * 35)

    def test_non_divisible_tile_count(self):
        """50x35 with 32x32 tiles: ceil(50/32)=2, ceil(35/32)=2 -> 4 tiles."""
        raster = _make_raster(50, 35)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def tile_fn(r: OwnedRasterArray) -> int:
            nonlocal call_count
            call_count += 1
            return r.pixel_count

        def merge_fn(a: int, b: int) -> int:
            return a + b

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert call_count == 4
        assert result == 50 * 35

    def test_asymmetric_tiles_sum(self):
        """Non-square raster with non-square tiles."""
        raster = _make_raster(100, 60, dtype=np.float64, fill=0.5)
        plan = _make_tiled_plan(40, 25)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result == pytest.approx(0.5 * 100 * 60)


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — multiband
# ---------------------------------------------------------------------------


class TestAccumulatorMultiband:
    def test_3band_sum(self):
        """3-band raster: tile_fn receives tiles with all bands."""
        raster = _make_raster(64, 64, bands=3, dtype=np.float32, fill=1.0)
        plan = _make_tiled_plan(32, 32)

        def tile_fn(r: OwnedRasterArray) -> float:
            assert r.band_count == 3
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        # 3 bands * 64 * 64 pixels * 1.0
        assert result == pytest.approx(3 * 64 * 64)

    def test_3band_per_band_stats(self):
        """Accumulate per-band sums via dict accumulator."""
        data = np.zeros((3, 32, 32), dtype=np.float32)
        data[0, :, :] = 1.0
        data[1, :, :] = 2.0
        data[2, :, :] = 3.0
        raster = from_numpy(data, nodata=None, affine=_TEST_AFFINE)
        plan = _make_tiled_plan(16, 16)

        def tile_fn(r: OwnedRasterArray) -> dict[int, float]:
            d = r.to_numpy()
            return {b: float(d[b].sum()) for b in range(d.shape[0])}

        def merge_fn(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
            merged = dict(a)
            for k, v in b.items():
                merged[k] = merged.get(k, 0.0) + v
            return merged

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result[0] == pytest.approx(1.0 * 32 * 32)
        assert result[1] == pytest.approx(2.0 * 32 * 32)
        assert result[2] == pytest.approx(3.0 * 32 * 32)


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — single tile (raster smaller than tile)
# ---------------------------------------------------------------------------


class TestAccumulatorSingleTile:
    def test_raster_smaller_than_tile(self):
        """Raster smaller than tile -> one tile, no merge."""
        raster = _make_raster(10, 15, dtype=np.float32, fill=5.0)
        plan = _make_tiled_plan(32, 32)
        call_count = 0

        def tile_fn(r: OwnedRasterArray) -> float:
            nonlocal call_count
            call_count += 1
            assert r.height == 10
            assert r.width == 15
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            raise AssertionError("merge_fn should not be called with a single tile")

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert call_count == 1
        assert result == pytest.approx(5.0 * 10 * 15)

    def test_single_pixel_raster(self):
        """1x1 raster: one tile, no merge."""
        raster = _make_raster(1, 1, dtype=np.float32, fill=42.0)
        plan = _make_tiled_plan(32, 32)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            raise AssertionError("merge_fn should not be called with a single tile")

        result = dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)
        assert result == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# dispatch_tiled_accumulator — DEVICE-resident guard
# ---------------------------------------------------------------------------


class TestAccumulatorDeviceGuard:
    def test_device_resident_raises(self):
        """DEVICE-resident input on TILED path raises ValueError."""
        from vibespatial.residency import Residency

        raster = _make_raster(32, 32)
        raster.residency = Residency.DEVICE  # simulate device-resident
        plan = _make_tiled_plan(16, 16)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        with pytest.raises(ValueError, match="HOST-resident"):
            dispatch_tiled_accumulator(raster, tile_fn, merge_fn, plan)

    def test_tiled_plan_without_tile_shape_raises(self):
        """TILED strategy with tile_shape=None is a malformed plan."""
        bad_plan = RasterPlan(
            strategy=TilingStrategy.TILED,
            tile_shape=None,
            halo=0,
            n_tiles=0,
            estimated_vram_per_tile=0,
        )
        raster = _make_raster(32, 32)

        def tile_fn(r: OwnedRasterArray) -> float:
            return float(r.to_numpy().sum())

        def merge_fn(a: float, b: float) -> float:
            return a + b

        with pytest.raises(ValueError, match="tile_shape is None"):
            dispatch_tiled_accumulator(raster, tile_fn, merge_fn, bad_plan)

    def test_accumulator_importable_via_init(self):
        """dispatch_tiled_accumulator is importable via __init__.py."""
        from vibespatial.raster import dispatch_tiled_accumulator as dta

        assert callable(dta)
