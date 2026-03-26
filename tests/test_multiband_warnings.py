"""Tests for multiband handling in algebra operations.

For raster_expression: verifies that UserWarning is still emitted when
multiband rasters are silently squeezed to band 0.

For focal/terrain/convolve: verifies that per-band dispatch produces
correct multiband results (vibeSpatial-2to.3.1).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vibespatial.raster.buffers import from_numpy

try:
    import cupy  # noqa: F401

    HAS_GPU = True
except ImportError:
    HAS_GPU = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="CuPy not available")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_AFFINE = (1.0, 0.0, 0.0, 0.0, -1.0, 4.0)


@pytest.fixture
def multiband_raster():
    """3-band float64 raster (3, 4, 4)."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 4, 4), dtype=np.float64)
    return from_numpy(data, affine=_AFFINE)


@pytest.fixture
def multiband_raster_nodata():
    """3-band float64 raster with nodata."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 4, 4), dtype=np.float64)
    data[0, 0, 0] = -9999.0
    return from_numpy(data, nodata=-9999.0, affine=_AFFINE)


@pytest.fixture
def multiband_binary_raster():
    """3-band uint8 binary raster (3, 4, 4) for label/morphology ops."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 2, size=(3, 4, 4), dtype=np.uint8)
    return from_numpy(data, affine=_AFFINE)


@pytest.fixture
def multiband_dem():
    """3-band float64 DEM raster for terrain ops."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 8, 8), dtype=np.float64) * 100
    return from_numpy(data, affine=_AFFINE)


# ---------------------------------------------------------------------------
# Algebra: raster_expression (squeeze+warn remains for expressions)
# ---------------------------------------------------------------------------


@requires_gpu
def test_expression_multiband_warns_gpu(multiband_raster):
    """raster_expression GPU path warns on multiband input."""
    from vibespatial.raster.algebra import raster_expression

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_expression("a + 1.0", a=multiband_raster, use_gpu=True)


def test_expression_multiband_warns_cpu(multiband_raster):
    """raster_expression CPU path warns on multiband input."""
    from vibespatial.raster.algebra import raster_expression

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_expression("a + 1.0", a=multiband_raster, use_gpu=False)


# ---------------------------------------------------------------------------
# Algebra: raster_convolve per-band dispatch
# ---------------------------------------------------------------------------


@requires_gpu
def test_convolve_multiband_3band():
    """raster_convolve processes all 3 bands via per-band GPU dispatch."""
    from vibespatial.raster.algebra import raster_convolve

    rng = np.random.default_rng(42)
    data = rng.random((3, 8, 8), dtype=np.float64)
    raster = from_numpy(data, affine=_AFFINE)

    kernel = np.ones((3, 3), dtype=np.float64) / 9.0

    # Verify multiband dispatch produces correct per-band results
    result = raster_convolve(raster, kernel)
    out = result.to_numpy()
    assert out.shape == (3, 8, 8), f"Expected (3, 8, 8), got {out.shape}"

    # Each band should match single-band convolution
    for i in range(3):
        band_raster = from_numpy(data[i], affine=_AFFINE)
        band_result = raster_convolve(band_raster, kernel)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband convolve",
        )

    # Metadata preservation
    assert result.affine == raster.affine
    assert result.crs == raster.crs


# ---------------------------------------------------------------------------
# Algebra: slope per-band dispatch
# ---------------------------------------------------------------------------


def test_slope_multiband():
    """raster_slope processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_slope

    rng = np.random.default_rng(42)
    data = rng.random((3, 8, 8), dtype=np.float64) * 100
    raster = from_numpy(data, affine=_AFFINE)

    result = raster_slope(raster, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 8, 8), f"Expected (3, 8, 8), got {out.shape}"

    # Verify each band matches the single-band result
    for i in range(3):
        band_raster = from_numpy(data[i], affine=_AFFINE)
        band_result = raster_slope(band_raster, use_gpu=False)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband slope",
        )

    # Metadata preservation
    assert result.affine == raster.affine
    assert result.crs == raster.crs


# ---------------------------------------------------------------------------
# Algebra: hillshade per-band dispatch
# ---------------------------------------------------------------------------


def test_hillshade_multiband_cpu(multiband_dem):
    """raster_hillshade processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_hillshade

    result = raster_hillshade(multiband_dem, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 8, 8), f"Expected (3, 8, 8), got {out.shape}"

    # Verify each band matches single-band result
    raw = multiband_dem.to_numpy()
    for i in range(3):
        band_raster = from_numpy(raw[i], affine=_AFFINE)
        band_result = raster_hillshade(band_raster, use_gpu=False)
        np.testing.assert_array_equal(
            out[i],
            band_result.to_numpy(),
            err_msg=f"Band {i} mismatch in multiband hillshade",
        )

    assert result.affine == multiband_dem.affine


# ---------------------------------------------------------------------------
# Algebra: TRI / TPI / curvature per-band dispatch
# ---------------------------------------------------------------------------


def test_tri_multiband_cpu(multiband_dem):
    """raster_tri processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_tri

    result = raster_tri(multiband_dem, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 8, 8), f"Expected (3, 8, 8), got {out.shape}"

    raw = multiband_dem.to_numpy()
    for i in range(3):
        band_raster = from_numpy(raw[i], affine=_AFFINE)
        band_result = raster_tri(band_raster, use_gpu=False)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband TRI",
        )


def test_tpi_multiband_cpu(multiband_dem):
    """raster_tpi processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_tpi

    result = raster_tpi(multiband_dem, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 8, 8), f"Expected (3, 8, 8), got {out.shape}"

    raw = multiband_dem.to_numpy()
    for i in range(3):
        band_raster = from_numpy(raw[i], affine=_AFFINE)
        band_result = raster_tpi(band_raster, use_gpu=False)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband TPI",
        )


def test_curvature_multiband_cpu(multiband_dem):
    """raster_curvature processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_curvature

    result = raster_curvature(multiband_dem, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 8, 8), f"Expected (3, 8, 8), got {out.shape}"

    raw = multiband_dem.to_numpy()
    for i in range(3):
        band_raster = from_numpy(raw[i], affine=_AFFINE)
        band_result = raster_curvature(band_raster, use_gpu=False)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband curvature",
        )


# ---------------------------------------------------------------------------
# Algebra: focal statistics per-band dispatch
# ---------------------------------------------------------------------------


def test_focal_min_multiband():
    """raster_focal_min processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_focal_min

    rng = np.random.default_rng(42)
    data = rng.random((3, 6, 6), dtype=np.float64)
    raster = from_numpy(data, affine=_AFFINE)

    result = raster_focal_min(raster, radius=1, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 6, 6), f"Expected (3, 6, 6), got {out.shape}"

    # Verify each band matches single-band result
    for i in range(3):
        band_raster = from_numpy(data[i], affine=_AFFINE)
        band_result = raster_focal_min(band_raster, radius=1, use_gpu=False)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband focal_min",
        )

    assert result.affine == raster.affine
    assert result.crs == raster.crs


def test_focal_mean_multiband_cpu():
    """raster_focal_mean processes all 3 bands via per-band CPU dispatch."""
    from vibespatial.raster.algebra import raster_focal_mean

    rng = np.random.default_rng(42)
    data = rng.random((3, 6, 6), dtype=np.float64)
    raster = from_numpy(data, affine=_AFFINE)

    result = raster_focal_mean(raster, radius=1, use_gpu=False)
    out = result.to_numpy()
    assert out.shape == (3, 6, 6), f"Expected (3, 6, 6), got {out.shape}"

    for i in range(3):
        band_raster = from_numpy(data[i], affine=_AFFINE)
        band_result = raster_focal_mean(band_raster, radius=1, use_gpu=False)
        np.testing.assert_allclose(
            out[i],
            band_result.to_numpy(),
            atol=1e-12,
            err_msg=f"Band {i} mismatch in multiband focal_mean",
        )


# ---------------------------------------------------------------------------
# Label: morphology CPU (unchanged -- still uses squeeze+warn)
# ---------------------------------------------------------------------------


def test_morphology_cpu_multiband_warns(multiband_binary_raster):
    """CPU morphology path warns on multiband input."""
    from vibespatial.raster.label import raster_morphology

    with pytest.warns(UserWarning, match=r"Multiband raster with 3 bands"):
        raster_morphology(multiband_binary_raster, "erode", use_gpu=False)


# ---------------------------------------------------------------------------
# Verify warning message content (expression path only)
# ---------------------------------------------------------------------------


def test_warning_message_includes_band_count():
    """Warning message should include the actual number of bands."""
    rng = np.random.default_rng(42)
    data = rng.random((5, 4, 4), dtype=np.float64)
    five_band = from_numpy(data, affine=_AFFINE)

    from vibespatial.raster.algebra import raster_expression

    with pytest.warns(UserWarning, match=r"Multiband raster with 5 bands"):
        raster_expression("a + 1.0", a=five_band, use_gpu=False)


def test_singleband_no_warning():
    """Single-band rasters should not emit any multiband warning."""
    data = np.ones((4, 4), dtype=np.float64)
    single_band = from_numpy(data, affine=_AFFINE)

    from vibespatial.raster.algebra import raster_expression

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Should NOT raise -- no multiband warning expected
        raster_expression("a + 1.0", a=single_band, use_gpu=False)
