"""Tests for connected component labeling, sieve filtering, and morphology."""

from __future__ import annotations

import numpy as np
import pytest

try:
    from scipy.ndimage import label as scipy_label  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cupy  # noqa: F401

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from vibespatial.raster.buffers import from_numpy
from vibespatial.raster.label import (
    label_connected_components,
    raster_morphology,
    sieve_filter,
)

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")


def _cpu_morph_reference(data, operation, connectivity, nodata=None, iterations=1):
    """Build CPU reference result for GPU comparison."""
    raster = from_numpy(data, nodata=nodata)
    return raster_morphology(
        raster, operation, connectivity=connectivity, iterations=iterations, use_gpu=False
    ).to_numpy()


# ---------------------------------------------------------------------------
# CPU tests: connected component labeling
# ---------------------------------------------------------------------------


class TestLabelConnectedComponents:
    def test_two_blobs_4conn(self):
        data = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 0, 0, 0, 2],
                [0, 0, 3, 3, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        result = label_connected_components(raster, connectivity=4)
        labeled = result.to_numpy()
        # Two separate blobs of 1s and one of 2s, one of 3s
        assert labeled[0, 0] == labeled[1, 0]  # same component
        assert labeled[0, 3] == labeled[0, 4] == labeled[1, 4]  # same component
        assert labeled[2, 2] == labeled[2, 3]  # same component
        # Different components have different labels
        assert labeled[0, 0] != labeled[0, 3]
        assert labeled[0, 0] != labeled[2, 2]

    def test_single_component_8conn(self):
        data = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        result_4 = label_connected_components(raster, connectivity=4)
        result_8 = label_connected_components(raster, connectivity=8)
        # 4-connectivity: 3 separate components (diagonal not connected)
        labels_4 = np.unique(result_4.to_numpy())
        labels_4 = labels_4[labels_4 != 0]
        assert len(labels_4) == 3
        # 8-connectivity: 1 component (diagonal connected)
        labels_8 = np.unique(result_8.to_numpy())
        labels_8 = labels_8[labels_8 != 0]
        assert len(labels_8) == 1

    def test_nodata_excluded(self):
        data = np.array([[1, -9999, 1]], dtype=np.int32)
        raster = from_numpy(data, nodata=-9999)
        result = label_connected_components(raster, connectivity=4)
        labeled = result.to_numpy()
        assert labeled[0, 1] == 0  # nodata -> background
        assert labeled[0, 0] != labeled[0, 2]  # separated by nodata

    def test_all_background(self):
        data = np.zeros((3, 3), dtype=np.int32)
        raster = from_numpy(data)
        result = label_connected_components(raster)
        assert not result.to_numpy().any()

    def test_bad_connectivity(self):
        raster = from_numpy(np.ones((2, 2), dtype=np.int32))
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            label_connected_components(raster, connectivity=6)


# ---------------------------------------------------------------------------
# CPU tests: sieve filter
# ---------------------------------------------------------------------------


class TestSieveFilter:
    def test_removes_small(self):
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4)
        sieved = sieve_filter(labeled, min_size=3)
        sieved_data = sieved.to_numpy()
        # Component with 4 pixels (label of the 1s) survives
        # Component with 1 pixel (label of the 2) is removed
        assert sieved_data[0, 0] != 0  # large component survives
        assert sieved_data[0, 3] == 0  # small component removed

    def test_min_size_1_keeps_all(self):
        data = np.array([[1, 0, 2]], dtype=np.int32)
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4)
        sieved = sieve_filter(labeled, min_size=1)
        assert (sieved.to_numpy() > 0).sum() == 2  # both kept


# ---------------------------------------------------------------------------
# CPU tests: morphology
# ---------------------------------------------------------------------------


class TestRasterMorphology:
    def test_dilate(self):
        data = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_morphology(raster, "dilate", connectivity=4)
        result_data = result.to_numpy()
        # Center + 4 neighbors should be nonzero
        assert result_data[1, 1] != 0
        assert result_data[0, 1] != 0  # top
        assert result_data[2, 1] != 0  # bottom
        assert result_data[1, 0] != 0  # left
        assert result_data[1, 2] != 0  # right

    def test_erode(self):
        data = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_morphology(raster, "erode", connectivity=4)
        result_data = result.to_numpy()
        # Only center survives erosion with 4-conn on 3x3
        assert result_data[1, 1] != 0

    def test_bad_operation(self):
        raster = from_numpy(np.ones((3, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="operation must be"):
            raster_morphology(raster, "invalid")


# ---------------------------------------------------------------------------
# GPU tests: connected component labeling
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
class TestLabelGPU:
    """GPU CCL tests -- skipped when CuPy is not available."""

    def test_two_blobs_4conn_gpu(self):
        """GPU CCL matches expected component structure (4-connected)."""
        from vibespatial.raster.label import label_gpu

        data = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 0, 0, 0, 2],
                [0, 0, 3, 3, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        result = label_gpu(raster, connectivity=4)
        labeled = result.to_numpy()

        # Same component structure as CPU
        assert labeled[0, 0] == labeled[1, 0]  # same component
        assert labeled[0, 3] == labeled[0, 4] == labeled[1, 4]  # same component
        assert labeled[2, 2] == labeled[2, 3]  # same component
        # Different components
        assert labeled[0, 0] != labeled[0, 3]
        assert labeled[0, 0] != labeled[2, 2]
        # Background is 0
        assert labeled[1, 1] == 0
        assert labeled[0, 2] == 0

    def test_gpu_matches_cpu_4conn(self):
        """GPU and CPU produce equivalent label counts (4-connected)."""
        from vibespatial.raster.label import label_gpu

        data = np.array(
            [
                [1, 1, 0, 2, 2],
                [1, 0, 0, 0, 2],
                [0, 0, 3, 3, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)

        cpu_result = label_connected_components(raster, connectivity=4, use_gpu=False)
        gpu_result = label_gpu(raster, connectivity=4)

        cpu_labels = cpu_result.to_numpy()
        gpu_labels = gpu_result.to_numpy()

        # Both should have same number of unique labels
        cpu_unique = set(np.unique(cpu_labels)) - {0}
        gpu_unique = set(np.unique(gpu_labels)) - {0}
        assert len(cpu_unique) == len(gpu_unique)

        # Both should have same foreground/background mask
        np.testing.assert_array_equal(cpu_labels > 0, gpu_labels > 0)

        # Same connectivity structure: pixels in same component on CPU should
        # be in same component on GPU
        for lbl in cpu_unique:
            cpu_component = cpu_labels == lbl
            gpu_vals_in_component = np.unique(gpu_labels[cpu_component])
            gpu_vals_in_component = gpu_vals_in_component[gpu_vals_in_component != 0]
            assert len(gpu_vals_in_component) == 1, (
                f"CPU component {lbl} maps to multiple GPU labels: {gpu_vals_in_component}"
            )

    def test_gpu_matches_cpu_8conn(self):
        """GPU and CPU produce equivalent results (8-connected)."""
        from vibespatial.raster.label import label_gpu

        data = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)

        cpu_result = label_connected_components(raster, connectivity=8, use_gpu=False)
        gpu_result = label_gpu(raster, connectivity=8)

        cpu_labels = cpu_result.to_numpy()
        gpu_labels = gpu_result.to_numpy()

        cpu_n = len(set(np.unique(cpu_labels)) - {0})
        gpu_n = len(set(np.unique(gpu_labels)) - {0})
        assert cpu_n == gpu_n == 1

    def test_gpu_4conn_vs_8conn(self):
        """4-connectivity gives more components than 8-connectivity on diagonal."""
        from vibespatial.raster.label import label_gpu

        data = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)

        result_4 = label_gpu(raster, connectivity=4)
        result_8 = label_gpu(raster, connectivity=8)

        n_4 = len(set(np.unique(result_4.to_numpy())) - {0})
        n_8 = len(set(np.unique(result_8.to_numpy())) - {0})

        assert n_4 == 3  # diagonal pixels are separate under 4-conn
        assert n_8 == 1  # diagonal pixels are connected under 8-conn

    def test_gpu_all_background(self):
        """GPU handles all-background raster."""
        from vibespatial.raster.label import label_gpu

        data = np.zeros((4, 4), dtype=np.int32)
        raster = from_numpy(data)
        result = label_gpu(raster, connectivity=4)
        assert not result.to_numpy().any()

    def test_gpu_single_pixel(self):
        """GPU labels a single foreground pixel correctly."""
        from vibespatial.raster.label import label_gpu

        data = np.zeros((3, 3), dtype=np.int32)
        data[1, 1] = 5
        raster = from_numpy(data)
        result = label_gpu(raster, connectivity=4)
        labeled = result.to_numpy()
        assert labeled[1, 1] == 1
        assert labeled[0, 0] == 0

    def test_gpu_nodata_excluded(self):
        """GPU excludes nodata pixels from labeling."""
        from vibespatial.raster.label import label_gpu

        data = np.array([[1, -9999, 1]], dtype=np.int32)
        raster = from_numpy(data, nodata=-9999)
        result = label_gpu(raster, connectivity=4)
        labeled = result.to_numpy()
        assert labeled[0, 1] == 0  # nodata -> background
        assert labeled[0, 0] != labeled[0, 2]  # separated by nodata

    def test_gpu_checkerboard(self):
        """GPU CCL on a checkerboard pattern (many small components)."""
        from vibespatial.raster.label import label_gpu

        data = np.zeros((8, 8), dtype=np.int32)
        data[0::2, 0::2] = 1
        data[1::2, 1::2] = 1
        raster = from_numpy(data)

        # 4-connected: each pixel is isolated (no 4-neighbors)
        result_4 = label_gpu(raster, connectivity=4)
        labels_4 = result_4.to_numpy()
        n_4 = len(set(np.unique(labels_4)) - {0})
        assert n_4 == 32  # each of the 32 foreground pixels is its own component

        # 8-connected: all foreground connected via diagonal
        result_8 = label_gpu(raster, connectivity=8)
        labels_8 = result_8.to_numpy()
        n_8 = len(set(np.unique(labels_8)) - {0})
        assert n_8 == 1  # all connected diagonally

    def test_gpu_large_connected_region(self):
        """GPU CCL on a fully connected foreground block."""
        from vibespatial.raster.label import label_gpu

        data = np.ones((20, 20), dtype=np.int32)
        raster = from_numpy(data)
        result = label_gpu(raster, connectivity=4)
        labeled = result.to_numpy()
        unique = set(np.unique(labeled)) - {0}
        assert len(unique) == 1  # single component
        assert labeled.min() == 1  # all pixels labeled 1

    def test_gpu_spiral(self):
        """GPU CCL on a spiral-like connected path."""
        from vibespatial.raster.label import label_gpu

        data = np.zeros((7, 7), dtype=np.int32)
        # Draw a spiral path
        data[0, 0:6] = 1
        data[0:6, 6] = 1  # right edge (rows 0-5)  -- note: data[0,6] already set
        data[6, 2:7] = 1  # bottom partial
        data[2:7, 2] = 1  # inner left
        raster = from_numpy(data)

        result = label_gpu(raster, connectivity=4)
        labeled = result.to_numpy()

        # All foreground pixels should be in ONE component
        fg_labels = set(np.unique(labeled[data > 0]))
        fg_labels.discard(0)
        assert len(fg_labels) == 1

    def test_gpu_diagnostics(self):
        """GPU CCL records diagnostic events."""
        from vibespatial.raster.label import label_gpu

        data = np.array([[1, 1], [1, 0]], dtype=np.int32)
        raster = from_numpy(data)
        result = label_gpu(raster, connectivity=4)
        runtime_events = [e for e in result.diagnostics if e.kind == "runtime"]
        assert len(runtime_events) >= 1
        assert "label_gpu" in runtime_events[-1].detail


# ---------------------------------------------------------------------------
# GPU tests: morphology
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@requires_gpu
class TestMorphologyGPU:
    """GPU morphology tests -- compare GPU kernels against CPU (scipy) baseline."""

    # -- erode --

    def test_erode_4conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.ones((5, 5), dtype=np.uint8)
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "erode", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "erode", 4)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_erode_8conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.ones((5, 5), dtype=np.uint8)
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "erode", connectivity=8).to_numpy()
        cpu_result = _cpu_morph_reference(data, "erode", 8)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_erode_cross_pattern(self):
        """Eroding a cross pattern with 4-conn should leave only center."""
        from vibespatial.raster.label import morphology_gpu

        data = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "erode", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "erode", 4)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    # -- dilate --

    def test_dilate_4conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((5, 5), dtype=np.uint8)
        data[2, 2] = 1
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "dilate", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "dilate", 4)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_dilate_8conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((5, 5), dtype=np.uint8)
        data[2, 2] = 1
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "dilate", connectivity=8).to_numpy()
        cpu_result = _cpu_morph_reference(data, "dilate", 8)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_dilate_diamond_pattern(self):
        """Dilate a diamond pattern and compare with CPU."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((7, 7), dtype=np.uint8)
        data[3, 3] = 1
        data[2, 3] = 1
        data[4, 3] = 1
        data[3, 2] = 1
        data[3, 4] = 1
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "dilate", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "dilate", 4)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    # -- open --

    def test_open_4conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((10, 10), dtype=np.uint8)
        data[2:8, 2:8] = 1
        data[4, 8] = 1  # small protrusion
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "open", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "open", 4)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_open_8conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((10, 10), dtype=np.uint8)
        data[2:8, 2:8] = 1
        data[4, 8] = 1
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "open", connectivity=8).to_numpy()
        cpu_result = _cpu_morph_reference(data, "open", 8)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    # -- close --

    def test_close_4conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((10, 10), dtype=np.uint8)
        data[2:8, 2:8] = 1
        data[4, 4] = 0  # hole that closing should fill
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "close", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "close", 4)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_close_8conn_matches_cpu(self):
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((10, 10), dtype=np.uint8)
        data[2:8, 2:8] = 1
        data[4, 4] = 0
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "close", connectivity=8).to_numpy()
        cpu_result = _cpu_morph_reference(data, "close", 8)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    # -- nodata handling --

    def test_nodata_excluded_from_foreground(self):
        """Nodata pixels should be treated as background."""
        from vibespatial.raster.label import morphology_gpu

        data = np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, -9999, 1, 1],
                [1, -9999, 1, -9999, 1],
                [1, 1, -9999, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data, nodata=-9999)
        gpu_result = morphology_gpu(raster, "erode", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "erode", 4, nodata=-9999)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_nodata_dilate(self):
        """Dilate with nodata -- nodata pixels are background."""
        from vibespatial.raster.label import morphology_gpu

        data = np.array(
            [
                [0, 0, 0],
                [0, 255, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
        raster = from_numpy(data, nodata=255)
        gpu_result = morphology_gpu(raster, "dilate", connectivity=4).to_numpy()
        cpu_result = _cpu_morph_reference(data, "dilate", 4, nodata=255)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    # -- shape variations --

    def test_square_block(self):
        """Large square block -- erode should shrink edges."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((20, 20), dtype=np.uint8)
        data[5:15, 5:15] = 1
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "erode", connectivity=8).to_numpy()
        cpu_result = _cpu_morph_reference(data, "erode", 8)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_non_square_raster(self):
        """Non-square (wide) raster."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((5, 30), dtype=np.uint8)
        data[1:4, 5:25] = 1
        raster = from_numpy(data)
        for op in ("erode", "dilate", "open", "close"):
            gpu_result = morphology_gpu(raster, op, connectivity=4).to_numpy()
            cpu_result = _cpu_morph_reference(data, op, 4)
            np.testing.assert_array_equal(
                gpu_result,
                cpu_result,
                err_msg=f"Mismatch for operation={op}",
            )

    def test_tall_raster(self):
        """Non-square (tall) raster."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((30, 5), dtype=np.uint8)
        data[5:25, 1:4] = 1
        raster = from_numpy(data)
        for op in ("erode", "dilate"):
            gpu_result = morphology_gpu(raster, op, connectivity=8).to_numpy()
            cpu_result = _cpu_morph_reference(data, op, 8)
            np.testing.assert_array_equal(
                gpu_result,
                cpu_result,
                err_msg=f"Mismatch for operation={op}",
            )

    def test_iterations_2(self):
        """Multiple iterations should match CPU."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((15, 15), dtype=np.uint8)
        data[3:12, 3:12] = 1
        raster = from_numpy(data)
        for op in ("erode", "dilate", "open", "close"):
            gpu_result = morphology_gpu(raster, op, connectivity=4, iterations=2).to_numpy()
            cpu_result = _cpu_morph_reference(data, op, 4, iterations=2)
            np.testing.assert_array_equal(
                gpu_result,
                cpu_result,
                err_msg=f"Mismatch for operation={op} iterations=2",
            )

    def test_all_zeros(self):
        """All-zero raster should stay all-zero for any operation."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((8, 8), dtype=np.uint8)
        raster = from_numpy(data)
        for op in ("erode", "dilate", "open", "close"):
            gpu_result = morphology_gpu(raster, op, connectivity=4).to_numpy()
            np.testing.assert_array_equal(gpu_result, data)

    def test_all_ones(self):
        """All-one raster for dilate should stay all-one."""
        from vibespatial.raster.label import morphology_gpu

        data = np.ones((8, 8), dtype=np.uint8)
        raster = from_numpy(data)
        gpu_result = morphology_gpu(raster, "dilate", connectivity=8).to_numpy()
        cpu_result = _cpu_morph_reference(data, "dilate", 8)
        np.testing.assert_array_equal(gpu_result, cpu_result)

    def test_gpu_morphology_diagnostics(self):
        """GPU path should record diagnostic events."""
        from vibespatial.raster.label import morphology_gpu

        data = np.zeros((10, 10), dtype=np.uint8)
        data[3:7, 3:7] = 1
        raster = from_numpy(data)
        result = morphology_gpu(raster, "erode", connectivity=4)
        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "morphology_gpu" in e.detail
        ]
        assert len(runtime_events) == 1
        assert runtime_events[0].elapsed_seconds > 0

    def test_gpu_morphology_no_dh_d_ping_pong(self):
        """morphology_gpu must not D->H->D ping-pong on device-resident input.

        Regression test for BUG_SWEEP #15: the legacy path used
        raster.to_numpy() + cp.asarray() instead of the device residency API,
        causing an unnecessary device->host->device transfer.
        """
        from vibespatial.raster.label import morphology_gpu
        from vibespatial.residency import Residency

        data = np.zeros((10, 10), dtype=np.uint8)
        data[3:7, 3:7] = 1
        # Create raster already on device
        raster = from_numpy(data, residency=Residency.DEVICE)

        # Snapshot diagnostics length after initial H->D transfer
        diag_baseline = len(raster.diagnostics)

        result = morphology_gpu(raster, "erode", connectivity=4)

        # The input raster must NOT have gained any device->host transfer
        # events during the morphology_gpu call.
        new_events = raster.diagnostics[diag_baseline:]
        d2h_events = [e for e in new_events if e.kind == "transfer" and "device->host" in e.detail]
        assert len(d2h_events) == 0, (
            f"morphology_gpu triggered {len(d2h_events)} device->host "
            f"transfer(s) on a device-resident input: {d2h_events}"
        )

        # Result should still be correct
        expected = _cpu_morph_reference(data, "erode", 4)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_gpu_morphology_no_ping_pong_with_nodata(self):
        """morphology_gpu avoids D->H->D even with nodata pixels present."""
        from vibespatial.raster.label import morphology_gpu
        from vibespatial.residency import Residency

        data = np.ones((10, 10), dtype=np.float32)
        data[3:7, 3:7] = 0.0
        data[5, 5] = -9999.0  # nodata pixel
        raster = from_numpy(data, nodata=-9999.0, residency=Residency.DEVICE)

        diag_baseline = len(raster.diagnostics)

        result = morphology_gpu(raster, "dilate", connectivity=4)

        new_events = raster.diagnostics[diag_baseline:]
        d2h_events = [e for e in new_events if e.kind == "transfer" and "device->host" in e.detail]
        assert len(d2h_events) == 0, (
            f"morphology_gpu triggered {len(d2h_events)} device->host "
            f"transfer(s) with nodata input: {d2h_events}"
        )

        # Verify result has correct runtime diagnostic
        runtime_events = [
            e for e in result.diagnostics if e.kind == "runtime" and "morphology_gpu" in e.detail
        ]
        assert len(runtime_events) == 1


# ---------------------------------------------------------------------------
# GPU tests: sieve filter
# ---------------------------------------------------------------------------


@requires_gpu
class TestSieveFilterGPU:
    """GPU sieve filter tests -- skipped when CuPy is not available."""

    def test_gpu_removes_small_components(self):
        """GPU sieve removes components below min_size."""
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        sieved = sieve_filter(labeled, min_size=3, use_gpu=True)
        sieved_data = sieved.to_numpy()
        # Component with 4 pixels survives
        assert sieved_data[0, 0] != 0
        # Component with 1 pixel (the "2") is removed
        assert sieved_data[0, 3] == 0

    def test_gpu_matches_cpu(self):
        """GPU sieve produces same result as CPU sieve."""
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 3],
                [0, 0, 4, 4, 4],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        cpu_sieved = sieve_filter(labeled, min_size=3, use_gpu=False)
        gpu_sieved = sieve_filter(labeled, min_size=3, use_gpu=True)
        np.testing.assert_array_equal(cpu_sieved.to_numpy(), gpu_sieved.to_numpy())

    def test_gpu_min_size_1_keeps_all(self):
        """GPU sieve with min_size=1 keeps all components."""
        data = np.array([[1, 0, 2]], dtype=np.int32)
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        sieved = sieve_filter(labeled, min_size=1, use_gpu=True)
        assert (sieved.to_numpy() > 0).sum() == 2

    def test_gpu_all_removed(self):
        """GPU sieve with large min_size removes all components."""
        data = np.array(
            [
                [1, 0, 2],
                [0, 3, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        sieved = sieve_filter(labeled, min_size=100, use_gpu=True)
        # All components are size 1, should all be removed
        assert not sieved.to_numpy().any()

    def test_gpu_preserves_background(self):
        """GPU sieve does not modify background (0) pixels."""
        data = np.array(
            [
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        sieved = sieve_filter(labeled, min_size=1, use_gpu=True)
        sieved_data = sieved.to_numpy()
        # Background stays 0
        assert sieved_data[0, 0] == 0
        assert sieved_data[1, 1] == 0
        # Foreground pixel stays labeled
        assert sieved_data[0, 1] != 0

    def test_gpu_larger_raster_matches_cpu(self):
        """GPU sieve matches CPU on a larger raster with many components."""
        rng = np.random.RandomState(42)
        data = rng.randint(0, 5, size=(30, 30)).astype(np.int32)
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        cpu_sieved = sieve_filter(labeled, min_size=5, use_gpu=False)
        gpu_sieved = sieve_filter(labeled, min_size=5, use_gpu=True)
        np.testing.assert_array_equal(cpu_sieved.to_numpy(), gpu_sieved.to_numpy())

    def test_gpu_sieve_diagnostics(self):
        """GPU sieve records diagnostic events."""
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        sieved = sieve_filter(labeled, min_size=3, use_gpu=True)
        runtime_events = [
            e for e in sieved.diagnostics if e.kind == "runtime" and "sieve_filter_gpu" in e.detail
        ]
        assert len(runtime_events) >= 1
        assert runtime_events[0].elapsed_seconds > 0

    def test_gpu_nodata_preserved(self):
        """GPU sieve preserves nodata value correctly."""
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data, nodata=0)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        cpu_sieved = sieve_filter(labeled, min_size=3, use_gpu=False)
        gpu_sieved = sieve_filter(labeled, min_size=3, use_gpu=True)
        np.testing.assert_array_equal(cpu_sieved.to_numpy(), gpu_sieved.to_numpy())


# ---------------------------------------------------------------------------
# Dispatcher tests (auto-dispatch, falls back to CPU gracefully)
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_label_auto_dispatch(self):
        """label_connected_components works with auto-dispatch."""
        data = np.array(
            [
                [1, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        result = label_connected_components(raster)
        assert result is not None
        labeled = result.to_numpy()
        assert labeled[0, 0] != 0

    def test_morphology_auto_dispatch(self):
        """raster_morphology works with auto-dispatch."""
        data = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
        raster = from_numpy(data)
        result = raster_morphology(raster, "dilate")
        assert result is not None

    def test_label_force_cpu(self):
        """Forcing CPU works regardless of GPU availability."""
        data = np.array([[1, 0, 1]], dtype=np.int32)
        raster = from_numpy(data)
        result = label_connected_components(raster, connectivity=4, use_gpu=False)
        labeled = result.to_numpy()
        assert labeled[0, 0] != labeled[0, 2]

    def test_morphology_force_cpu(self):
        """Forcing CPU works for morphology."""
        data = np.ones((3, 3), dtype=np.float32)
        raster = from_numpy(data)
        result = raster_morphology(raster, "erode", use_gpu=False)
        assert result is not None

    def test_sieve_auto_dispatch(self):
        """sieve_filter works with auto-dispatch."""
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        result = sieve_filter(labeled, min_size=3)
        assert result is not None

    def test_sieve_force_cpu(self):
        """Forcing CPU works for sieve_filter."""
        data = np.array(
            [
                [1, 1, 0, 2, 0],
                [1, 1, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        raster = from_numpy(data)
        labeled = label_connected_components(raster, connectivity=4, use_gpu=False)
        result = sieve_filter(labeled, min_size=3, use_gpu=False)
        assert result is not None
        assert result.to_numpy()[0, 0] != 0
        assert result.to_numpy()[0, 3] == 0
