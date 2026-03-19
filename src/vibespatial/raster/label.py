"""Connected component labeling, sieve filtering, and morphology.

CPU baseline uses scipy.ndimage. GPU path uses custom NVRTC union-find
kernels (kernels/ccl.py) and morphology stencil kernels (kernels/morphology.py).

ADR-0040: CCCL Connected Component Labeling
"""

from __future__ import annotations

import logging
import time

import numpy as np

from vibespatial.raster.buffers import (
    OwnedRasterArray,
    RasterDiagnosticEvent,
    RasterDiagnosticKind,
    from_numpy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _structure_for_connectivity(connectivity: int) -> np.ndarray:
    """Return a structuring element for the given connectivity."""
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    elif connectivity == 8:
        return np.ones((3, 3), dtype=np.int32)
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def _has_cupy() -> bool:
    """Return True if CuPy is importable."""
    try:
        import cupy  # noqa: F401

        return True
    except ImportError:
        return False


def _should_use_gpu(raster: OwnedRasterArray, threshold: int = 100_000) -> bool:
    """Auto-dispatch heuristic: use GPU when available and image is large enough."""
    try:
        import cupy  # noqa: F401

        from vibespatial.cuda_runtime import get_cuda_runtime

        runtime = get_cuda_runtime()
        return runtime.available() and raster.pixel_count >= threshold
    except (ImportError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# CPU baseline: connected component labeling
# ---------------------------------------------------------------------------


def _label_cpu(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
) -> OwnedRasterArray:
    """CPU connected component labeling via scipy.ndimage.label."""
    from scipy.ndimage import label as scipy_label

    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("connected component labeling requires a single-band raster")
        data = data[0]

    # Build foreground mask
    foreground = data != 0
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            foreground &= ~np.isnan(data)
        else:
            foreground &= data != raster.nodata

    structure = _structure_for_connectivity(connectivity)
    labeled, num_features = scipy_label(foreground.astype(np.int32), structure=structure)

    result = from_numpy(
        labeled.astype(np.int32),
        nodata=0,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"label_cpu components={num_features} connectivity={connectivity}",
            residency=result.residency,
        )
    )
    return result


# ---------------------------------------------------------------------------
# GPU: connected component labeling (iterative union-find)
# ---------------------------------------------------------------------------


def label_gpu(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
) -> OwnedRasterArray:
    """GPU connected component labeling using iterative union-find.

    Uses NVRTC kernels: init_labels -> local_merge -> pointer_jump (iterate)
    -> relabel to compact sequential labels 1..N.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster. Nonzero values are foreground.
    connectivity : int
        4 or 8 neighbor connectivity.

    Returns
    -------
    OwnedRasterArray
        HOST-resident integer-labeled raster (int32, nodata=0).
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.ccl import (
        INIT_LABELS_SOURCE,
        LOCAL_MERGE_4C_SOURCE,
        LOCAL_MERGE_8C_SOURCE,
        POINTER_JUMP_SOURCE,
        RELABEL_SOURCE,
    )

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Prepare data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        if data.shape[0] != 1:
            raise ValueError("connected component labeling requires a single-band raster")
        data = data[0]

    height, width = data.shape
    n = height * width

    # Build foreground mask on host, then transfer once
    foreground_host = (data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            foreground_host &= (~np.isnan(data)).astype(np.uint8)
        else:
            foreground_host &= (data != raster.nodata).astype(np.uint8)

    # H->D transfer (only transfer at start)
    d_foreground = cp.asarray(np.ascontiguousarray(foreground_host.ravel()))

    # Allocate device buffers
    d_labels = cp.empty(n, dtype=np.int32)
    d_changed = cp.zeros(1, dtype=np.int32)

    # --- Compile kernels ---
    # Init labels kernel
    init_key = make_kernel_cache_key("init_labels", INIT_LABELS_SOURCE)
    init_kernels = runtime.compile_kernels(
        cache_key=init_key,
        source=INIT_LABELS_SOURCE,
        kernel_names=("init_labels",),
    )

    # Merge kernel (4c or 8c)
    if connectivity == 4:
        merge_source = LOCAL_MERGE_4C_SOURCE
        merge_name = "local_merge_4c"
    else:
        merge_source = LOCAL_MERGE_8C_SOURCE
        merge_name = "local_merge_8c"

    merge_key = make_kernel_cache_key(merge_name, merge_source)
    merge_kernels = runtime.compile_kernels(
        cache_key=merge_key,
        source=merge_source,
        kernel_names=(merge_name,),
    )

    # Pointer jump kernel
    pj_key = make_kernel_cache_key("pointer_jump", POINTER_JUMP_SOURCE)
    pj_kernels = runtime.compile_kernels(
        cache_key=pj_key,
        source=POINTER_JUMP_SOURCE,
        kernel_names=("pointer_jump",),
    )

    # Relabel kernel
    relabel_key = make_kernel_cache_key("relabel", RELABEL_SOURCE)
    relabel_kernels = runtime.compile_kernels(
        cache_key=relabel_key,
        source=RELABEL_SOURCE,
        kernel_names=("relabel",),
    )

    # --- Phase 1: Init labels ---
    block_1d = (256, 1, 1)
    grid_1d = ((n + 255) // 256, 1, 1)

    runtime.launch(
        kernel=init_kernels["init_labels"],
        grid=grid_1d,
        block=block_1d,
        params=(
            (d_labels.data.ptr, d_foreground.data.ptr, n),
            (KERNEL_PARAM_PTR, KERNEL_PARAM_PTR, KERNEL_PARAM_I32),
        ),
    )

    # --- Phases 2-3: Iterate merge + pointer jump until convergence ---
    block_2d = (16, 16, 1)
    grid_2d = ((width + 15) // 16, (height + 15) // 16, 1)

    max_iterations = 1000
    for _iteration in range(max_iterations):
        # Reset changed flag
        d_changed.fill(0)

        # Phase 2: Local merge
        runtime.launch(
            kernel=merge_kernels[merge_name],
            grid=grid_2d,
            block=block_2d,
            params=(
                (d_labels.data.ptr, d_foreground.data.ptr, width, height, d_changed.data.ptr),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_PTR,
                ),
            ),
        )

        # Phase 3: Pointer jumping (multiple rounds per iteration)
        for _pj in range(10):
            d_pj_changed = cp.zeros(1, dtype=np.int32)
            runtime.launch(
                kernel=pj_kernels["pointer_jump"],
                grid=grid_1d,
                block=block_1d,
                params=(
                    (d_labels.data.ptr, n, d_pj_changed.data.ptr),
                    (KERNEL_PARAM_PTR, KERNEL_PARAM_I32, KERNEL_PARAM_PTR),
                ),
            )
            if int(d_pj_changed.item()) == 0:
                break

        if int(d_changed.item()) == 0:
            break

    iterations_done = _iteration + 1

    # --- Phase 4: Compact relabel using CuPy sort + unique ---
    # Extract only foreground labels
    h_labels = cp.asnumpy(d_labels)
    fg_mask = h_labels >= 0
    if not fg_mask.any():
        # No foreground pixels at all
        result = from_numpy(
            np.zeros((height, width), dtype=np.int32),
            nodata=0,
            affine=raster.affine,
            crs=raster.crs,
        )
        elapsed = time.perf_counter() - t0
        result.diagnostics.append(
            RasterDiagnosticEvent(
                kind=RasterDiagnosticKind.RUNTIME,
                detail=f"label_gpu components=0 connectivity={connectivity} iterations={iterations_done} elapsed={elapsed:.3f}s",
                residency=result.residency,
                visible_to_user=True,
                elapsed_seconds=elapsed,
            )
        )
        return result

    # Find unique root labels and build compact mapping
    unique_roots = np.unique(h_labels[fg_mask])
    num_components = len(unique_roots)

    # Build compact IDs 1..N
    compact_ids = np.arange(1, num_components + 1, dtype=np.int32)

    # Transfer mapping to device
    d_unique_roots = cp.asarray(unique_roots.astype(np.int32))
    d_compact_ids = cp.asarray(compact_ids)

    # Launch relabel kernel
    runtime.launch(
        kernel=relabel_kernels["relabel"],
        grid=grid_1d,
        block=block_1d,
        params=(
            (d_labels.data.ptr, d_compact_ids.data.ptr, d_unique_roots.data.ptr, num_components, n),
            (
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_PTR,
                KERNEL_PARAM_I32,
                KERNEL_PARAM_I32,
            ),
        ),
    )

    # --- D->H transfer (final) ---
    host_labels = cp.asnumpy(d_labels).reshape(height, width)

    elapsed = time.perf_counter() - t0
    result = from_numpy(
        host_labels.astype(np.int32),
        nodata=0,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"label_gpu components={num_components} connectivity={connectivity} "
                f"iterations={iterations_done} pixels={n} elapsed={elapsed:.3f}s"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    return result


# ---------------------------------------------------------------------------
# GPU: morphology (erosion, dilation, open, close)
# ---------------------------------------------------------------------------


def morphology_gpu(
    raster: OwnedRasterArray,
    operation: str,
    *,
    connectivity: int = 4,
    iterations: int = 1,
) -> OwnedRasterArray:
    """GPU binary morphology using NVRTC 3x3 stencil kernels.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    operation : str
        One of "erode", "dilate", "open", "close".
    connectivity : int
        4 or 8 neighbor connectivity for the structuring element.
    iterations : int
        Number of times to apply the operation.

    Returns
    -------
    OwnedRasterArray
        HOST-resident result raster.
    """
    import cupy as cp

    from vibespatial.cuda_runtime import (
        KERNEL_PARAM_I32,
        KERNEL_PARAM_PTR,
        get_cuda_runtime,
        make_kernel_cache_key,
    )
    from vibespatial.raster.kernels.morphology import (
        BINARY_DILATE_KERNEL_SOURCE,
        BINARY_ERODE_KERNEL_SOURCE,
    )

    valid_ops = ("erode", "dilate", "open", "close")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {list(valid_ops)}, got {operation!r}")

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    t0 = time.perf_counter()
    runtime = get_cuda_runtime()

    # --- Prepare data on device ---
    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    height, width = data.shape

    # Build binary foreground mask
    binary = (data != 0).astype(np.uint8)
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            binary &= (~np.isnan(data)).astype(np.uint8)
        else:
            binary &= (data != raster.nodata).astype(np.uint8)

    # H->D transfer
    d_input = cp.asarray(np.ascontiguousarray(binary.ravel()))

    # Build structuring element
    structure = _structure_for_connectivity(connectivity).astype(np.uint8)
    d_structure = cp.asarray(np.ascontiguousarray(structure.ravel()))

    # Allocate output
    n = height * width
    d_output = cp.empty(n, dtype=np.uint8)

    # --- Compile kernels (shared-memory tiled versions) ---
    erode_key = make_kernel_cache_key("binary_erode", BINARY_ERODE_KERNEL_SOURCE)
    erode_kernels = runtime.compile_kernels(
        cache_key=erode_key,
        source=BINARY_ERODE_KERNEL_SOURCE,
        kernel_names=("binary_erode",),
    )

    dilate_key = make_kernel_cache_key("binary_dilate", BINARY_DILATE_KERNEL_SOURCE)
    dilate_kernels = runtime.compile_kernels(
        cache_key=dilate_key,
        source=BINARY_DILATE_KERNEL_SOURCE,
        kernel_names=("binary_dilate",),
    )

    block_2d = (16, 16, 1)
    grid_2d = ((width + 15) // 16, (height + 15) // 16, 1)

    def _run_erode(d_in: object, d_out: object) -> None:
        runtime.launch(
            kernel=erode_kernels["binary_erode"],
            grid=grid_2d,
            block=block_2d,
            params=(
                (d_in.data.ptr, d_out.data.ptr, d_structure.data.ptr, width, height),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

    def _run_dilate(d_in: object, d_out: object) -> None:
        runtime.launch(
            kernel=dilate_kernels["binary_dilate"],
            grid=grid_2d,
            block=block_2d,
            params=(
                (d_in.data.ptr, d_out.data.ptr, d_structure.data.ptr, width, height),
                (
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_PTR,
                    KERNEL_PARAM_I32,
                    KERNEL_PARAM_I32,
                ),
            ),
        )

    # Determine sequence of operations
    if operation == "erode":
        ops_sequence = ["erode"] * iterations
    elif operation == "dilate":
        ops_sequence = ["dilate"] * iterations
    elif operation == "open":
        # Open = erode then dilate, repeated
        ops_sequence = (["erode"] * iterations) + (["dilate"] * iterations)
    elif operation == "close":
        # Close = dilate then erode, repeated
        ops_sequence = (["dilate"] * iterations) + (["erode"] * iterations)
    else:
        raise ValueError(f"Unknown operation: {operation!r}")

    # Run the operation sequence, ping-ponging between buffers
    current_in = d_input
    current_out = d_output
    for op in ops_sequence:
        if op == "erode":
            _run_erode(current_in, current_out)
        else:
            _run_dilate(current_in, current_out)
        # Swap buffers for next iteration
        current_in, current_out = current_out, current_in

    # After the loop, the result is in current_in (last swap put it there)
    d_result = current_in

    # --- D->H transfer ---
    host_result = cp.asnumpy(d_result).reshape(height, width)
    elapsed = time.perf_counter() - t0
    kernel_launches = len(ops_sequence)

    result = from_numpy(
        host_result.astype(np.uint8),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=(
                f"morphology_gpu op={operation} connectivity={connectivity} "
                f"iterations={iterations} pixels={n} "
                f"kernel_launches={kernel_launches} grid={grid_2d}"
            ),
            residency=result.residency,
            visible_to_user=True,
            elapsed_seconds=elapsed,
        )
    )
    logger.debug(
        "morphology_gpu op=%s conn=%d iter=%d pixels=%d elapsed=%.4fs",
        operation,
        connectivity,
        iterations,
        n,
        elapsed,
    )
    return result


# ---------------------------------------------------------------------------
# Public API: dispatchers (GPU/CPU auto-selection)
# ---------------------------------------------------------------------------


def label_connected_components(
    raster: OwnedRasterArray,
    *,
    connectivity: int = 4,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Label connected components in a raster.

    Each group of connected nonzero (and non-nodata) pixels receives a unique
    integer label. Background (zero or nodata) pixels get label 0.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input raster. Nonzero values are foreground.
    connectivity : int
        4 or 8 neighbor connectivity.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).
        Auto uses GPU when available and pixel count exceeds threshold.

    Returns
    -------
    OwnedRasterArray
        Integer-labeled raster where each connected component has a unique label.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return label_gpu(raster, connectivity=connectivity)
    else:
        return _label_cpu(raster, connectivity=connectivity)


def sieve_filter(
    labeled: OwnedRasterArray,
    min_size: int,
    *,
    connectivity: int = 4,
    replace_value: int = 0,
) -> OwnedRasterArray:
    """Remove small connected components from a labeled raster.

    Parameters
    ----------
    labeled : OwnedRasterArray
        Integer-labeled raster (e.g., from label_connected_components).
    min_size : int
        Minimum pixel count to keep a component.
    connectivity : int
        4 or 8 neighbor connectivity (used for counting).
    replace_value : int
        Value to assign to removed components (default 0 = background).

    Returns
    -------
    OwnedRasterArray
        Sieved raster with small components replaced.
    """
    data = labeled.to_numpy().copy()
    if data.ndim == 3:
        data = data[0]

    unique_labels, counts = np.unique(data, return_counts=True)
    small_labels = unique_labels[counts < min_size]

    # Don't remove the background label
    nodata = labeled.nodata
    if nodata is not None:
        small_labels = small_labels[small_labels != int(nodata)]

    mask = np.isin(data, small_labels)
    data[mask] = replace_value

    removed_count = len(small_labels)
    result = from_numpy(data, nodata=nodata, affine=labeled.affine, crs=labeled.crs)
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"sieve_filter removed={removed_count} min_size={min_size}",
            residency=result.residency,
        )
    )
    return result


def raster_morphology(
    raster: OwnedRasterArray,
    operation: str,
    *,
    connectivity: int = 4,
    iterations: int = 1,
    use_gpu: bool | None = None,
) -> OwnedRasterArray:
    """Apply binary morphological operation to a raster.

    Parameters
    ----------
    raster : OwnedRasterArray
        Input binary raster (nonzero = foreground).
    operation : str
        One of "erode", "dilate", "open", "close".
    connectivity : int
        4 or 8 neighbor connectivity for the structuring element.
    iterations : int
        Number of times to apply the operation.
    use_gpu : bool or None
        Force GPU (True), force CPU (False), or auto-dispatch (None).

    Returns
    -------
    OwnedRasterArray
        Result raster.
    """
    valid_ops = ("erode", "dilate", "open", "close")
    if operation not in valid_ops:
        raise ValueError(f"operation must be one of {list(valid_ops)}, got {operation!r}")

    if use_gpu is None:
        use_gpu = _should_use_gpu(raster)

    if use_gpu:
        return morphology_gpu(raster, operation, connectivity=connectivity, iterations=iterations)
    else:
        return _morphology_cpu(raster, operation, connectivity=connectivity, iterations=iterations)


# ---------------------------------------------------------------------------
# CPU baseline: morphology
# ---------------------------------------------------------------------------


def _morphology_cpu(
    raster: OwnedRasterArray,
    operation: str,
    *,
    connectivity: int = 4,
    iterations: int = 1,
) -> OwnedRasterArray:
    """CPU binary morphology via scipy.ndimage."""
    from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_opening

    ops = {
        "erode": binary_erosion,
        "dilate": binary_dilation,
        "open": binary_opening,
        "close": binary_closing,
    }

    data = raster.to_numpy()
    if data.ndim == 3:
        data = data[0]

    binary = data != 0
    if raster.nodata is not None:
        if np.isnan(raster.nodata):
            binary &= ~np.isnan(data)
        else:
            binary &= data != raster.nodata

    structure = _structure_for_connectivity(connectivity)
    result_data = ops[operation](binary, structure=structure, iterations=iterations)

    result = from_numpy(
        result_data.astype(raster.dtype),
        nodata=raster.nodata,
        affine=raster.affine,
        crs=raster.crs,
    )
    result.diagnostics.append(
        RasterDiagnosticEvent(
            kind=RasterDiagnosticKind.RUNTIME,
            detail=f"morphology_cpu op={operation} connectivity={connectivity} iterations={iterations}",
            residency=result.residency,
        )
    )
    return result
