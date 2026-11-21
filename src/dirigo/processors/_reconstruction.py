"""Prepare reconstruction for a raster processor's upstream capture layout."""

from importlib import import_module

from numba import get_num_threads, threading_layer

from dirigo.resonant_raster import ResonantFastAxisPlan, ResonantLineReconstructor
from dirigo.sources.raster_acquisition import RasterAcquisition


def _prepare_reconstructor(
    source: RasterAcquisition, fast_axis: ResonantFastAxisPlan, *, columns: int
) -> tuple[ResonantLineReconstructor, bool]:
    """Build the bins and load the eagerly compiled kernel when selected."""
    reconstructor = ResonantLineReconstructor(fast_axis, columns=columns)
    if source.capture_layout.dtype.itemsize > 2:
        return reconstructor, False

    # Initialize the threading layer before querying its name.
    get_num_threads()
    if threading_layer() == "workqueue":
        raise RuntimeError(
            "Parallel raster reconstruction requires a thread-safe Numba "
            "threading layer. Numba selected 'workqueue'. Enable TBB or "
            "OpenMP and restart with NUMBA_THREADING_LAYER=threadsafe."
        )

    import_module("dirigo._resonant_kernel")
    return reconstructor, True
