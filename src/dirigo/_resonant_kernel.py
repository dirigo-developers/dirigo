"""Compiled sample accumulation and signed pixel encoding."""

import numpy as np
import numpy.typing as npt
from numba import njit, prange, types

from dirigo.hardware.capture import SampleArray

_SIGNATURES = [
    types.void(
        types.Array(dtype, 3, "C", readonly=True),
        types.Array(types.intp, 2, "C"),
        types.float64,
        types.float64,
        types.Array(types.int16, 3, "C"),
    )
    for dtype in (types.int8, types.uint8, types.int16, types.uint16)
]


@njit(_SIGNATURES, cache=True, nogil=True, parallel=True)
def _reconstruct_int16(
    samples: SampleArray,
    sample_edges: npt.NDArray[np.intp],
    zero_code: float,
    gain: float,
    pixels: npt.NDArray[np.int16],
) -> None:
    """Accumulate prepared bins and write scaled pixels in spatial order."""
    records, _, channels = samples.shape
    sweeps, edge_count = sample_edges.shape
    columns = edge_count - 1

    for record in prange(records):
        record_index = np.int64(record)
        for sweep in range(sweeps):
            line = record_index * sweeps + sweep
            for time_column in range(columns):
                start = sample_edges[sweep, time_column]
                stop = sample_edges[sweep, time_column + 1]
                column = time_column if sweep == 0 else columns - 1 - time_column
                samples_in_bin = stop - start

                for channel in range(channels):
                    total = np.int64(0)
                    for sample in range(start, stop):
                        total += np.int64(samples[record_index, sample, channel])

                    average = total / samples_in_bin
                    rounded = np.rint((average - zero_code) * gain)
                    pixels[line, column, channel] = min(32767.0, max(-32768.0, rounded))
