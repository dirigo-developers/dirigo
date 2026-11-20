"""Shared signed pixel encoding for raster processors."""

import numpy as np
import numpy.typing as npt

from dirigo import units


def _encode_int16(
    averages: npt.NDArray[np.float64], sample_code_range: units.IntRange
) -> npt.NDArray[np.int16]:
    """Scale scratch averages in place and return new signed pixel storage."""
    code_count = sample_code_range.count
    zero_code = sample_code_range.min + code_count // 2
    averages -= zero_code
    averages *= 2**16 / code_count
    np.rint(averages, out=averages)
    np.clip(averages, -32768, 32767, out=averages)
    return averages.astype(np.int16)
