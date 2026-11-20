"""Reconstructed line batches published by line processors."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from dirigo import units
from dirigo.hardware.capture import CaptureInfo


@dataclass(frozen=True, slots=True, eq=False)
class LineBatch:
    """Read-only signed pixels with shape (lines, columns, channels).

    The batch owns its pixels. Columns increase along the fast axis, and
    channel_indices gives the physical input order. Each record contributes
    lines_per_record consecutive rows in sweep order, starting at first_record.
    info.timestamps, when present, contains one first-sample time per record.
    """

    pixels: npt.NDArray[np.int16]
    pixel_width: units.Length
    channel_indices: tuple[int, ...]
    first_record: int
    lines_per_record: int
    info: CaptureInfo
