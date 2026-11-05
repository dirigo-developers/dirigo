from dataclasses import dataclass
from typing import Any, Literal

import numpy.typing as npt

from dirigo.pipeline.product import Metadata

type FrameArray = npt.NDArray[Any]
type FrameInterpretation = Literal["monochrome", "multichannel", "rgb"]


@dataclass(frozen=True, slots=True, kw_only=True)
class FrameMetadata(Metadata):
    """Coordinates of one XY frame containing all its channels.

    time_index identifies a time point, including a volume time point
    when acquiring repeated Z-stacks. z_index identifies a depth plane.

    None means unspecified. Coordinates do not prescribe delivery order.
    """

    time_index: int | None = None
    z_index: int | None = None

    def __post_init__(self) -> None:
        for name, index in (
            ("time_index", self.time_index),
            ("z_index", self.z_index),
        ):
            if index is None:
                continue

            if not isinstance(index, int) or isinstance(index, bool):
                raise TypeError(f"{name} must be an integer or None.")

            if index < 0:
                raise ValueError(f"{name} must be nonnegative.")
