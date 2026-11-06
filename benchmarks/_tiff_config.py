"""Configuration and deadline arithmetic, independent of pipeline execution."""

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TiffBenchmarkConfig:
    """Describe products containing all channels and their expected arrival rate."""

    width: int
    height: int
    channels: int = 1
    dtype: Literal["uint8", "uint16"] = "uint16"
    interpretation: Literal["monochrome", "multichannel", "rgb"] = "monochrome"
    fps: float = 30.0
    duration: float = 10.0
    buffers: int = 16
    contiguous: bool = False

    def __post_init__(self) -> None:
        for name in ("width", "height", "channels", "buffers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("fps", "duration"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.fps * self.duration) or self.fps * self.duration <= 0:
            raise ValueError("fps * duration must be finite and positive.")
        if self.dtype not in ("uint8", "uint16"):
            raise ValueError("dtype must be uint8 or uint16.")
        if self.interpretation not in ("monochrome", "multichannel", "rgb"):
            raise ValueError("Unknown interpretation.")
        if self.interpretation == "monochrome" and self.channels != 1:
            raise ValueError("monochrome requires channels=1.")
        if self.interpretation == "rgb" and self.channels != 3:
            raise ValueError("rgb requires channels=3.")
        if not isinstance(self.contiguous, bool):
            raise TypeError("contiguous must be a boolean.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the writer's input shape; channels are always last."""
        if self.interpretation == "monochrome":
            return (self.height, self.width)
        return (self.height, self.width, self.channels)

    @property
    def frame_bytes(self) -> int:
        """Pixel payload size, excluding TIFF headers and metadata."""
        return self.width * self.height * self.channels * (int(self.dtype[4:]) // 8)

    @property
    def required_bytes_per_second(self) -> float:
        """Expected pixel throughput for products containing all channels."""
        return self.frame_bytes * self.fps

    @property
    def scheduled_frames(self) -> int:
        """Number of frame arrivals in the half-open interval [0, duration)."""
        return math.ceil(self.duration * self.fps)


def due_slot(index: int, elapsed: float, fps: float, count: int) -> tuple[int, int]:
    """Skip expired frame intervals instead of publishing a catch-up burst.

    The current interval may be serviced late. Whole missed intervals are
    reported separately from buffer exhaustion, and never shift the schedule.
    """
    current = min(count, max(index, math.floor(max(0.0, elapsed) * fps)))
    return current, current - index
