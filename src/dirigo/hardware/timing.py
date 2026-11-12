"""Clock descriptions shared by hardware output backends."""

import math
from dataclasses import dataclass
from enum import StrEnum

from dirigo import units


class ClockEdge(StrEnum):
    """Select the transition that advances a clocked operation."""

    RISING = "rising"
    FALLING = "falling"


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalSampleClock:
    """Describe a sample clock supplied through a backend-defined source.

    source is an opaque identifier interpreted by the backend. max_rate is the
    maximum expected rate of active edges, distinct from a waveform's nominal
    sample rate. This description does not generate clock pulses.
    """

    source: str
    max_rate: units.SampleRate
    active_edge: ClockEdge = ClockEdge.RISING

    def __post_init__(self) -> None:
        """Require a source, positive finite maximum rate, and typed edge."""
        if not isinstance(self.source, str):
            raise TypeError("source must be a string.")
        if not self.source or self.source != self.source.strip():
            raise ValueError(
                "source must be nonempty and have no surrounding whitespace."
            )
        if not isinstance(self.max_rate, units.SampleRate):
            raise TypeError("max_rate must be a SampleRate.")
        if not math.isfinite(float(self.max_rate)) or float(self.max_rate) <= 0:
            raise ValueError("max_rate must be finite and positive.")
        if not isinstance(self.active_edge, ClockEdge):
            raise TypeError("active_edge must be a ClockEdge.")
