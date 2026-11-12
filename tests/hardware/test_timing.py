"""Test backend-independent external sample-clock descriptions."""

from dataclasses import replace

import pytest

from dirigo import units
from dirigo.hardware.timing import ClockEdge, ExternalSampleClock


def test_external_clock_accepts_opaque_source_and_defaults_to_rising() -> None:
    """Accept a backend-defined identifier without imposing NI terminal syntax."""
    clock = ExternalSampleClock(
        source="line_sync", max_rate=units.SampleRate("1.2 kS/s")
    )

    assert clock.source == "line_sync"
    assert clock.max_rate == units.SampleRate("1.2 kS/s")
    assert clock.active_edge is ClockEdge.RISING


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"source": None}, TypeError, "source must be a string"),
        ({"source": ""}, ValueError, "source must be nonempty"),
        ({"source": "  "}, ValueError, "source must be nonempty"),
        ({"source": " line_sync"}, ValueError, "surrounding whitespace"),
        ({"max_rate": 1200.0}, TypeError, "max_rate must be a SampleRate"),
        ({"max_rate": units.SampleRate(0.0)}, ValueError, "finite and positive"),
        ({"max_rate": units.SampleRate(-1.0)}, ValueError, "finite and positive"),
        (
            {"max_rate": units.SampleRate(float("inf"))},
            ValueError,
            "finite and positive",
        ),
        ({"active_edge": "rising"}, TypeError, "active_edge must be a ClockEdge"),
    ],
)
def test_external_clock_rejects_invalid_fields(
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject malformed clock fields independently of any hardware backend."""
    clock = ExternalSampleClock(
        source="line_sync", max_rate=units.SampleRate("1.2 kS/s")
    )

    with pytest.raises(error_type, match=message):
        replace(clock, **changes)
