"""Validate immutable per-run capture requests independently of drivers."""

from dataclasses import FrozenInstanceError, replace

import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.timing import DigitalStartTrigger, ExternalSampleClock


def test_capture_settings_defaults_and_immutability() -> None:
    """Default to one finite, single-channel batch and reject later mutation."""
    settings = CaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=8,
    )

    assert settings.channel_indices == (0,)
    assert settings.records_per_batch == 1
    assert settings.batch_count == 1
    assert settings.timestamps_enabled is False
    assert settings.external_clock is None
    assert settings.start_trigger is None

    with pytest.raises(FrozenInstanceError):
        settings.record_length = 16  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"sample_rate": 1000}, TypeError, "sample_rate"),
        ({"sample_rate": units.SampleRate(0)}, ValueError, "sample_rate"),
        ({"sample_rate": units.SampleRate(float("inf"))}, ValueError, "sample_rate"),
        ({"channel_indices": [0]}, TypeError, "tuple"),
        ({"channel_indices": ()}, ValueError, "At least one"),
        ({"channel_indices": (True,)}, TypeError, "integers"),
        ({"channel_indices": (-1,)}, ValueError, "nonnegative"),
        ({"channel_indices": (0, 0)}, ValueError, "distinct"),
        ({"record_length": True}, TypeError, "record_length"),
        ({"record_length": 0}, ValueError, "record_length"),
        ({"records_per_batch": 1.5}, TypeError, "records_per_batch"),
        ({"records_per_batch": 0}, ValueError, "records_per_batch"),
        ({"batch_count": True}, TypeError, "batch_count"),
        ({"batch_count": 0}, ValueError, "batch_count"),
        ({"timestamps_enabled": 1}, TypeError, "timestamps_enabled"),
        ({"external_clock": "PFI0"}, TypeError, "external_clock"),
        ({"start_trigger": "PFI1"}, TypeError, "start_trigger"),
    ],
)
def test_capture_settings_reject_invalid_fields(
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject malformed requests before a driver or capture session is needed."""
    settings = CaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=8,
    )

    with pytest.raises(error_type, match=message):
        replace(settings, **changes)


def test_capture_settings_accept_optional_timing_and_unlimited_delivery() -> None:
    """Retain typed timing requests without requiring universal backend support.

    The generic model permits unlimited delivery and an external sample clock
    with a whole-run start trigger. A concrete driver decides whether it can
    implement that combination.
    """
    clock = ExternalSampleClock(
        source="PFI0",
        max_rate=units.SampleRate("2 kS/s"),
    )
    trigger = DigitalStartTrigger(source="PFI1")
    settings = CaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=8,
        batch_count=None,
        external_clock=clock,
        start_trigger=trigger,
    )

    assert settings.batch_count is None
    assert settings.external_clock is clock
    assert settings.start_trigger is trigger

    with pytest.raises(ValueError, match="max_rate"):
        replace(settings, sample_rate=units.SampleRate("3 kS/s"))
