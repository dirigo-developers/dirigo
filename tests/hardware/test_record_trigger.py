"""Validate record-trigger requests and delivery with nonconsecutive records."""

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureInfo, CaptureSettings
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    DigitalStartTrigger,
    SignalEdge,
)
from dirigo.hardware.waveform import AngularWaveform

from .capture_stubs import RecordingDigitizer, RecordingSession


def test_record_trigger_defaults_and_signed_window_offsets() -> None:
    """Keep consecutive capture as the default and accept signed record offsets."""
    settings = CaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
    )
    assert settings.record_trigger is None
    assert settings.trigger_offset_samples == 0

    trigger = DigitalRecordTrigger(source="external")
    assert trigger.active_edge is SignalEdge.RISING

    for offset in (-128, 0, 64):
        requested = replace(
            settings,
            record_trigger=trigger,
            trigger_offset_samples=offset,
        )
        assert requested.record_trigger is trigger
        assert requested.trigger_offset_samples == offset


@pytest.mark.parametrize("source", ["", " ", " external", "external "])
def test_record_trigger_rejects_invalid_source_names(source: str) -> None:
    """Reject empty or padded source identifiers before hardware access."""
    with pytest.raises(ValueError, match="source"):
        DigitalRecordTrigger(source=source)


def test_record_trigger_rejects_incorrect_field_types() -> None:
    """Require a string source and SignalEdge rather than untyped substitutes."""
    with pytest.raises(TypeError, match="source"):
        DigitalRecordTrigger(source=cast(str, 123))

    with pytest.raises(TypeError, match="active_edge"):
        DigitalRecordTrigger(
            source="external",
            active_edge=cast(SignalEdge, "rising"),
        )


@pytest.mark.parametrize("offset", [True, 1.5])
def test_record_offset_requires_an_integer(offset: object) -> None:
    """Reject booleans and fractional sample offsets."""
    settings = CaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
        record_trigger=DigitalRecordTrigger(source="external"),
    )

    with pytest.raises(TypeError, match="trigger_offset_samples"):
        replace(settings, trigger_offset_samples=offset)


def test_record_offset_requires_a_record_trigger() -> None:
    """Reject trigger-relative positioning for consecutive untriggered windows."""
    with pytest.raises(ValueError, match="requires a record_trigger"):
        CaptureSettings(
            sample_rate=units.SampleRate("100 MS/s"),
            record_length=4096,
            trigger_offset_samples=-128,
        )


def test_start_and_record_trigger_roles_are_distinct() -> None:
    """Allow both roles while rejecting a description supplied for the wrong role."""
    start = DigitalStartTrigger(source="run_gate")
    record = DigitalRecordTrigger(
        source="external",
        active_edge=SignalEdge.FALLING,
    )
    settings = CaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
        start_trigger=start,
        record_trigger=record,
    )

    assert settings.start_trigger is start
    assert settings.record_trigger is record

    with pytest.raises(TypeError, match="record_trigger"):
        replace(settings, record_trigger=start)

    with pytest.raises(TypeError, match="start_trigger"):
        replace(settings, start_trigger=record)


def test_triggered_batches_preserve_gaps_with_a_different_output_rate() -> None:
    """Deliver record metadata without imposing AO/ADC sample equality.

    Model four record triggers at AO sample indices 0, 100, 200, and 300 in a
    100 kS/s waveform. A 100 MS/s digitizer returns 4096 samples per record in
    two batches. First-sample times therefore have millisecond spacing.

    The recording backend supplies acquisition results. This exercises the
    contract, rather than physical trigger delivery or board capabilities.
    """
    settings = CaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
        records_per_batch=2,
        batch_count=2,
        timestamps_enabled=True,
        record_trigger=DigitalRecordTrigger(source="external"),
    )
    output = AngularWaveform(
        np.zeros(400),
        sample_rate=units.SampleRate("100 kS/s"),
    )
    record_times = np.array([0, 100, 200, 300]) / float(output.sample_rate)
    assert settings.sample_rate != output.sample_rate

    digitizer = RecordingDigitizer()
    session = digitizer.create_capture(settings)
    assert isinstance(session, RecordingSession)

    retained: list[CaptureInfo] = []
    try:
        session.prepare()
        session.start()
        samples = np.empty(session.layout.shape, dtype=session.layout.dtype)
        assert samples.shape == (2, 4096, 1)

        for batch in range(2):
            session.result = CaptureInfo(record_times[2 * batch : 2 * batch + 2])
            info = session.read_into(samples)
            assert info is not None
            retained.append(info)

        assert session.state is CaptureState.COMPLETED
        assert session.read_into(samples) is None
    finally:
        session.close()
        digitizer.close()

    for index, info in enumerate(retained):
        timestamps = info.timestamps
        assert timestamps is not None
        np.testing.assert_allclose(
            timestamps,
            record_times[2 * index : 2 * index + 2],
        )
        assert timestamps[1] - timestamps[0] > (
            settings.record_length / float(settings.sample_rate)
        )
