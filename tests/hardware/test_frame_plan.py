"""Test acquired-record grouping independently of transfer and hardware timing."""

from dataclasses import replace
from typing import cast

import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.timing import DigitalRecordTrigger


@pytest.fixture
def settings() -> CaptureSettings:
    """Request six two-channel records delivered in three two-record batches."""
    return CaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
        channel_indices=(0, 1),
        records_per_batch=2,
        batch_count=3,
        record_trigger=DigitalRecordTrigger(source="external"),
    )


def test_frames_can_cross_transfer_batch_boundaries(
    settings: CaptureSettings,
) -> None:
    """Group records into frames even when a frame boundary splits a batch.

    Batches contain records (0, 1), (2, 3), and (4, 5). With three records per
    frame, the second batch ends frame zero and begins frame one. Channel count
    and samples per record do not increase the number of acquired records.
    """
    plan = RecordFramePlan(
        capture_settings=settings,
        records_per_frame=3,
    )

    assert plan.record_count == 6
    assert plan.frame_count == 2
    assert tuple(plan.frame_start_indices) == (0, 3)
    assert plan.records_for_frame(0) == range(0, 3)
    assert plan.records_for_frame(1) == range(3, 6)
    assert 2 in plan.records_for_frame(0)
    assert 3 in plan.records_for_frame(1)


@pytest.mark.parametrize(
    ("records_per_batch", "batch_count"),
    [(1, 6), (2, 3), (3, 2), (6, 1)],
)
def test_transfer_batching_does_not_change_frame_membership(
    settings: CaptureSettings,
    records_per_batch: int,
    batch_count: int,
) -> None:
    """Keep the same frames when six records are divided into different batches."""
    request = replace(
        settings,
        records_per_batch=records_per_batch,
        batch_count=batch_count,
    )
    plan = RecordFramePlan(
        capture_settings=request,
        records_per_frame=3,
    )

    assert plan.frame_count == 2
    assert plan.frame_start_indices == range(0, 6, 3)
    assert plan.records_for_frame(1) == range(3, 6)


@pytest.mark.parametrize(
    ("records_per_frame", "frame_count"),
    [(1, 6), (2, 3), (3, 2), (6, 1)],
)
def test_frame_sizes_cover_the_capture_exactly(
    settings: CaptureSettings,
    records_per_frame: int,
    frame_count: int,
) -> None:
    """Support single-record frames through one frame containing the capture."""
    plan = RecordFramePlan(
        capture_settings=settings,
        records_per_frame=records_per_frame,
    )

    assert plan.frame_count == frame_count
    assert plan.frame_start_indices == range(0, 6, records_per_frame)
    assert plan.records_for_frame(frame_count - 1) == range(6 - records_per_frame, 6)


def test_sample_geometry_and_trigger_mode_do_not_change_grouping(
    settings: CaptureSettings,
) -> None:
    """Apply the same grouping to different ADC geometry and consecutive records."""
    request = replace(
        settings,
        sample_rate=units.SampleRate("200 MS/s"),
        record_length=8192,
        channel_indices=(1,),
        record_trigger=None,
    )
    plan = RecordFramePlan(
        capture_settings=request,
        records_per_frame=3,
    )

    assert plan.record_count == 6
    assert plan.frame_count == 2
    assert plan.frame_start_indices == range(0, 6, 3)


@pytest.mark.parametrize("records_per_frame", [4, 5, 7])
def test_partial_frames_are_rejected(
    settings: CaptureSettings,
    records_per_frame: int,
) -> None:
    """Reject both a partial final frame and a frame larger than the capture."""
    with pytest.raises(ValueError, match="whole number of frames"):
        RecordFramePlan(
            capture_settings=settings,
            records_per_frame=records_per_frame,
        )


def test_unlimited_capture_is_rejected(settings: CaptureSettings) -> None:
    """Require a finite record count to derive frame count and bounds."""
    with pytest.raises(ValueError, match="finite capture"):
        RecordFramePlan(
            capture_settings=replace(settings, batch_count=None),
            records_per_frame=3,
        )


def test_capture_settings_requires_the_model_type() -> None:
    """Reject an unrelated object before reading capture geometry."""
    with pytest.raises(TypeError, match=r"capture_settings.*CaptureSettings"):
        RecordFramePlan(
            capture_settings=cast(CaptureSettings, object()),
            records_per_frame=3,
        )


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        (3.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_frame_size_requires_a_positive_integer(
    settings: CaptureSettings,
    value: object,
    error_type: type[Exception],
) -> None:
    """Reject Boolean, float, zero, and negative frame sizes."""
    with pytest.raises(error_type, match="records_per_frame"):
        RecordFramePlan(
            capture_settings=settings,
            records_per_frame=cast(int, value),
        )


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        (True, TypeError),
        (0.0, TypeError),
        (-1, IndexError),
        (2, IndexError),
    ],
)
def test_frame_lookup_rejects_invalid_indices(
    settings: CaptureSettings,
    value: object,
    error_type: type[Exception],
) -> None:
    """Reject noninteger indices and indices before or beyond the plan."""
    plan = RecordFramePlan(
        capture_settings=settings,
        records_per_frame=3,
    )

    with pytest.raises(error_type, match="frame_index"):
        plan.records_for_frame(cast(int, value))
