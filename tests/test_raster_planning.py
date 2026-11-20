"""Test reusable fast-axis capture and the general frame-planning entry point."""

from dataclasses import replace

import pytest

from dirigo import units
from dirigo.hardware.scanner import GalvoScannerConfig, ResonantScannerConfig
from dirigo.optics import LaserScanningOptics
from dirigo.raster import RasterFrameAcquisitionSpec
from dirigo.raster_planning import plan_raster_frame
from dirigo.resonant_raster import (
    ResonantFastAxisPlan,
    ResonantRasterPlan,
    ResonantRasterSettings,
    plan_resonant_fast_axis,
)


def _scanner() -> ResonantScannerConfig:
    """Describe a deterministic scanner without constructing a device."""
    return ResonantScannerConfig(
        nominal_frequency=units.Frequency("1 kHz"),
        max_amplitude=units.Angle("20 deg"),
        settling_time=units.Time(0),
        frequency_tolerance=0.0,
    )


def _fast_plan(
    *,
    bidirectional: bool = True,
    record_length_minimum: int = 64,
    trigger_phase: units.Angle | None = None,
) -> ResonantFastAxisPlan:
    """Plan capture using only fast-axis settings and digitizer constraints."""
    return plan_resonant_fast_axis(
        units.Angle(0.0256),
        _scanner(),
        bidirectional=bidirectional,
        sample_rate=units.SampleRate("1 MS/s"),
        max_record_rate=units.SampleRate("1 kS/s"),
        record_length_minimum=record_length_minimum,
        record_length_step=16,
        line_fill_fraction=0.8,
        trigger_phase=units.Angle(0) if trigger_phase is None else trigger_phase,
        capture_rearm_time=units.Time("10 us"),
    )


def _frame_plan(
    *,
    bidirectional: bool = True,
    fast_scanner_config: ResonantScannerConfig | GalvoScannerConfig | None = None,
    slow_angle_range: units.AngleRange | None = None,
) -> ResonantRasterPlan:
    """Build a frame through the public planning entry point."""
    return plan_raster_frame(
        RasterFrameAcquisitionSpec(
            pixel_width=units.Length("1 um"),
            pixel_height=units.Length("1 um"),
            frame_width=units.Length("128 um"),
            frame_height=units.Length("64 um"),
            bidirectional=bidirectional,
            frames_per_acquisition=3,
        ),
        optics=LaserScanningOptics(
            objective_focal_length=units.Length("10 mm"),
            relay_magnification=2.0,
        ),
        fast_scanner_config=(
            _scanner() if fast_scanner_config is None else fast_scanner_config
        ),
        slow_scanner_config=GalvoScannerConfig(
            angle_range=(
                units.AngleRange("-20 deg", "20 deg")
                if slow_angle_range is None
                else slow_angle_range
            )
        ),
        settings=ResonantRasterSettings(
            line_fill_fraction=0.8,
            trigger_phase=units.Angle(0),
            capture_rearm_time=units.Time("10 us"),
            slow_sample_rate=units.SampleRate("100 kS/s"),
            slow_flyback_time=units.Time("2 ms"),
            output_rearm_time=units.Time("100 us"),
            slow_settling_time=units.Time(0),
        ),
        sample_rate=units.SampleRate("1 MS/s"),
        max_record_rate=units.SampleRate("1 kS/s"),
        record_length_minimum=64,
        record_length_step=16,
    )


@pytest.mark.parametrize(
    ("bidirectional", "length", "lines"),
    [(False, 464, 1), (True, 960, 2)],
)
def test_fast_axis_capture_does_not_require_a_frame(
    bidirectional: bool,
    length: int,
    lines: int,
) -> None:
    """Resolve capture windows and rearm margins without slow-axis geometry."""
    plan = _fast_plan(bidirectional=bidirectional)

    assert plan.record_length == length
    assert plan.lines_per_record == lines
    assert plan.nominal_forward_samples == range(50, 450)
    assert plan.nominal_reverse_samples == (range(550, 950) if bidirectional else None)
    assert float(plan.nominal_line_period) == pytest.approx(
        0.0005 if bidirectional else 0.001
    )
    assert float(plan.latest_useful_sample_end) == pytest.approx(
        0.00095 if bidirectional else 0.00045
    )


def test_fast_axis_rejects_insufficient_capture_rearm_time() -> None:
    """Reject hardware alignment constraints that make a record too long."""
    with pytest.raises(ValueError, match="rearm"):
        _fast_plan(record_length_minimum=1024)


def test_fast_axis_rejects_useful_data_before_the_trigger() -> None:
    """Keep unsupported negative capture offsets out of the resulting plan."""
    with pytest.raises(ValueError, match="before"):
        _fast_plan(trigger_phase=units.Angle("90 deg"))


@pytest.mark.parametrize(
    ("bidirectional", "image_records", "frame_records"),
    [(False, 64, 66), (True, 32, 35)],
)
def test_frame_planning_adds_slow_motion_to_fast_capture(
    bidirectional: bool,
    image_records: int,
    frame_records: int,
) -> None:
    """Preserve image grouping and flyback around the reusable capture plan."""
    plan = _frame_plan(bidirectional=bidirectional)

    assert isinstance(plan.fast_axis, ResonantFastAxisPlan)
    assert plan.geometry.columns == 128
    assert plan.geometry.rows == 64
    assert plan.image_records == image_records
    assert plan.records_per_frame == frame_records
    assert plan.record_length == (960 if bidirectional else 464)
    assert plan.batch_count == 3
    assert plan.slow_waveform.samples[0] == plan.slow_waveform.samples[-1]


def test_declared_frequency_drift_extends_the_capture_window() -> None:
    """Expose timing bounds that frame motion can use without scanner access."""
    scanner = replace(_scanner(), frequency_tolerance=0.003)
    plan = plan_resonant_fast_axis(
        units.Angle(0.0256),
        scanner,
        bidirectional=True,
        sample_rate=units.SampleRate("1 MS/s"),
        max_record_rate=units.SampleRate("1.003 kS/s"),
        record_length_minimum=64,
        record_length_step=16,
        line_fill_fraction=0.8,
        trigger_phase=units.Angle(0),
        capture_rearm_time=units.Time("10 us"),
    )

    assert float(plan.maximum_record_period) == pytest.approx(1 / 997)
    assert float(plan.latest_useful_sample_end) == pytest.approx(0.95 / 997)


def test_general_entry_point_rejects_unimplemented_fast_axis_models() -> None:
    """Reject galvo planning until its trajectory calculation is implemented."""
    scanner = GalvoScannerConfig(angle_range=units.AngleRange("-20 deg", "20 deg"))
    with pytest.raises(NotImplementedError, match="resonant"):
        _frame_plan(fast_scanner_config=scanner)


def test_general_entry_point_checks_slow_axis_bounds() -> None:
    """Reject a frame whose planned waveform exceeds the installed galvo range."""
    with pytest.raises(ValueError, match="angle_range"):
        _frame_plan(slow_angle_range=units.AngleRange("-0.1 deg", "0.1 deg"))
