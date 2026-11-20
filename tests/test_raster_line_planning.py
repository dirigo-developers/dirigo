"""Test finite repeated-line planning and shared resonant settings."""

from dataclasses import replace
from pathlib import Path

import pytest

from dirigo import units
from dirigo.hardware.config import load_system
from dirigo.hardware.scanner import ResonantScannerConfig
from dirigo.optics import LaserScanningOptics
from dirigo.raster import RasterLineAcquisitionSpec
from dirigo.raster_planning import plan_raster_line
from dirigo.resonant_raster import (
    ResonantLinePlan,
    ResonantRasterSettings,
    ResonantScanSettings,
)
from dirigo.system import System


def _spec() -> RasterLineAcquisitionSpec:
    """Request twelve useful lines in batches of four."""
    return RasterLineAcquisitionSpec(
        pixel_width=units.Length("1 um"),
        line_width=units.Length("128 um"),
        bidirectional=True,
        lines_per_acquisition=12,
        lines_per_batch=4,
    )


def _settings() -> ResonantScanSettings:
    """Specify resonant capture without slow-axis timing."""
    return ResonantScanSettings(
        line_fill_fraction=0.8,
        trigger_phase=units.Angle(0),
        capture_rearm_time=units.Time("10 us"),
    )


def _plan(spec: RasterLineAcquisitionSpec) -> ResonantLinePlan:
    """Use deterministic scanner timing and digitizer alignment constraints."""
    return plan_raster_line(
        spec,
        optics=LaserScanningOptics(
            objective_focal_length=units.Length("10 mm"),
            relay_magnification=2.0,
        ),
        fast_scanner_config=ResonantScannerConfig(
            nominal_frequency=units.Frequency("1 kHz"),
            max_amplitude=units.Angle("20 deg"),
            settling_time=units.Time(0),
            frequency_tolerance=0.0,
        ),
        settings=_settings(),
        sample_rate=units.SampleRate("1 MS/s"),
        record_length_minimum=64,
        record_length_step=16,
    )


@pytest.mark.parametrize(
    ("bidirectional", "records_per_batch", "record_length"),
    [(False, 4, 464), (True, 2, 960)],
)
def test_line_plan_counts_useful_lines_exactly(
    bidirectional: bool,
    records_per_batch: int,
    record_length: int,
) -> None:
    """Convert useful lines into complete records without adding frame flyback."""
    plan = _plan(replace(_spec(), bidirectional=bidirectional))

    assert plan.columns == 128
    assert float(plan.pixel_width) == pytest.approx(1e-6)
    assert plan.records_per_batch == records_per_batch
    assert plan.batch_count == 3
    assert plan.fast_axis.record_length == record_length
    assert (
        plan.records_per_batch * plan.batch_count * plan.fast_axis.lines_per_record
    ) == 12


def test_line_plan_preserves_extent_when_adjusting_pixel_spacing() -> None:
    """Resolve an integer pixel count within the configured spacing tolerance."""
    plan = _plan(replace(_spec(), line_width=units.Length("128.4 um")))

    assert plan.columns == 128
    assert float(plan.pixel_width) == pytest.approx(128.4e-6 / 128)


def test_line_plan_rejects_incomplete_bidirectional_pairs() -> None:
    """Avoid silently discarding one sweep from a bidirectional record."""
    with pytest.raises(ValueError, match="pairs"):
        _plan(replace(_spec(), lines_per_batch=3))


def test_line_plan_rejects_a_partial_final_batch() -> None:
    """Keep finite delivery exact until partial batches are supported."""
    with pytest.raises(ValueError, match="multiple"):
        _plan(replace(_spec(), lines_per_acquisition=10))


def test_frame_settings_supply_shared_scan_settings() -> None:
    """Let existing frame configurations support line acquisition unchanged."""
    settings = ResonantRasterSettings(
        line_fill_fraction=0.8,
        trigger_phase=units.Angle(0),
        capture_rearm_time=units.Time("10 us"),
        slow_sample_rate=units.SampleRate("100 kS/s"),
        slow_flyback_time=units.Time("2 ms"),
        output_rearm_time=units.Time("100 us"),
        slow_settling_time=units.Time(0),
    )
    system = System({}, resonant_raster=settings)
    try:
        assert system.require_resonant_scan() is settings
    finally:
        system.close()


def test_line_only_configuration_has_no_frame_settings(tmp_path: Path) -> None:
    """Load common settings without declaring slow-axis timing."""
    path = tmp_path / "system.toml"
    path.write_text(
        """
[devices]

[resonant_scan]
line_fill_fraction = 0.8
trigger_phase = "0 deg"
capture_rearm_time = "10 us"
""",
        encoding="utf-8",
    )
    system = load_system(path, drivers={}, sync_backends={})
    try:
        assert isinstance(system.require_resonant_scan(), ResonantScanSettings)
        with pytest.raises(KeyError, match="resonant_raster"):
            system.require_resonant_raster()
    finally:
        system.close()
