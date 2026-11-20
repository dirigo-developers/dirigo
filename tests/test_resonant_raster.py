"""Test nominal resonant timing without hardware or vendor SDKs."""

import math
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.config import load_system
from dirigo.hardware.scanner import ResonantScannerConfig
from dirigo.optics import LaserScanningOptics
from dirigo.raster import (
    RasterFrameAcquisitionSpec,
    RasterFrameGeometry,
    resolve_raster_frame_geometry,
)
from dirigo.resonant_raster import plan_resonant_raster_frame


def _geometry(
    *,
    bidirectional: bool = True,
    rows: int = 64,
) -> RasterFrameGeometry:
    """Provide an exact grid using a known optical calibration."""
    spec = RasterFrameAcquisitionSpec(
        pixel_width=units.Length("1 um"),
        pixel_height=units.Length("1 um"),
        frame_width=units.Length("128 um"),
        frame_height=units.Length(rows * 1e-6),
        bidirectional=bidirectional,
        frames_per_acquisition=3,
    )
    optics = LaserScanningOptics(
        objective_focal_length=units.Length("10 mm"),
        relay_magnification=2.0,
    )

    return resolve_raster_frame_geometry(spec, optics)


def _scanner() -> ResonantScannerConfig:
    """Declare a fixed nominal scanner for deterministic timing checks."""
    return ResonantScannerConfig(
        nominal_frequency=units.Frequency("1 kHz"),
        max_amplitude=units.Angle("20 deg"),
        settling_time=units.Time("0 s"),
        frequency_tolerance=0.0,
    )


def _inputs() -> dict[str, Any]:
    """Supply explicit test timing and record-alignment constraints."""
    return {
        "sample_rate": units.SampleRate("1 MS/s"),
        "max_record_rate": units.SampleRate("1 kS/s"),
        "record_length_minimum": 64,
        "record_length_step": 16,
        "line_fill_fraction": 0.8,
        "trigger_phase": units.Angle("0 deg"),
        "capture_rearm_time": units.Time("10 us"),
        "slow_sample_rate": units.SampleRate("100 kS/s"),
        "slow_flyback_time": units.Time("2 ms"),
        "output_rearm_time": units.Time("100 us"),
    }


@pytest.mark.parametrize(
    ("bidirectional", "record_length", "image_records"),
    [
        (True, 960, 32),
        (False, 464, 64),
    ],
)
def test_capture_windows_and_record_alignment(
    bidirectional: bool,
    record_length: int,
    image_records: int,
) -> None:
    """Cover useful sweeps while respecting native length increments."""
    geometry = _geometry(bidirectional=bidirectional)
    plan = plan_resonant_raster_frame(
        geometry,
        _scanner(),
        **_inputs(),
    )

    assert plan.record_length == record_length
    assert plan.image_records == image_records
    assert plan.nominal_forward_samples == range(50, 450)

    if bidirectional:
        assert plan.nominal_reverse_samples == range(550, 950)
    else:
        assert plan.nominal_reverse_samples is None

    assert plan.record_length % 16 == 0
    assert plan.record_length / 1e6 + 10e-6 < 1e-3

    expected_amplitude = float(geometry.fast_angle_span) / math.sin(0.8 * math.pi / 2)
    assert float(plan.fast_amplitude) == pytest.approx(expected_amplitude)


def test_slow_waveform_hits_nominal_row_centers_and_returns() -> None:
    """Place row centers on the ramp and return to the initial command."""
    geometry = _geometry()
    plan = plan_resonant_raster_frame(
        geometry,
        _scanner(),
        **_inputs(),
    )

    samples = plan.slow_waveform.samples
    span = float(geometry.slow_angle_span)

    # At 1 kHz, forward/reverse centers occur at 0.25 and 0.75 ms.
    # At 100 kS/s AO, those are samples 25 and 75.
    assert samples[25] == pytest.approx(span * (-0.5 + 0.5 / geometry.rows))
    assert samples[75] == pytest.approx(span * (-0.5 + 1.5 / geometry.rows))

    assert samples[-1] == samples[0]
    assert np.all(samples >= -span / 2)
    assert np.all(samples <= span / 2)
    assert not samples.flags.writeable


def test_frame_period_includes_output_and_rearm_margin() -> None:
    """Reserve enough record ticks for the complete waveform and margin."""
    plan = plan_resonant_raster_frame(
        _geometry(),
        _scanner(),
        **_inputs(),
    )

    waveform_duration = plan.slow_waveform.samples.size / 100_000

    assert plan.records_per_frame == 35
    assert plan.records_per_frame / 1000 > waveform_duration + 100e-6
    assert plan.blank_records == 3
    assert plan.records_per_batch == 35
    assert plan.batch_count == 3
    assert float(plan.nominal_frame_rate) == pytest.approx(1000 / 35)


@pytest.mark.parametrize(
    "changes",
    [
        {"record_length_minimum": 1024},
        {"record_length_step": 128},
        {"capture_rearm_time": units.Time("100 us")},
    ],
)
def test_infeasible_capture_window_is_rejected(
    changes: dict[str, Any],
) -> None:
    """Do not shorten useful data to accommodate incompatible constraints."""
    inputs = _inputs()
    inputs.update(changes)

    with pytest.raises(ValueError, match="rearm"):
        plan_resonant_raster_frame(
            _geometry(),
            _scanner(),
            **inputs,
        )


def test_frequency_tolerance_can_make_capture_infeasible() -> None:
    """Cover the slowest sweep while allowing the fastest next trigger."""
    scanner = replace(_scanner(), frequency_tolerance=0.02)
    inputs = _inputs()
    inputs["max_record_rate"] = units.SampleRate("1.02 kS/s")

    with pytest.raises(ValueError, match="rearm"):
        plan_resonant_raster_frame(
            _geometry(),
            scanner,
            **inputs,
        )


def test_trigger_phase_requiring_pretrigger_data_is_rejected() -> None:
    """Zero-offset capture cannot recover a sweep that already began."""
    inputs = _inputs()
    inputs["trigger_phase"] = units.Angle("90 deg")

    with pytest.raises(ValueError, match="before the record trigger"):
        plan_resonant_raster_frame(
            _geometry(),
            _scanner(),
            **inputs,
        )


def test_bidirectional_geometry_requires_paired_rows() -> None:
    """Reject odd row counts rather than silently altering resolved geometry."""
    with pytest.raises(ValueError, match="row_multiple"):
        plan_resonant_raster_frame(
            _geometry(rows=63),
            _scanner(),
            **_inputs(),
        )


def test_required_amplitude_must_fit_scanner() -> None:
    """Reject an optical extent that exceeds available resonant amplitude."""
    scanner = replace(
        _scanner(),
        max_amplitude=units.Angle("1 deg"),
    )

    with pytest.raises(ValueError, match="amplitude"):
        plan_resonant_raster_frame(
            _geometry(),
            scanner,
            **_inputs(),
        )


def test_system_loads_resonant_raster_settings(
    tmp_path: Path,
) -> None:
    """Parse installed timing settings without constructing hardware."""
    path = tmp_path / "system.toml"
    path.write_text(
        """
[devices]

[resonant_raster]
line_fill_fraction = 0.8
trigger_phase = "0 deg"
capture_rearm_time = "10 us"
slow_sample_rate = "100 kS/s"
slow_flyback_time = "2 ms"
output_rearm_time = "100 us"
slow_settling_time = "5 ms"
stop_fast_scanner_on_close = false
""",
        encoding="utf-8",
    )

    with closing(
        load_system(
            path,
            drivers={},
            sync_backends={},
        )
    ) as system:
        settings = system.require_resonant_raster()

        assert settings.line_fill_fraction == 0.8
        assert settings.slow_sample_rate == units.SampleRate("100 kS/s")
        assert settings.slow_settling_time == units.Time("5 ms")
        assert settings.stop_fast_scanner_on_close is False
