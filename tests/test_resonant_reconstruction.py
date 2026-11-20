"""Test nominal fast-axis reconstruction with synthetic digitizer records."""

from dataclasses import replace

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.scanner import ResonantScannerConfig
from dirigo.resonant_raster import (
    ResonantFastAxisPlan,
    ResonantLineReconstructor,
    plan_resonant_fast_axis,
)


def _plan(
    *, bidirectional: bool = False, trigger_phase: str = "0 deg"
) -> ResonantFastAxisPlan:
    scanner = ResonantScannerConfig(
        nominal_frequency=units.Frequency("1 kHz"),
        max_amplitude=units.Angle("20 deg"),
        frequency_tolerance=0.0,
    )
    return plan_resonant_fast_axis(
        units.Angle("5 deg"),
        scanner,
        bidirectional=bidirectional,
        sample_rate=units.SampleRate("1 MS/s"),
        max_record_rate=units.SampleRate("1 kS/s"),
        record_length_minimum=64,
        record_length_step=16,
        line_fill_fraction=0.8,
        trigger_phase=units.Angle(trigger_phase),
        capture_rearm_time=units.Time("10 us"),
    )


@pytest.mark.parametrize(
    ("bidirectional", "trigger_phase", "expected"),
    [
        # Forward edges at ADC indices 50, 172, 250, 329, 450.
        (False, "0 deg", [[110.5, 210.5, 289.0, 389.0]]),
        # 9 degrees advances both sweeps by 25 samples. The reverse sweep
        # is acquired 500 samples after the forward sweep, then flipped.
        (True, "9 deg", [[85.5, 185.5, 264.0, 364.0], [864.0, 764.0, 685.5, 585.5]]),
    ],
)
def test_reconstructs_uniform_spatial_pixels(
    bidirectional: bool, trigger_phase: str, expected: list[list[float]]
) -> None:
    """Check exact-edge rounding, reverse orientation, and row/channel order."""
    plan = _plan(bidirectional=bidirectional, trigger_phase=trigger_phase)
    reconstructor = ResonantLineReconstructor(plan, columns=4)
    ramp = np.arange(plan.record_length, dtype=np.int16)
    offsets = np.array([[0, -1000], [2000, 1000]], dtype=np.int16)
    samples = ramp[None, :, None] + offsets[:, None, :]
    samples.setflags(write=False)

    lines = reconstructor.reconstruct(samples)

    sweeps = len(expected)
    assert lines.shape == (2 * sweeps, 4, 2)
    assert lines.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(lines[:sweeps, :, 0], expected)
    np.testing.assert_array_equal(lines[sweeps:, :, 0], np.asarray(expected) + 2000)
    np.testing.assert_array_equal(lines[:, :, 1], lines[:, :, 0] - 1000)


def test_rejects_empty_pixel_bins_near_the_sweep_center() -> None:
    plan = _plan()
    # The total number of samples is sufficient, but central bins are narrower.
    with pytest.raises(ValueError, match="empty pixel bin"):
        ResonantLineReconstructor(plan, columns=len(plan.nominal_forward_samples))


def test_rejects_a_sweep_outside_the_record() -> None:
    plan = _plan()
    shortened = replace(plan, record_length=plan.nominal_forward_samples.stop - 1)
    with pytest.raises(ValueError, match="fit within the record"):
        ResonantLineReconstructor(shortened, columns=4)


def test_rejects_samples_with_the_wrong_record_length() -> None:
    plan = _plan()
    reconstructor = ResonantLineReconstructor(plan, columns=4)
    samples = np.zeros((1, plan.record_length - 1, 1), dtype=np.uint16)

    with pytest.raises(ValueError, match="record_length"):
        reconstructor.reconstruct(samples)
