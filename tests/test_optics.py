"""Test optical calibration without hardware dependencies."""

from dataclasses import replace
from typing import Literal

import pytest

from dirigo import units
from dirigo.optics import LaserScanningOptics


def _optics() -> LaserScanningOptics:
    """Provide a simple calibration with distinct axis corrections."""
    return LaserScanningOptics(
        objective_focal_length=units.Length("10 mm"),
        relay_magnification=2.0,
        fast_axis_correction=1.1,
        slow_axis_correction=0.9,
    )


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        ("fast", 0.22),
        ("slow", 0.18),
    ],
)
def test_complete_extent_maps_to_complete_optical_span(
    axis: Literal["fast", "slow"],
    expected: float,
) -> None:
    """Apply relay scaling and the selected axis correction without halving."""
    angle = _optics().object_length_to_scan_angle(
        units.Length("1 mm"),
        axis=axis,
    )

    assert float(angle) == pytest.approx(expected)


def test_optical_conversion_round_trip() -> None:
    """Recover the original object extent using the inverse calibration."""
    optics = _optics()
    requested = units.Length("250 um")

    angle = optics.object_length_to_scan_angle(
        requested,
        axis="fast",
    )
    recovered = optics.scan_angle_to_object_length(
        angle,
        axis="fast",
    )

    assert float(recovered) == pytest.approx(float(requested))


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"objective_focal_length": units.Length(0)}, ValueError),
        ({"relay_magnification": 0.0}, ValueError),
        ({"relay_magnification": True}, TypeError),
        ({"fast_axis_correction": float("nan")}, ValueError),
        ({"slow_axis_correction": -1.0}, ValueError),
    ],
)
def test_invalid_calibration_is_rejected(
    changes: dict[str, object],
    error: type[Exception],
) -> None:
    """Reject calibration that cannot describe a positive finite scale."""
    with pytest.raises(error):
        replace(_optics(), **changes)
