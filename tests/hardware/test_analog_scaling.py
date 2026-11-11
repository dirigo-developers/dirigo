"""Test nominal analog scaling from vendor angular and command-voltage ranges."""

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.analog import angle_voltage_coefficients


@pytest.mark.parametrize(
    ("angle_bounds", "voltage_bounds", "inverted", "expected"),
    [
        (("0 deg", "26 deg"), ("0 V", "5 V"), False, [0.0, 2.5, 5.0]),
        (("0 deg", "26 deg"), ("1 V", "5 V"), False, [1.0, 3.0, 5.0]),
        (("-13 deg", "13 deg"), ("-10 V", "10 V"), False, [-10.0, 0.0, 10.0]),
        (("-13 deg", "13 deg"), ("-10 V", "10 V"), True, [10.0, 0.0, -10.0]),
        (("-10 deg", "10 deg"), ("-4 V", "6 V"), False, [-4.0, 1.0, 6.0]),
        (("-10 deg", "10 deg"), ("-4 V", "6 V"), True, [6.0, 1.0, -4.0]),
    ],
)
def test_full_scale_endpoints_and_midpoint(
    angle_bounds: tuple[str, str],
    voltage_bounds: tuple[str, str],
    inverted: bool,
    expected: list[float],
) -> None:
    """Map vendor endpoints and preserve electrical offsets and polarity."""
    angle_range = units.AngleRange(*angle_bounds)
    voltage_range = units.VoltageRange(*voltage_bounds)
    gain, offset = angle_voltage_coefficients(
        angle_range, voltage_range, inverted=inverted
    )
    angles = [
        float(angle_range.min),
        (float(angle_range.min) + float(angle_range.max)) / 2,
        float(angle_range.max),
    ]

    assert type(gain) is float
    assert type(offset) is float

    for angle, voltage in zip(angles, expected, strict=True):
        result = units.Voltage(angle * gain + offset)
        assert float(result) == pytest.approx(voltage)


@pytest.mark.parametrize("scan_extent", [3.25, 6.5, 13.0])
@pytest.mark.parametrize("inverted", [False, True])
def test_smaller_galvo_waveforms_keep_vendor_full_scale(
    scan_extent: float,
    inverted: bool,
) -> None:
    """Reduce scan extent without rescaling the waveform to full voltage."""
    gain, offset = angle_voltage_coefficients(
        units.AngleRange("-13 deg", "13 deg"),
        units.VoltageRange("-10 V", "10 V"),
        inverted=inverted,
    )
    angles = np.deg2rad(np.linspace(-scan_extent, scan_extent, 5, dtype=np.float64))
    voltages = np.empty_like(angles)

    np.multiply(angles, gain, out=voltages)
    np.add(voltages, offset, out=voltages)

    expected = np.linspace(-10 * scan_extent / 13, 10 * scan_extent / 13, 5)
    if inverted:
        expected = -expected

    np.testing.assert_allclose(voltages, expected, atol=1e-12)


@pytest.mark.parametrize(
    "value",
    [None, (-1, 1), units.VoltageRange("-1 V", "1 V")],
)
def test_requires_an_angle_range(value: object) -> None:
    """Reject unparsed ranges and ranges with the wrong physical dimension."""
    voltage_range = units.VoltageRange("-10 V", "10 V")

    with pytest.raises(TypeError, match="angle_range"):
        angle_voltage_coefficients(
            value,  # ty: ignore[invalid-argument-type]
            voltage_range,
        )


@pytest.mark.parametrize(
    "value",
    [None, {"min": "-10 V", "max": "10 V"}, units.AngleRange("-1 deg", "1 deg")],
)
def test_requires_a_voltage_range(value: object) -> None:
    """Require a typed command-voltage range."""
    angle_range = units.AngleRange("-13 deg", "13 deg")

    with pytest.raises(TypeError, match="command_voltage_range"):
        angle_voltage_coefficients(
            angle_range,
            value,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_inversion_requires_a_boolean(value: object) -> None:
    """Reject truthy substitutes for the polarity flag."""
    angle_range = units.AngleRange("-13 deg", "13 deg")
    voltage_range = units.VoltageRange("-10 V", "10 V")

    with pytest.raises(TypeError, match="inverted"):
        angle_voltage_coefficients(
            angle_range,
            voltage_range,
            inverted=value,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("angle_min", "angle_max", "voltage_min", "voltage_max"),
    [
        (float("-inf"), 1.0, -1.0, 1.0),
        (0.0, float("inf"), -1.0, 1.0),
        (-1.0, 1.0, float("-inf"), 1.0),
        (-1.0, 1.0, -1.0, float("inf")),
    ],
)
def test_full_scale_endpoints_must_be_finite(
    angle_min: float,
    angle_max: float,
    voltage_min: float,
    voltage_max: float,
) -> None:
    """Reject infinite endpoints accepted by the general range objects."""
    with pytest.raises(ValueError, match="endpoints"):
        angle_voltage_coefficients(
            units.AngleRange(angle_min, angle_max),
            units.VoltageRange(voltage_min, voltage_max),
        )


@pytest.mark.parametrize(
    ("angle_bounds", "voltage_bounds"),
    [
        ((0.0, 1e-308), (0.0, 1e308)),
        ((0.0, 1e308), (0.0, 1e-308)),
        ((1e308, 1.5e308), (0.0, 1e308)),
    ],
)
def test_coefficients_must_be_finite_with_nonzero_gain(
    angle_bounds: tuple[float, float],
    voltage_bounds: tuple[float, float],
) -> None:
    """Reject gain overflow, gain underflow, and offset overflow."""
    with pytest.raises(ValueError, match="coefficients"):
        angle_voltage_coefficients(
            units.AngleRange(*angle_bounds),
            units.VoltageRange(*voltage_bounds),
        )
