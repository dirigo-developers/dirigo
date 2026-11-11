"""Test quantity-range parsing using explicit unit-bearing endpoints."""

from copy import deepcopy
from dataclasses import dataclass, field

import pytest

from dirigo import units
from dirigo.hardware.config import parse_config


@dataclass(frozen=True, kw_only=True)
class RangeConfig:
    """Declare a required voltage range and a default angular range."""

    command_voltage_range: units.VoltageRange
    angle_range: units.AngleRange = field(
        default_factory=lambda: units.AngleRange("-13 deg", "13 deg")
    )


def test_parses_concrete_range_types_without_mutating_input() -> None:
    """Use each declared range's quantity type to interpret endpoint units."""
    values = {
        "command_voltage_range": {"min": "-500 mV", "max": "2 V"},
        "angle_range": {"min": "-1 deg", "max": "2 deg"},
    }
    original = deepcopy(values)

    config = parse_config(RangeConfig, values)

    assert config.command_voltage_range == units.VoltageRange("-0.5 V", "2 V")
    assert config.angle_range == units.AngleRange("-1 deg", "2 deg")
    assert values == original


def test_omitted_range_uses_its_declared_default_factory() -> None:
    """Keep default handling unchanged for quantity-range fields."""
    config = parse_config(
        RangeConfig,
        {"command_voltage_range": {"min": "0 V", "max": "5 V"}},
    )

    assert config.angle_range == units.AngleRange("-13 deg", "13 deg")


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (None, TypeError),
        ("±5 V", TypeError),
        ([-5, 5], TypeError),
        ({}, ValueError),
        ({"min": "0 V"}, ValueError),
        ({"max": "5 V"}, ValueError),
        ({"min": "0 V", "max": "5 V", "unit": "V"}, ValueError),
        ({"min": 0, "max": "5 V"}, TypeError),
        ({"min": "0 V", "max": 5}, TypeError),
        ({"min": "0", "max": "5 V"}, ValueError),
        ({"min": "0 V", "max": "5 deg"}, ValueError),
        ({"min": "5 V", "max": "0 V"}, ValueError),
        ({"min": "5 V", "max": "5 V"}, ValueError),
    ],
)
def test_invalid_range_values_identify_the_configuration_field(
    value: object,
    error: type[Exception],
) -> None:
    """Reject malformed tables, missing units, and invalid bounds."""
    with pytest.raises(error, match="command_voltage_range"):
        parse_config(RangeConfig, {"command_voltage_range": value})
