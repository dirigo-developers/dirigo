"""Test resonant scanner configuration without accessing hardware."""

from dataclasses import FrozenInstanceError, fields, replace
from typing import get_type_hints

import pytest

from dirigo import units
from dirigo.hardware.scanner import ResonantScannerConfig


@pytest.fixture
def config() -> ResonantScannerConfig:
    """Provide representative resonant scanner installation parameters."""
    return ResonantScannerConfig(
        nominal_frequency=units.Frequency("7.91 kHz"),
        max_amplitude=units.Angle("26 deg"),
        frequency_tolerance=0.003,
    )


def test_preserves_quantities_and_defaults(
    config: ResonantScannerConfig,
) -> None:
    """Keep physical quantities and default to 1.0 s settling time."""
    assert config.nominal_frequency == units.Frequency("7910 Hz")
    assert config.max_amplitude == units.Angle("26 deg")
    assert config.settling_time == units.Time("1.0 s")
    assert config.frequency_tolerance == 0.003


def test_boundary_values_are_valid(config: ResonantScannerConfig) -> None:
    """Allow zero settling time and zero declared frequency tolerance."""
    updated = replace(
        config,
        settling_time=units.Time("0 s"),
        frequency_tolerance=0,
    )

    assert updated.settling_time == units.Time("0 s")
    assert updated.frequency_tolerance == 0


@pytest.mark.parametrize(
    ("name", "quantity_type"),
    [
        ("nominal_frequency", units.Frequency),
        ("max_amplitude", units.Angle),
        ("settling_time", units.Time),
    ],
)
def test_quantity_fields_validate_types_and_finiteness(
    config: ResonantScannerConfig,
    name: str,
    quantity_type: type[units.Quantity],
) -> None:
    """Reject unparsed values, wrong dimensions, and nonfinite quantities."""
    for value in (True, 1.0, "1", units.Voltage("1 V")):
        with pytest.raises(TypeError, match=name):
            replace(config, **{name: value})

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match=name):
            replace(config, **{name: quantity_type(value)})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("nominal_frequency", units.Frequency("0 Hz")),
        ("nominal_frequency", units.Frequency("-1 Hz")),
        ("max_amplitude", units.Angle("0 deg")),
        ("max_amplitude", units.Angle("-1 deg")),
        ("settling_time", units.Time("-1 ms")),
        ("frequency_tolerance", -0.001),
        ("frequency_tolerance", 1.0),
        ("frequency_tolerance", float("nan")),
        ("frequency_tolerance", float("inf")),
    ],
)
def test_invalid_values_are_rejected(
    config: ResonantScannerConfig,
    name: str,
    value: object,
) -> None:
    """Enforce physical limits."""
    with pytest.raises(ValueError, match=name):
        replace(config, **{name: value})


@pytest.mark.parametrize("value", [True, "0.003", units.Angle("1 deg")])
def test_tolerance_requires_a_plain_number(
    config: ResonantScannerConfig,
    value: object,
) -> None:
    """Reject booleans, strings, and quantities as fractional tolerances."""
    with pytest.raises(TypeError, match="frequency_tolerance"):
        replace(config, frequency_tolerance=value)


def test_config_is_frozen(config: ResonantScannerConfig) -> None:
    """Prevent mutation that could bypass initialization validation."""
    with pytest.raises(FrozenInstanceError):
        config.max_amplitude = units.Angle("-1 deg")  # ty: ignore[invalid-assignment]


def test_fields_are_inspectable() -> None:
    """Expose descriptions and valid quantity display units for forms."""
    hints = get_type_hints(ResonantScannerConfig)

    for definition in fields(ResonantScannerConfig):
        assert definition.metadata["description"]

        if "display_unit" in definition.metadata:
            quantity_type = hints[definition.name]
            assert (
                definition.metadata["display_unit"]
                in quantity_type.ALLOWED_UNITS_AND_MULTIPLIERS
            )
