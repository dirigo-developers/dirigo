"""Test typed configuration parsing independently of hardware construction."""

import tomllib
from dataclasses import dataclass, field

import pytest

from dirigo import units
from dirigo.hardware.config import parse_config
from dirigo.hardware.scanner import ResonantScannerConfig


@dataclass(frozen=True, kw_only=True)
class ExampleConfig:
    """Exercise supported scalar types and a quantity default factory."""

    channel: str
    enabled: bool = True
    count: int = 1
    gain: float = 1.0
    timeout: units.Time = field(default_factory=lambda: units.Time("100 ms"))


def test_parses_resonant_scanner_toml_without_mutating_input() -> None:
    """Convert quantity strings while preserving the original table."""
    raw = tomllib.loads(
        'nominal_frequency = "7.91 kHz"\n'
        'max_amplitude = "26 deg"\n'
        'settling_time = "100 ms"\n'
        "frequency_tolerance = 0.003\n"
    )
    original = raw.copy()

    result = parse_config(ResonantScannerConfig, raw)

    assert isinstance(result, ResonantScannerConfig)
    assert result.nominal_frequency == units.Frequency("7910 Hz")
    assert result.max_amplitude == units.Angle("26 deg")
    assert result.settling_time == units.Time("0.1 s")
    assert result.frequency_tolerance == 0.003
    assert raw == original


def test_preserves_defaults_and_runs_default_factories() -> None:
    """Let dataclass construction provide omitted values for each instance."""
    first = parse_config(ExampleConfig, {"channel": "Dev1/ao0"})
    second = parse_config(ExampleConfig, {"channel": "Dev1/ao0"})

    assert first.enabled is True
    assert first.count == 1
    assert first.gain == 1.0
    assert first.timeout == units.Time("100 ms")
    assert first.timeout is not second.timeout


def test_parses_scalar_values() -> None:
    """Accept exact scalar types and convert integer inputs to float fields."""
    result = parse_config(
        ExampleConfig,
        {"channel": "Dev1/ao0", "enabled": False, "count": 4, "gain": 2},
    )

    assert result.channel == "Dev1/ao0"
    assert result.enabled is False
    assert result.count == 4
    assert result.gain == 2.0
    assert type(result.gain) is float


@pytest.mark.parametrize(
    ("values", "name"),
    [({}, "channel"), ({"channel": "Dev1/ao0", "enabeld": True}, "enabeld")],
)
def test_rejects_missing_or_unknown_fields(
    values: dict[str, object],
    name: str,
) -> None:
    """Identify omitted required fields and spelling mistakes."""
    with pytest.raises(ValueError, match=name):
        parse_config(ExampleConfig, values)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("channel", 1),
        ("enabled", 1),
        ("enabled", "false"),
        ("count", True),
        ("count", 1.0),
        ("gain", True),
        ("gain", "2.0"),
        ("gain", units.Angle("1 deg")),
        ("timeout", 0.1),
    ],
)
def test_rejects_inappropriate_value_types(name: str, value: object) -> None:
    """Reject implicit coercions and identify the affected field."""
    with pytest.raises(TypeError, match=name):
        parse_config(ExampleConfig, {"channel": "Dev1/ao0", name: value})


@pytest.mark.parametrize("value", ["100", "1 V", "not a duration"])
def test_quantity_conversion_errors_identify_the_field(value: str) -> None:
    """Report missing units, incompatible units, and malformed quantities."""
    with pytest.raises(ValueError, match="timeout"):
        parse_config(ExampleConfig, {"channel": "Dev1/ao0", "timeout": value})


def test_model_validation_still_runs() -> None:
    """Let the model reject a parsed value that violates physical constraints."""
    with pytest.raises(ValueError, match="nominal_frequency"):
        parse_config(
            ResonantScannerConfig,
            {
                "nominal_frequency": "-1 Hz",
                "max_amplitude": "26 deg",
                "settling_time": "100 ms",
                "frequency_tolerance": 0.003,
            },
        )


def test_unsupported_field_annotations_are_rejected() -> None:
    """Fail explicitly for field types outside this increment's scope."""

    @dataclass
    class UnsupportedConfig:
        samples: list[int] = field(default_factory=list)

    with pytest.raises(TypeError, match="samples"):
        parse_config(UnsupportedConfig, {})


@pytest.mark.parametrize(
    "model_type",
    [int, ExampleConfig(channel="Dev1/ao0")],
    ids=["non-dataclass-type", "dataclass-instance"],
)
def test_requires_a_dataclass_type(model_type: object) -> None:
    """Reject ordinary types and dataclass instances."""
    with pytest.raises(TypeError, match="model_type must be a dataclass type"):
        parse_config(model_type, {})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "values",
    [None, [("channel", "Dev1/ao0")]],
    ids=["none", "list-of-pairs"],
)
def test_requires_a_mapping(values: object) -> None:
    """Reject inputs that are not mappings."""
    with pytest.raises(TypeError, match="values must be a mapping"):
        parse_config(ExampleConfig, values)  # ty: ignore[invalid-argument-type]


def test_requires_string_field_names() -> None:
    """Reject nonstring keys before checking configuration field names."""
    values: dict[object, object] = {
        "channel": "Dev1/ao0",
        123: "unexpected",
    }

    with pytest.raises(TypeError, match="Configuration field names must be strings"):
        parse_config(ExampleConfig, values)  # ty: ignore[invalid-argument-type]
