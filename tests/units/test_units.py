import math
from typing import cast

import pytest

from dirigo.units import (
    Acceleration,
    Angle,
    AngularAcceleration,
    AngularVelocity,
    Frequency,
    Position,
    Resistance,
    SampleRate,
    SpatialFrequency,
    Time,
    UnitQuantity,
    Velocity,
    Voltage,
)


def test_quantity_types_parse_into_base_units() -> None:
    cases = [
        (Voltage, "250 mV", 0.25, "V"),
        (Resistance, "2 kΩ", 2000.0, "Ω"),
        (Angle, "180 deg", math.pi, "rad"),
        (Frequency, "2 MHz", 2e6, "Hz"),
        (SampleRate, "2 MS/s", 2e6, "S/s"),
        (Time, "3 ms", 0.003, "s"),
        (Position, "7 µm", 7e-6, "m"),  # ruff: ignore[ambiguous-unicode-character-string]
        (SpatialFrequency, "2 1/mm", 2000.0, "m⁻¹"),
        (Velocity, "3 mm/s", 0.003, "m/s"),
        (AngularVelocity, "180 deg/s", math.pi, "rad/s"),
        (Acceleration, "1 um/s^2", 1e-6, "m/s^2"),
        (AngularAcceleration, "180 deg/s^2", math.pi, "rad/s^2"),
    ]

    for quantity_type, text, expected, unit in cases:
        quantity = quantity_type(text)

        assert isinstance(quantity, float)
        assert math.isclose(float(quantity), expected)
        assert quantity.unit == unit


def test_numeric_inputs_always_use_base_units() -> None:
    assert float(Voltage(2)) == 2.0
    assert float(Angle(math.pi)) == math.pi
    assert Frequency(SampleRate("2 kS/s")) == Frequency("2 kHz")


def test_parser_accepts_scientific_notation_and_micro_aliases() -> None:
    assert Voltage(" -1.5e-3V ") == Voltage("-1.5 mV")
    assert Time("1 us") == Time("1 μs") == Time("1 µs")  # ruff: ignore[ambiguous-unicode-character-string]


@pytest.mark.parametrize(
    ("text", "m"),
    [
        ("2", "Invalid unit"),
        ("2 bananas", "Invalid unit"),
        ("oops", "Invalid quantity"),
        ("1 V trailing", "Invalid unit"),
    ],
)
def test_constructor_rejects_invalid_values_and_dimensions(text: str, m: str) -> None:
    with pytest.raises(ValueError, match=m):
        Voltage(text)

    with pytest.raises(TypeError):
        Voltage(cast(float, object()))

    with pytest.raises(TypeError):
        Voltage(Time("1 s"))


def test_formatting_and_base_unit_property() -> None:
    voltage = Voltage("250 mV")

    assert voltage.unit == "V"
    assert str(voltage) == "250 mV"
    assert voltage.with_unit("V") == "0.25 V"
    assert str(Time(0)) == "0 s"
    assert repr(voltage) == "Voltage('250 mV')"

    with pytest.raises(ValueError, match="Invalid unit"):
        voltage.with_unit("s")


def test_same_type_addition_and_subtraction() -> None:
    assert Voltage("1 V") + Voltage("500 mV") == Voltage("1.5 V")
    assert Voltage("1 V") - Voltage("500 mV") == Voltage("500 mV")

    with pytest.raises(TypeError):
        Voltage(1) + Time(1)

    with pytest.raises(TypeError):
        Frequency(1) + SampleRate(1)


def test_scalar_and_unary_operations_preserve_concrete_type() -> None:
    rate = SampleRate("2 kS/s")
    results = [
        rate * 2,
        2 * rate,
        rate / 2,
        -rate,
        +rate,
        abs(-rate),
    ]

    assert all(type(result) is SampleRate for result in results)
    assert [float(result) for result in results] == [
        4000,
        4000,
        1000,
        -2000,
        2000,
        2000,
    ]


def test_registered_dimensional_arithmetic() -> None:
    cases = [
        (Position("2 mm") / Time("2 ms"), Velocity, 1.0),
        (Velocity("2 m/s") * Time("3 s"), Position, 6.0),
        (Angle("2 rad") / Time("2 s"), AngularVelocity, 1.0),
        (Acceleration("2 m/s^2") * Time("3 s"), Velocity, 6.0),
        (
            AngularAcceleration("2 rad/s^2") * Time("3 s"),
            AngularVelocity,
            6.0,
        ),
        (1 / Time("2 ms"), Frequency, 500.0),
        (1 / SampleRate("2 kS/s"), Time, 0.0005),
        (Voltage(2) * UnitQuantity(3), Voltage, 6.0),
        (Resistance(2) / UnitQuantity(2), Resistance, 1.0),
    ]

    for result, expected_type, expected_value in cases:
        assert type(result) is expected_type
        assert math.isclose(float(result), expected_value)


def test_dimensionless_results_are_plain_floats() -> None:
    results = [
        Frequency("2 kHz") * Time("1 ms"),
        Position("2 mm") * SpatialFrequency("1 1/mm"),
        Voltage("2 V") / Voltage("1 V"),
    ]

    assert all(type(result) is float for result in results)
    assert results == [2.0, 2.0, 2.0]


def test_unsupported_arithmetic_does_not_fall_back_to_float() -> None:
    value = Voltage(2)
    operations = [
        lambda: value + 1,
        lambda: 1 + value,
        lambda: value - 1,
        lambda: 1 - value,
        lambda: value * Time(1),
        lambda: value // 2,
        lambda: 2 // value,
        lambda: value % 2,
        lambda: 2 % value,
        lambda: value**2,
        lambda: 2**value,
        lambda: divmod(value, 2),
        lambda: divmod(2, value),
    ]

    for operation in operations:
        with pytest.raises(TypeError):
            operation()


def test_comparisons_check_dimensions() -> None:
    assert Voltage("1 V") == Voltage("1000 mV")
    assert Frequency("1 kHz") == SampleRate("1 kS/s")
    assert Voltage(1) != Time(1)
    assert Voltage(1) != 1
    assert Voltage(1) != 1
    assert Voltage(1) < Voltage(2)
    assert Frequency(1) <= SampleRate(2)
    assert Voltage(2) > Voltage(1)
    assert Voltage(2) >= Voltage(2)

    for operation in (
        lambda: Voltage(1) < Time(2),
        lambda: Voltage(1) < 2,
        lambda: Voltage(1) > 2,
    ):
        with pytest.raises(TypeError):
            operation()


def test_hashing_agrees_with_quantity_equality() -> None:
    assert hash(Frequency(1)) == hash(SampleRate(1))
    assert len({Voltage(1), Time(1), 1.0}) == 3


def test_division_by_zero_keeps_standard_exception() -> None:
    with pytest.raises(ZeroDivisionError):
        Voltage(1) / 0

    with pytest.raises(ZeroDivisionError):
        Voltage(1) / Voltage(0)

    with pytest.raises(ZeroDivisionError):
        1 / Time(0)


def test_duplicate_dimension_registration_is_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):

        class DuplicateVoltage(Voltage):
            pass

    # Failed registration must leave the existing canonical type intact.
    result = Voltage(2) * UnitQuantity(3)
    assert type(result) is Voltage


def test_formatting_uses_smallest_unit_for_tiny_values() -> None:
    assert str(Voltage("0.1 nV")) == "0.1 nV"
    assert str(Voltage("-0.1 nV")) == "-0.1 nV"


def test_reflected_subtraction_preserves_operand_order() -> None:
    result = Voltage("500 mV").__rsub__(Voltage("2 V"))

    assert type(result) is Voltage
    assert result == Voltage("1.5 V")


def test_multiplication_rejects_unsupported_operand() -> None:
    with pytest.raises(TypeError, match="Multiplication requires"):
        Voltage(1) * object()


def test_division_by_scalar_preserves_concrete_type() -> None:
    for scalar in (2, 2.0):
        result = SampleRate("4 kS/s") / scalar

        assert type(result) is SampleRate
        assert result == SampleRate("2 kS/s")


def test_reflected_division_accepts_quantity_numerator() -> None:
    result = Time("2 s").__rtruediv__(Position("6 m"))

    assert type(result) is Velocity
    assert result == Velocity("3 m/s")


def test_division_rejects_unsupported_operand() -> None:
    with pytest.raises(TypeError, match="Division requires"):
        Voltage(1) / object()


def test_reflected_division_rejects_unsupported_operand() -> None:
    with pytest.raises(TypeError, match="Division requires"):
        Voltage(1).__rtruediv__(object())


def test_less_than_or_equal_compares_base_unit_values() -> None:
    assert Voltage("500 mV") <= Voltage("1 V")
    assert Voltage("1000 mV") <= Voltage("1 V")
    assert not Voltage("2 V") <= Voltage("1 V")
