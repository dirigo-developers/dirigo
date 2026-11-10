import math
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from dirigo.units import (
    Angle,
    AngleRange,
    Frequency,
    FrequencyRange,
    Length,
    LengthRange,
    Quantity,
    SampleRate,
    SampleRateRange,
    Time,
    TimeRange,
    Voltage,
    VoltageRange,
)


def test_endpoints_are_normalized_to_the_range_quantity_type() -> None:
    limits = VoltageRange(-5, "5000 mV")

    assert type(limits.min) is Voltage
    assert type(limits.max) is Voltage
    assert limits.min == Voltage("-5 V")
    assert limits.max == Voltage("5 V")
    assert limits.base_unit == "V"
    assert limits.span == Voltage("10 V")
    assert limits == VoltageRange(Voltage(-5), Voltage(5))


def test_all_concrete_range_types_preserve_quantity_types() -> None:
    cases = [
        (AngleRange("-90 deg", "90 deg"), Angle, math.pi),
        (VoltageRange("-5 V", "5 V"), Voltage, 10.0),
        (LengthRange("0 um", "1 mm"), Length, 0.001),
        (TimeRange("0 ms", "2 s"), Time, 2.0),
        (FrequencyRange("1 kHz", "2 kHz"), Frequency, 1000.0),
        (
            SampleRateRange("1 kS/s", "2 MS/s"),
            SampleRate,
            1_999_000.0,
        ),
    ]

    for limits, quantity_type, span in cases:
        assert type(limits.min) is quantity_type
        assert type(limits.max) is quantity_type
        assert type(limits.span) is quantity_type
        assert math.isclose(float(limits.span), span)


def test_symmetric_ranges_reuse_quantity_parsing() -> None:
    for text in ("±5 V", " ± 5V ", "±5e-3 kV", "±+5 V"):
        assert VoltageRange(text) == VoltageRange(-5, 5)

    for text in ("±2 us", "±2 μs", "±2 µs"):  # ruff: ignore[ambiguous-unicode-character-string]
        assert TimeRange(text) == TimeRange("-2 us", "2 us")

    assert AngleRange("±90 deg") == AngleRange("-90 deg", "90 deg")


@pytest.mark.parametrize(
    ("arg", "message"),
    [
        ("5 V", "two endpoints"),
        ("±", "Invalid quantity"),
        ("±0 V", "min must be less than max"),
        ("±-5 V", "min must be less than max"),
        ("±5 s", "Invalid unit"),
        ("±5 bananas", "Invalid unit"),
        (5, "two endpoints"),
    ],
)
def test_invalid_symmetric_ranges_are_rejected(arg, message: str) -> None:  # ruff: ignore[missing-type-function-argument]
    with pytest.raises(ValueError, match=message):
        VoltageRange(arg)


def test_endpoints_must_be_strictly_increasing() -> None:
    for lower, upper in (
        (1, 1),
        (2, 1),
        (math.nan, 1),
        (0, math.nan),
    ):
        with pytest.raises(ValueError, match="min must be less than max"):
            VoltageRange(lower, upper)


def test_endpoint_quantities_must_have_compatible_dimensions() -> None:
    with pytest.raises(TypeError):
        VoltageRange(Time(-1), Voltage(1))

    with pytest.raises(TypeError):
        VoltageRange(Voltage(-1), Time(1))


def test_membership_includes_endpoints() -> None:
    limits = VoltageRange("±5 V")

    assert Voltage("-5 V") in limits
    assert Voltage("250 mV") in limits
    assert Voltage("5 V") in limits
    assert Voltage("-6 V") not in limits
    assert Voltage("6 V") not in limits


def test_membership_requires_a_compatible_quantity() -> None:
    limits = VoltageRange("±5 V")

    with pytest.raises(TypeError):
        Time(1) in limits  # ruff: ignore[useless-comparison]

    for invalid in (1, "1 V", object(), False, True):
        with pytest.raises(TypeError):
            cast(Quantity, invalid) in limits  # ruff: ignore[useless-comparison]


def test_dimensionally_equivalent_quantities_are_accepted() -> None:
    limits = FrequencyRange(
        SampleRate("1 kS/s"),
        SampleRate("2 kS/s"),
    )

    assert type(limits.min) is Frequency
    assert type(limits.max) is Frequency
    assert SampleRate("1.5 kS/s") in limits


def test_span_calculation_preserves_precision() -> None:
    limits = VoltageRange("1.234567891 V", "2.345678912 V")

    assert math.isclose(
        float(limits.span),
        2.345678912 - 1.234567891,
        rel_tol=1e-15,
    )


def test_equality_and_hashing_use_range_type_and_normalized_endpoints() -> None:
    first = VoltageRange("-5 V", "5 V")
    second = VoltageRange("-5000 mV", "5000 mV")

    assert first == second
    assert hash(first) == hash(second)
    assert len({first, second}) == 1
    assert first != VoltageRange(-4, 5)
    assert first != object()
    assert FrequencyRange(1, 2) != SampleRateRange(1, 2)


def test_ranges_are_immutable() -> None:
    limits = VoltageRange("±5 V")

    for name in ("min", "max", "_min", "_max", "range", "unit"):
        with pytest.raises(FrozenInstanceError):
            setattr(limits, name, Voltage(0))

    with pytest.raises(FrozenInstanceError):
        delattr(limits, "_min")


def test_display_and_angle_convenience_properties() -> None:
    limits = VoltageRange("±5 V")

    assert str(limits) == "-5 V to 5 V"
    assert repr(limits) == "VoltageRange(Voltage('-5 V'), Voltage('5 V'))"

    angles = AngleRange("±90 deg")

    assert math.isclose(angles.min_degrees, -90.0)
    assert math.isclose(angles.max_degrees, 90.0)
