import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar, Never, Self, cast


class UnitQuantity(float):
    """A scalar stored in base units with checked quantity arithmetic.

    Numeric inputs use base units. Strings must name an allowed unit.
    Explicit float() conversion returns the base-unit value.
    """

    __slots__ = ()

    DIMENSIONAL_QUANTITY: ClassVar[tuple[str, str]] = ("1", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {"": 1.0}

    _dimension_registry: ClassVar[
        dict[tuple[str, str], type[UnitQuantity]]
    ] = {}

    _value_pattern = re.compile(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)\s*(.*?)\s*"
    )

    def __init_subclass__(cls, *, register: bool = True) -> None:
        super().__init_subclass__()

        numerator, denominator = cls.DIMENSIONAL_QUANTITY
        dimension = cls._normalize_dim(
            Counter(numerator),
            Counter(denominator),
        )
        cls.DIMENSIONAL_QUANTITY = dimension

        if register:
            if dimension in UnitQuantity._dimension_registry:
                raise ValueError(
                    f"A quantity is already registered for {dimension}."
                )
            UnitQuantity._dimension_registry[dimension] = cls

    def __new__(cls, quantity: str | float) -> Self:
        if isinstance(quantity, UnitQuantity):
            if quantity.DIMENSIONAL_QUANTITY != cls.DIMENSIONAL_QUANTITY:
                raise TypeError(
                    "Cannot convert between incompatible dimensions."
                )
            value = float(quantity)

        elif isinstance(quantity, str):
            match = cls._value_pattern.fullmatch(quantity)
            if match is None:
                raise ValueError(f"Invalid quantity: {quantity!r}.")

            number, unit = match.groups()
            unit = unit.replace("µ", "μ")

            if unit not in cls.ALLOWED_UNITS_AND_MULTIPLIERS:
                raise ValueError(
                    f"Invalid unit {unit!r} for {cls.__name__}."
                )

            value = (
                float(number)
                * cls.ALLOWED_UNITS_AND_MULTIPLIERS[unit]
            )

        elif isinstance(quantity, (int, float)):
            value = float(quantity)

        else:
            raise TypeError(
                "Expected a unit string or a numeric base-unit value."
            )

        return super().__new__(cls, value)

    @property
    def unit(self) -> str:
        """The base-unit label, not the original input unit."""
        return next(iter(self.ALLOWED_UNITS_AND_MULTIPLIERS))

    def _get_optimal_unit(self) -> tuple[float, str]:
        value = float(self)

        if value == 0 or not math.isfinite(value):
            return value, self.unit

        units = sorted(
            self.ALLOWED_UNITS_AND_MULTIPLIERS.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        for unit, multiplier in units:
            if abs(value) / multiplier >= 1:
                return value / multiplier, unit

        unit, multiplier = units[-1]
        return value / multiplier, unit

    def __str__(self) -> str:
        value, unit = self._get_optimal_unit()
        return f"{value:.3g} {unit}".rstrip()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    def with_unit(self, unit: str, sig_figs: int = 3) -> str:
        unit = unit.replace("µ", "μ")

        if unit not in self.ALLOWED_UNITS_AND_MULTIPLIERS:
            raise ValueError(
                f"Invalid unit {unit!r} for {type(self).__name__}."
            )

        value = float(self) / self.ALLOWED_UNITS_AND_MULTIPLIERS[unit]
        return f"{value:.{sig_figs}g} {unit}".rstrip()

    def _same_type(self, other: object) -> UnitQuantity:
        if (
            not isinstance(other, UnitQuantity)
            or type(other) is not type(self)
        ):
            raise TypeError(
                "Addition and subtraction require the same quantity type."
            )
        return other

    def __add__(self, other: object) -> Self:
        other = self._same_type(other)
        return type(self)(float(self) + float(other))

    __radd__ = __add__

    def __sub__(self, other: object) -> Self:
        other = self._same_type(other)
        return type(self)(float(self) - float(other))

    def __rsub__(self, other: object) -> Self:
        other = self._same_type(other)
        return type(self)(float(other) - float(self))

    def __mul__(self, other: object) -> UnitQuantity | float:
        if isinstance(other, UnitQuantity):
            n0, d0 = self.DIMENSIONAL_QUANTITY
            n1, d1 = other.DIMENSIONAL_QUANTITY

            return self._result(
                float(self) * float(other),
                Counter(n0) + Counter(n1),
                Counter(d0) + Counter(d1),
            )

        if isinstance(other, (int, float)):
            return type(self)(float(self) * other)

        raise TypeError("Multiplication requires a scalar or quantity.")

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> UnitQuantity | float:
        if isinstance(other, UnitQuantity):
            n0, d0 = self.DIMENSIONAL_QUANTITY
            n1, d1 = other.DIMENSIONAL_QUANTITY

            return self._result(
                float(self) / float(other),
                Counter(n0) + Counter(d1),
                Counter(d0) + Counter(n1),
            )

        if isinstance(other, (int, float)):
            return type(self)(float(self) / other)

        raise TypeError("Division requires a scalar or quantity.")

    def __rtruediv__(self, other: object) -> UnitQuantity | float:
        if isinstance(other, UnitQuantity):
            return other.__truediv__(self)

        if isinstance(other, (int, float)):
            numerator, denominator = self.DIMENSIONAL_QUANTITY

            return self._result(
                other / float(self),
                Counter(denominator),
                Counter(numerator),
            )

        raise TypeError("Division requires a scalar or quantity.")

    @staticmethod
    def _normalize_dim(
        numerator: Counter[str],
        denominator: Counter[str],
    ) -> tuple[str, str]:
        numerator.pop("1", None)
        denominator.pop("1", None)

        common = numerator & denominator
        numerator, denominator = (
            numerator - common,
            denominator - common,
        )

        return (
            "".join(
                symbol * count
                for symbol, count in sorted(numerator.items())
            ) or "1",
            "".join(
                symbol * count
                for symbol, count in sorted(denominator.items())
            ) or "1",
        )

    @staticmethod
    def _result(
        value: float,
        numerator: Counter[str],
        denominator: Counter[str],
    ) -> UnitQuantity | float:
        dimension = UnitQuantity._normalize_dim(
            numerator,
            denominator,
        )

        if dimension == ("1", "1"):
            return value

        target = UnitQuantity._dimension_registry.get(dimension)

        if target is None:
            raise TypeError(
                f"No quantity type registered for dimensions {dimension}."
            )

        return target(value)

    def __neg__(self) -> Self:
        return type(self)(-float(self))

    def __pos__(self) -> Self:
        return type(self)(float(self))

    def __abs__(self) -> Self:
        return type(self)(abs(float(self)))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, UnitQuantity)
            and self.DIMENSIONAL_QUANTITY == other.DIMENSIONAL_QUANTITY
            and float(self) == float(other)
        )

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = float.__hash__

    def _comparison_value(self, other: object) -> float:
        if not isinstance(other, UnitQuantity):
            raise TypeError("Ordering requires another quantity.")

        if self.DIMENSIONAL_QUANTITY != other.DIMENSIONAL_QUANTITY:
            raise TypeError(
                "Cannot order quantities with incompatible dimensions."
            )

        return float(other)

    def __lt__(self, other: object) -> bool:
        return float(self) < self._comparison_value(other)

    def __le__(self, other: object) -> bool:
        return float(self) <= self._comparison_value(other)

    def __gt__(self, other: object) -> bool:
        return float(self) > self._comparison_value(other)

    def __ge__(self, other: object) -> bool:
        return float(self) >= self._comparison_value(other)

    def _unsupported(self, *_operands: object) -> Never:
        raise TypeError(
            "Unsupported quantity operation; "
            "convert with float() explicitly."
        )

    __floordiv__ = __rfloordiv__ = _unsupported
    __mod__ = __rmod__ = _unsupported
    __divmod__ = __rdivmod__ = _unsupported
    __pow__ = __rpow__ = _unsupported


class Voltage(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("MLL", "TTTI")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "V": 1.0, "mV": 1e-3, "μV": 1e-6, "uV": 1e-6,
        "nV": 1e-9, "kV": 1e3,
    }


class Resistance(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("MLL", "TTTII")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "Ω": 1.0, "ohm": 1.0, "kΩ": 1e3, "kohm": 1e3,
        "MΩ": 1e6, "Mohm": 1e6, "GΩ": 1e9, "Gohm": 1e9,
    }


class Angle(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("Θ", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "rad": 1.0, "mrad": 1e-3, "deg": math.pi / 180,
    }


class Frequency(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("1", "T")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9,
    }


class SampleRate(Frequency, register=False):
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "S/s": 1.0, "kS/s": 1e3, "MS/s": 1e6, "GS/s": 1e9,
    }


class Time(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("T", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "s": 1.0, "ms": 1e-3, "μs": 1e-6, "us": 1e-6,
        "ns": 1e-9, "min": 60.0, "hr": 3600.0, "days": 86400.0,
    }


class Position(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("L", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m": 1.0, "cm": 1e-2, "mm": 1e-3, "μm": 1e-6,
        "um": 1e-6, "nm": 1e-9, "km": 1e3,
    }


class SpatialFrequency(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("1", "L")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m⁻¹": 1.0, "1/m": 1.0, "cm⁻¹": 1e2, "1/cm": 1e2,
        "mm⁻¹": 1e3, "1/mm": 1e3,
    }


class Velocity(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("L", "T")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m/s": 1.0, "cm/s": 1e-2, "mm/s": 1e-3,
        "μm/s": 1e-6, "um/s": 1e-6,
    }


class AngularVelocity(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("Θ", "T")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "rad/s": 1.0, "deg/s": math.pi / 180,
    }


class Acceleration(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("L", "TT")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m/s^2": 1.0, "mm/s^2": 1e-3,
        "μm/s^2": 1e-6, "um/s^2": 1e-6,
    }


class AngularAcceleration(UnitQuantity):
    DIMENSIONAL_QUANTITY = ("Θ", "TT")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "rad/s^2": 1.0, "deg/s^2": math.pi / 180,
    }


@dataclass(frozen=True, init=False, repr=False)
class RangeWithUnits[T: UnitQuantity]:
    """An immutable interval with inclusive membership and min < max.

    Numeric endpoint inputs use base units. A single string must use the
    symmetric form, such as "±5 V". Concrete subclasses select the endpoint
    quantity type through UNIT_QUANTITY_CLASS.
    """

    _min: T
    _max: T

    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = UnitQuantity

    def __init__(
        self,
        min: str | float,
        max: str | float | None = None,
    ) -> None:
        quantity_type = self.UNIT_QUANTITY_CLASS

        if max is None:
            if (
                not isinstance(min, str)
                or not min.strip().startswith("±")
            ):
                raise ValueError(
                    "Provide two endpoints or a symmetric ± quantity."
                )

            upper = cast(
                T,
                quantity_type(min.strip()[1:].strip()),
            )
            lower = -upper

        else:
            lower = cast(T, quantity_type(min))
            upper = cast(T, quantity_type(max))

        if not lower < upper:
            raise ValueError("Range min must be less than max.")

        object.__setattr__(self, "_min", lower)
        object.__setattr__(self, "_max", upper)

    @property
    def min(self) -> T:
        return self._min

    @property
    def max(self) -> T:
        return self._max

    @property
    def unit(self) -> str:
        return self._min.unit

    @property
    def range(self) -> T:
        return self._max - self._min

    def within_range(self, value: UnitQuantity) -> bool:
        """Check inclusive membership, rejecting incompatible dimensions."""
        if not isinstance(value, UnitQuantity):
            raise TypeError(
                "Range membership requires a UnitQuantity."
            )

        return self._min <= value <= self._max

    def __str__(self) -> str:
        return f"{self._min} to {self._max}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}"
            f"({self._min!r}, {self._max!r})"
        )


@dataclass(frozen=True, init=False, repr=False)
class VoltageRange(RangeWithUnits[Voltage]):
    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = Voltage


@dataclass(frozen=True, init=False, repr=False)
class AngleRange(RangeWithUnits[Angle]):
    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = Angle

    @property
    def min_degrees(self) -> float:
        return math.degrees(float(self.min))

    @property
    def max_degrees(self) -> float:
        return math.degrees(float(self.max))


@dataclass(frozen=True, init=False, repr=False)
class PositionRange(RangeWithUnits[Position]):
    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = Position


@dataclass(frozen=True, init=False, repr=False)
class TimeRange(RangeWithUnits[Time]):
    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = Time


@dataclass(frozen=True, init=False, repr=False)
class FrequencyRange(RangeWithUnits[Frequency]):
    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = Frequency


@dataclass(frozen=True, init=False, repr=False)
class SampleRateRange(RangeWithUnits[SampleRate]):
    UNIT_QUANTITY_CLASS: ClassVar[type[UnitQuantity]] = SampleRate


@dataclass(frozen=True, slots=True)
class IntRange:
    """Inclusive integer bounds with min strictly less than max."""

    min: int
    max: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min, int)
            or not isinstance(self.max, int)
            or isinstance(self.min, bool)
            or isinstance(self.max, bool)
        ):
            raise TypeError("Integer range endpoints must be integers.")

        if self.min >= self.max:
            raise ValueError("Range max must be greater than min.")

    @property
    def range(self) -> int:
        """Number of integers in the inclusive range."""
        return self.max - self.min + 1

    def within_range(self, value: int) -> bool:
        return self.min <= value <= self.max
