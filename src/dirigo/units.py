"""Scalar physical quantities and immutable inclusive ranges.

Quantities store numerical values in base units. Construct concrete quantity
types from numeric base-unit values or strings containing supported units.

Use float(quantity) to obtain a plain base-unit value, quantity.format_in()
to format a specific unit, and value in limits to test range membership.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar, Never, Self, cast


class Quantity(float):
    """A scalar physical quantity stored in base units.

    This base class supplies parsing, formatting, and dimension-aware
    arithmetic for concrete types such as Length, Voltage, and Time.
    Use concrete types when constructing physical quantities.

    Args:
        quantity: A numeric base-unit value, a string containing a number
            and a supported unit, or another Quantity with compatible
            dimensions. The original input unit is not retained.

    Attributes:
        DIMENSIONS: Numerator and denominator dimension symbols. Each
            character represents one factor; repetition represents a
            power, and "1" represents no dimensions. For example,
            ("L", "TT") represents length divided by time squared.
        ALLOWED_UNITS_AND_MULTIPLIERS: Supported unit labels mapped to
            multipliers that convert their values to base units. The
            first entry defines the base-unit label and must have a
            multiplier of 1.

    Raises:
        TypeError: If the input type is unsupported or an input Quantity
            has incompatible dimensions.
        ValueError: If a string cannot be parsed or names an unsupported
            unit.

    Notes:
        Numeric inputs are not restricted to positive or finite values.
        Explicit float() conversion returns the stored base-unit value.

        Addition and subtraction require identical concrete quantity
        types. Multiplication or division by a plain scalar preserves
        the concrete type. Operations between quantities combine their
        dimensions and return the registered result type, or a plain
        float when the dimensions cancel. An unregistered result
        dimension raises TypeError.

        Equality compares dimensions and numerical values; plain numbers
        are not equal to quantities. Ordering requires another Quantity
        with compatible dimensions. Unary plus, minus, and abs() preserve
        the concrete type. Floor division, modulo, divmod(), and
        exponentiation are unsupported.

        str() selects a display unit and uses three significant figures.
        repr() also uses this rounded display value and is not a
        lossless serialization.

        Subclasses define DIMENSIONS and ALLOWED_UNITS_AND_MULTIPLIERS.
        By default, each subclass registers as the arithmetic result type
        for its dimensions. Pass register=False when defining an alternate
        representation of dimensions that already have a registered type.
    """

    __slots__ = ()

    DIMENSIONS: ClassVar[tuple[str, str]] = ("1", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {"": 1.0}

    _dimension_registry: ClassVar[dict[tuple[str, str], type[Quantity]]] = {}

    _value_pattern = re.compile(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)\s*(.*?)\s*"
    )

    def __init_subclass__(cls, *, register: bool = True) -> None:
        """Normalize subclass dimensions and optionally register the type.

        Args:
            register: Whether arithmetic producing these dimensions should
                return instances of this subclass.

        Raises:
            ValueError: If registration is requested and another quantity
                type is already registered for the normalized dimensions.
        """
        super().__init_subclass__()

        numerator, denominator = cls.DIMENSIONS
        dimension = cls._normalize_dim(
            Counter(numerator),
            Counter(denominator),
        )
        cls.DIMENSIONS = dimension

        if register:
            if dimension in Quantity._dimension_registry:
                raise ValueError(f"A quantity is already registered for {dimension}.")
            Quantity._dimension_registry[dimension] = cls

    def __new__(cls, quantity: str | float) -> Self:
        """Construct a quantity using the input rules documented on Quantity."""

        if isinstance(quantity, Quantity):
            if quantity.DIMENSIONS != cls.DIMENSIONS:
                raise TypeError("Cannot convert between incompatible dimensions.")
            value = float(quantity)

        elif isinstance(quantity, str):
            match = cls._value_pattern.fullmatch(quantity)
            if match is None:
                raise ValueError(f"Invalid quantity: {quantity!r}.")

            number, unit = match.groups()
            unit = unit.replace("µ", "μ")  # ruff: ignore[ambiguous-unicode-character-string]

            if unit not in cls.ALLOWED_UNITS_AND_MULTIPLIERS:
                raise ValueError(f"Invalid unit {unit!r} for {cls.__name__}.")

            value = float(number) * cls.ALLOWED_UNITS_AND_MULTIPLIERS[unit]

        elif isinstance(quantity, (int, float)):
            value = float(quantity)

        else:
            raise TypeError("Expected a unit string or a numeric base-unit value.")

        return super().__new__(cls, value)

    @property
    def base_unit(self) -> str:
        """The base-unit label, independent of input or display units."""
        return next(iter(self.ALLOWED_UNITS_AND_MULTIPLIERS))

    def _get_optimal_unit(self) -> tuple[float, str]:
        value = float(self)

        if value == 0 or not math.isfinite(value):
            return value, self.base_unit

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
        """Format three significant figures in an automatically selected unit."""
        value, unit = self._get_optimal_unit()
        return f"{value:.3g} {unit}".rstrip()

    def __repr__(self) -> str:
        """Return the concrete class name with its rounded display value."""
        return f"{type(self).__name__}({str(self)!r})"

    def format_in(self, unit: str, sig_figs: int = 3) -> str:
        """Format the quantity in a specified supported unit.

        Args:
            unit: A label from ALLOWED_UNITS_AND_MULTIPLIERS.
            sig_figs: Significant figures used by general-format notation.

        Returns:
            A string containing the formatted value and unit. The stored
            value is unchanged.

        Raises:
            ValueError: If the unit is unsupported or the precision produces
                an invalid format specification.
        """
        unit = unit.replace("µ", "μ")  # ruff: ignore[ambiguous-unicode-character-string]

        if unit not in self.ALLOWED_UNITS_AND_MULTIPLIERS:
            raise ValueError(f"Invalid unit {unit!r} for {type(self).__name__}.")

        value = float(self) / self.ALLOWED_UNITS_AND_MULTIPLIERS[unit]
        return f"{value:.{sig_figs}g} {unit}".rstrip()

    def _same_type(self, other: object) -> Quantity:
        if not isinstance(other, Quantity) or type(other) is not type(self):
            raise TypeError("Addition and subtraction require the same quantity type.")
        return other

    def __add__(self, other: object) -> Self:
        """Add a quantity of the identical concrete type, preserving that type."""
        other = self._same_type(other)
        return type(self)(float(self) + float(other))

    __radd__ = __add__

    def __sub__(self, other: object) -> Self:
        """Subtract a quantity of the identical concrete type."""
        other = self._same_type(other)
        return type(self)(float(self) - float(other))

    def __rsub__(self, other: object) -> Self:
        """Subtract this quantity from another of the identical concrete type."""
        other = self._same_type(other)
        return type(self)(float(other) - float(self))

    def __mul__(self, other: object) -> Quantity | float:
        """Multiply by a scalar or combine dimensions with another quantity.

        Scalar multiplication preserves the concrete type. Quantity
        multiplication returns the registered result type, or a float
        for a dimensionless result. Unsupported operands or unregistered
        result dimensions raise TypeError.
        """
        if isinstance(other, Quantity):
            n0, d0 = self.DIMENSIONS
            n1, d1 = other.DIMENSIONS

            return self._result(
                float(self) * float(other),
                Counter(n0) + Counter(n1),
                Counter(d0) + Counter(d1),
            )

        if isinstance(other, (int, float)):
            return type(self)(float(self) * other)

        raise TypeError("Multiplication requires a scalar or quantity.")

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Quantity | float:
        """Divide by a scalar or combine dimensions with another quantity.

        Scalar division preserves the concrete type. Quantity division
        returns the registered result type, or a float for a dimensionless
        result. Unsupported operands or unregistered result dimensions
        raise TypeError. A zero divisor raises ZeroDivisionError.
        """
        if isinstance(other, Quantity):
            n0, d0 = self.DIMENSIONS
            n1, d1 = other.DIMENSIONS

            return self._result(
                float(self) / float(other),
                Counter(n0) + Counter(d1),
                Counter(d0) + Counter(n1),
            )

        if isinstance(other, (int, float)):
            return type(self)(float(self) / other)

        raise TypeError("Division requires a scalar or quantity.")

    def __rtruediv__(self, other: object) -> Quantity | float:
        """Divide a scalar or another quantity by this quantity.

        A scalar numerator produces reciprocal dimensions. The result
        follows the registration rules described by __truediv__.
        """
        if isinstance(other, Quantity):
            return other.__truediv__(self)

        if isinstance(other, (int, float)):
            numerator, denominator = self.DIMENSIONS

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
            "".join(symbol * count for symbol, count in sorted(numerator.items()))
            or "1",
            "".join(symbol * count for symbol, count in sorted(denominator.items()))
            or "1",
        )

    @staticmethod
    def _result(
        value: float,
        numerator: Counter[str],
        denominator: Counter[str],
    ) -> Quantity | float:
        dimension = Quantity._normalize_dim(
            numerator,
            denominator,
        )

        if dimension == ("1", "1"):
            return value

        target = Quantity._dimension_registry.get(dimension)

        if target is None:
            raise TypeError(f"No quantity type registered for dimensions {dimension}.")

        return target(value)

    def __neg__(self) -> Self:
        """Return the negated value with the same concrete quantity type."""
        return type(self)(-float(self))

    def __pos__(self) -> Self:
        """Return the value with the same concrete quantity type."""
        return type(self)(float(self))

    def __abs__(self) -> Self:
        """Return the absolute value with the same concrete quantity type."""
        return type(self)(abs(float(self)))

    def __eq__(self, other: object) -> bool:
        """Compare dimensions and values, returning False for non-quantities."""
        return (
            isinstance(other, Quantity)
            and self.DIMENSIONS == other.DIMENSIONS
            and float(self) == float(other)
        )

    def __ne__(self, other: object) -> bool:
        """Return the logical inverse of quantity equality."""
        return not self == other

    __hash__ = float.__hash__

    def _comparison_value(self, other: object) -> float:
        if not isinstance(other, Quantity):
            raise TypeError("Ordering requires another quantity.")

        if self.DIMENSIONS != other.DIMENSIONS:
            raise TypeError("Cannot order quantities with incompatible dimensions.")

        return float(other)

    def __lt__(self, other: object) -> bool:
        """Compare with a dimension-compatible quantity, otherwise raise TypeError."""
        return float(self) < self._comparison_value(other)

    def __le__(self, other: object) -> bool:
        """Compare with a dimension-compatible quantity, otherwise raise TypeError."""
        return float(self) <= self._comparison_value(other)

    def __gt__(self, other: object) -> bool:
        """Compare with a dimension-compatible quantity, otherwise raise TypeError."""
        return float(self) > self._comparison_value(other)

    def __ge__(self, other: object) -> bool:
        """Compare with a dimension-compatible quantity, otherwise raise TypeError."""
        return float(self) >= self._comparison_value(other)

    def _unsupported(self, *_operands: object) -> Never:
        """Reject an unsupported quantity operation with TypeError."""
        raise TypeError(
            "Unsupported quantity operation; convert with float() explicitly."
        )

    __floordiv__ = __rfloordiv__ = _unsupported
    __mod__ = __rmod__ = _unsupported
    __divmod__ = __rdivmod__ = _unsupported
    __pow__ = __rpow__ = _unsupported


class Voltage(Quantity):
    """Electric potential difference stored in volts (V)."""

    DIMENSIONS = ("MLL", "TTTI")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "V": 1.0,
        "mV": 1e-3,
        "μV": 1e-6,
        "uV": 1e-6,
        "nV": 1e-9,
        "kV": 1e3,
    }


class Resistance(Quantity):
    """Electrical resistance stored in ohms (Ω)."""

    DIMENSIONS = ("MLL", "TTTII")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "Ω": 1.0,
        "ohm": 1.0,
        "kΩ": 1e3,
        "kohm": 1e3,
        "MΩ": 1e6,
        "Mohm": 1e6,
        "GΩ": 1e9,
        "Gohm": 1e9,
    }


class Angle(Quantity):
    """Plane angle stored in radians (rad).

    Angle is treated as a separate dimension, represented by "Θ", rather
    than as a dimensionless quantity.
    """

    DIMENSIONS = ("Θ", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "rad": 1.0,
        "mrad": 1e-3,
        "deg": math.pi / 180,
    }


class Frequency(Quantity):
    """Frequency stored in hertz (Hz), with dimensions of inverse time."""

    DIMENSIONS = ("1", "T")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "Hz": 1.0,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
    }


class SampleRate(Frequency, register=False):
    """Sampling rate stored in samples per second (S/s).

    This type shares Frequency's dimensions but accepts sampling-rate
    unit labels. It does not replace Frequency as the registered
    arithmetic result type for inverse time.

    Scaling by a plain number preserves SampleRate. Arithmetic between
    quantities uses the dimension registry, so an inverse-time result
    is a Frequency.
    """

    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "S/s": 1.0,
        "kS/s": 1e3,
        "MS/s": 1e6,
        "GS/s": 1e9,
    }


class Time(Quantity):
    """Time stored in seconds (s), suitable for durations and relative times."""

    DIMENSIONS = ("T", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "s": 1.0,
        "ms": 1e-3,
        "μs": 1e-6,
        "us": 1e-6,
        "ns": 1e-9,
        "min": 60.0,
        "hr": 3600.0,
        "days": 86400.0,
    }


class Length(Quantity):
    """Length stored in meters (m).

    Suitable for sizes, wavelengths, displacements, and scalar position
    coordinates. Negative values are allowed.
    """

    DIMENSIONS = ("L", "1")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m": 1.0,
        "cm": 1e-2,
        "mm": 1e-3,
        "μm": 1e-6,
        "um": 1e-6,
        "nm": 1e-9,
        "km": 1e3,
    }


class SpatialFrequency(Quantity):
    """Inverse-length quantity stored in reciprocal meters (m⁻¹)."""

    DIMENSIONS = ("1", "L")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m⁻¹": 1.0,
        "1/m": 1.0,
        "cm⁻¹": 1e2,
        "1/cm": 1e2,
        "mm⁻¹": 1e3,
        "1/mm": 1e3,
    }


class Velocity(Quantity):
    """Signed scalar linear velocity stored in meters per second (m/s)."""

    DIMENSIONS = ("L", "T")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m/s": 1.0,
        "cm/s": 1e-2,
        "mm/s": 1e-3,
        "μm/s": 1e-6,
        "um/s": 1e-6,
    }


class AngularVelocity(Quantity):
    """Signed angular velocity stored in radians per second (rad/s)."""

    DIMENSIONS = ("Θ", "T")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "rad/s": 1.0,
        "deg/s": math.pi / 180,
    }


class Acceleration(Quantity):
    """Signed linear acceleration stored in meters per second squared."""

    DIMENSIONS = ("L", "TT")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "m/s^2": 1.0,
        "mm/s^2": 1e-3,
        "μm/s^2": 1e-6,
        "um/s^2": 1e-6,
    }


class AngularAcceleration(Quantity):
    """Signed angular acceleration stored in radians per second squared."""

    DIMENSIONS = ("Θ", "TT")
    ALLOWED_UNITS_AND_MULTIPLIERS: ClassVar[dict[str, float]] = {
        "rad/s^2": 1.0,
        "deg/s^2": math.pi / 180,
    }


@dataclass(frozen=True, init=False, repr=False)
class QuantityRange[T: Quantity]:
    """An immutable quantity interval with inclusive endpoints and min < max.

    Use concrete subclasses such as VoltageRange or LengthRange.
    Subclasses select their endpoint quantity type through QUANTITY_TYPE;
    the generic type parameter alone does not select it at runtime.

    Args:
        min: The lower endpoint, or a symmetric range string such as
            "±5 V" when max is omitted. Numeric endpoints use base units;
            string endpoints use the selected quantity type's parser.
        max: The upper endpoint. Omit only when min supplies a symmetric
            range string with a strictly positive magnitude.

    Attributes:
        QUANTITY_TYPE: The concrete quantity class used to construct both
            endpoints. Subclasses should match this to their generic
            endpoint type.

    Raises:
        TypeError: If an endpoint has an unsupported type or incompatible
            dimensions.
        ValueError: If endpoint parsing fails, the symmetric form is
            invalid, or the lower endpoint is not strictly below the upper.

    Notes:
        Membership uses value in limits and includes both endpoints.
        The value must be a Quantity with compatible dimensions; it is
        not automatically converted from a string or plain number.

    Examples:
        >>> limits = VoltageRange("±5 V")
        >>> Voltage("-5 V") in limits
        True
        >>> Voltage("6 V") in limits
        False
        >>> limits.span
        Voltage('10 V')
    """

    _min: T
    _max: T

    QUANTITY_TYPE: ClassVar[type[Quantity]] = Quantity

    def __init__(
        self,
        min: str | float,
        max: str | float | None = None,
    ) -> None:
        """Construct ordered endpoints using the QuantityRange input rules."""

        quantity_type = self.QUANTITY_TYPE

        if max is None:
            if not isinstance(min, str) or not min.strip().startswith("±"):
                raise ValueError("Provide two endpoints or a symmetric ± quantity.")

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
        """The inclusive lower endpoint as the selected quantity type."""

        return self._min

    @property
    def max(self) -> T:
        """The inclusive upper endpoint as the selected quantity type."""

        return self._max

    @property
    def base_unit(self) -> str:
        """The base-unit label shared by both endpoints."""

        return self._min.base_unit

    @property
    def span(self) -> T:
        """The difference max - min, preserving the endpoint quantity type."""

        return self._max - self._min

    def __contains__(self, value: Quantity) -> bool:
        """Return whether a quantity lies between the inclusive endpoints.

        Args:
            value: A Quantity with dimensions compatible with the endpoints.
                Its concrete type need not match the endpoint type.

        Raises:
            TypeError: If value is not a Quantity or has incompatible
                dimensions.
        """

        if not isinstance(value, Quantity) or isinstance(value, bool):
            raise TypeError("Range membership requires a Quantity.")

        return self._min <= value <= self._max

    def __str__(self) -> str:
        """Format both endpoints with automatic units and a 'to' separator."""

        return f"{self._min} to {self._max}"

    def __repr__(self) -> str:
        """Return the concrete range class with both endpoint representations."""

        return f"{type(self).__name__}({self._min!r}, {self._max!r})"


@dataclass(frozen=True, init=False, repr=False)
class VoltageRange(QuantityRange[Voltage]):
    """An immutable inclusive voltage interval with Voltage endpoints."""

    QUANTITY_TYPE: ClassVar[type[Quantity]] = Voltage


@dataclass(frozen=True, init=False, repr=False)
class AngleRange(QuantityRange[Angle]):
    """An immutable inclusive angle interval with Angle endpoints.

    Membership uses ordinary numerical ordering; angles are not wrapped
    modulo a full turn.
    """

    QUANTITY_TYPE: ClassVar[type[Quantity]] = Angle

    @property
    def min_degrees(self) -> float:
        """The inclusive lower endpoint in degrees as a plain float."""

        return math.degrees(float(self.min))

    @property
    def max_degrees(self) -> float:
        """The inclusive upper endpoint in degrees as a plain float."""

        return math.degrees(float(self.max))


@dataclass(frozen=True, init=False, repr=False)
class LengthRange(QuantityRange[Length]):
    """An immutable inclusive length interval with Length endpoints."""

    QUANTITY_TYPE: ClassVar[type[Quantity]] = Length


@dataclass(frozen=True, init=False, repr=False)
class TimeRange(QuantityRange[Time]):
    """An immutable inclusive time interval with Time endpoints."""

    QUANTITY_TYPE: ClassVar[type[Quantity]] = Time


@dataclass(frozen=True, init=False, repr=False)
class FrequencyRange(QuantityRange[Frequency]):
    """An immutable inclusive frequency interval with Frequency endpoints."""

    QUANTITY_TYPE: ClassVar[type[Quantity]] = Frequency


@dataclass(frozen=True, init=False, repr=False)
class SampleRateRange(QuantityRange[SampleRate]):
    """An immutable inclusive sampling-rate interval with SampleRate endpoints."""

    QUANTITY_TYPE: ClassVar[type[Quantity]] = SampleRate


@dataclass(frozen=True, slots=True)
class IntRange:
    """Immutable inclusive integer bounds with min strictly less than max.

    Args:
        min: The inclusive lower integer bound.
        max: The inclusive upper integer bound.

    Raises:
        TypeError: If either bound is not an int or is a bool.
        ValueError: If min is greater than or equal to max. Singleton
            ranges are not allowed.

    Examples:
        >>> bounds = IntRange(0, 255)
        >>> bounds.count
        256
        >>> 255 in bounds
        True
        >>> 256 in bounds
        False
    """

    min: int
    max: int

    def __post_init__(self) -> None:
        """Validate integer endpoint types and strictly increasing bounds."""

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
    def count(self) -> int:
        """The number of included integers, max - min + 1."""

        return self.max - self.min + 1

    def __contains__(self, value: int) -> bool:
        """Check inclusive membership, rejecting non-integers and booleans."""

        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("Integer range membership requires an integer.")

        return self.min <= value <= self.max
