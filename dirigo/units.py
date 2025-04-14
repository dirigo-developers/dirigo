import math
import re
from typing import Dict
from dataclasses import dataclass

import numpy as np


class UnitQuantity(float):
    """
    Represents a single value with an associated unit and supports unit conversion.

    Supported mathematical operations (implemented with dunder methods, eg. 
    __add__, __truediv__, etc.):
    - Add two UnitQuantity of same class, returns new instance of that class
    - Subtract UnitQuantity of same class, returns new instance of that class
    - Negate UnitQuantity, returns new instance of that class
    - Multiplication of UnitQuantity and a generic float/int, returns new 
        instance of that class.
    - Multiplication of a UnitQuantity with another different UnitQuantity class:
        Permitted only if DIMENSIONAL_QUANTITY cancels, returns a generic float.
    - Division of UnitQuantity by a generic float/int, returns new instance of 
        that class.
    - Division of a UnitQuantity with a UnitQuantity of the same 
        DIMENSIONAL_QUANTITY, returns a generic float


    Attributes:
    - unit (str): The original unit of the quantity (e.g., "V", "mV").
    """
    DIMENSIONAL_QUANTITY: tuple[str] = ('1', '1') # Interpret as unity over unity (ie dimensionless)
    ALLOWED_UNITS_AND_MULTIPLIERS: Dict[str, float] = None  # Define allowed units and their factors in subclasses

    def __new__(cls, quantity: str | float):
        """
        Create a new instance of UnitQuantity.

        Args:
            quantity (str or float): Value with unit (e.g., "5 V", "100 mV") or a float (base unit).

        Returns:
            UnitQuantity: An instance of the class.

        Raises:
            ValueError: If the input format is invalid or the unit is not allowed.
        """
        if isinstance(quantity, str):
            value, unit = cls._parse_value_with_unit(quantity)
            if cls.ALLOWED_UNITS_AND_MULTIPLIERS and unit not in cls.ALLOWED_UNITS_AND_MULTIPLIERS:
                raise ValueError(f"Invalid unit '{unit}'. Allowed units are: {list(cls.ALLOWED_UNITS_AND_MULTIPLIERS.keys())}.")
            multiplier = cls.ALLOWED_UNITS_AND_MULTIPLIERS[unit]
            base_value = value * multiplier
        elif isinstance(quantity, (int, float)):
            base_value = float(quantity)
            unit = next(iter(cls.ALLOWED_UNITS_AND_MULTIPLIERS))  # Default to the base unit
            value = base_value
        else:
            raise TypeError("Input must be a string with units or a float representing the base unit.")

        # Use float's __new__ to initialize the value
        instance = super().__new__(cls, base_value)
        instance.unit = next(iter(cls.ALLOWED_UNITS_AND_MULTIPLIERS))
        return instance

    @staticmethod
    def _parse_value_with_unit(quantity: str) -> tuple[float, str]:
        """
        Parses a string containing a value and unit.

        Args:
            quantity (str): Input string (e.g., "5 V").

        Returns:
            tuple[float, str]: Parsed value as a float and the unit as a string.

        Raises:
            ValueError: If the input string is not in the expected format.
        """
        #pattern = r"^\s*([-+]?\d+(\.\d+)?)\s*([\w/]+)\s*$"
        pattern = r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*([\w/]+)\s*$" # 100% from ChatGPT
        match = re.match(pattern, quantity)

        if not match:
            raise ValueError(f"Invalid format for value with unit: '{quantity}'. Expected format: '<value> <unit>'.")

        value_str, unit = match.groups()
        return float(value_str), unit
    
    def _get_optimal_unit(self):
        """
        Determines the most appropriate unit for the current base value.

        Returns:
            tuple[float, str]: The converted value and the corresponding unit.
        """
        if not self.ALLOWED_UNITS_AND_MULTIPLIERS:
            raise ValueError("No allowed units defined for this class.")

        # Sort units by multiplier in descending order
        sorted_units = sorted(self.ALLOWED_UNITS_AND_MULTIPLIERS.items(), key=lambda x: x[1], reverse=True)
        for unit, multiplier in sorted_units:
            if abs(self / multiplier) >= 1:
                return self / multiplier, unit
        # Default to the smallest unit
        smallest_unit, smallest_multiplier = sorted_units[-1]
        return self / smallest_multiplier, smallest_unit

    def __str__(self) -> str:
        """Return a human-readable string with the optimal unit."""
        value, unit = self._get_optimal_unit()
        return f"{value:.3g} {unit}"  # Use 3 significant digits

    def __repr__(self):
        return str(self)
    
    def __neg__(self):
        """Return new instance of this class with negated value."""
        return type(self)(-float(self))
    
    def __add__(self, other):
        # Check if other is also a UnitQuantity, ensure both are same subclass (like Position + Position)
        if not isinstance(other, UnitQuantity):
            return NotImplemented  # Defer to other __radd__ or raise TypeError
        if type(self) != type(other):
            raise TypeError("Cannot add different UnitQuantity subclasses.")
                
        return type(self)(float(self) + float(other))
    
    def __sub__(self, other):
        # Check if other is also a UnitQuantity, ensure both are same subclass (like Position + Position)
        if not isinstance(other, UnitQuantity):
            return NotImplemented  # Defer to other __radd__ or raise TypeError
        if type(self) != type(other):
            raise TypeError("Cannot subtract different UnitQuantity subclasses.")

        return type(self)(float(self) - float(other))
        
    def __mul__(self, other):
        """
        Multiply this quantity by a dimensionless scalar (int or float),
        returning a new instance of the same subclass with the same unit.
        """
        if isinstance(other, UnitQuantity):
            if self.DIMENSIONAL_QUANTITY == tuple(reversed(other.DIMENSIONAL_QUANTITY)):
                # if the two UnitQuantity classes dimensions cancel, then allow it
                return float(self) * float(other)
            else:
                return NotImplemented # Don't allow mixed unit multiplication (yet)
        if isinstance(other, (int, float)):
            return type(self)(float(self) * other)
        return NotImplemented  # For any other type, we can't handle it

    def __rmul__(self, other):
        """
        Called if __mul__ is not implemented for the 'other' operand.
        In practice, for int/float, we want the same behavior as __mul__.
        """
        return self.__mul__(other)
    
    def __truediv__(self, other):
        """
        Divide this quantity by a dimensionless scalar (int or float),
        returning a new instance of the same subclass with the same unit.
        """
        if isinstance(other, UnitQuantity):
            if self.DIMENSIONAL_QUANTITY == other.DIMENSIONAL_QUANTITY:
                return float(self) / float(other) # allow division by SAME unit type
            else:
                return NotImplemented # Don't allow mixed unit division
        if isinstance(other, (int, float)):
            return type(self)(float(self) / other)
        return NotImplemented  # For any other type, we can't handle it



class Voltage(UnitQuantity):
    """
    Represents a voltage value with units (e.g., V, mV, kV, etc.).
    """
    DIMENSIONAL_QUANTITY = ('MLL', 'TTTI') # M L^2  T^-3 I^-1
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "V": 1,        # base unit: volts
        "mV": 1e-3,    # millivolts to volts
        "μV": 1e-6,    # microvolts to volts
        "uV": 1e-6,    # alias
        "nV": 1e-9,    # nanovolts to volts
        "kV": 1e3,     # kilovolts to volts
    }


class Resistance(UnitQuantity):
    """
    Represents a resistance value with units (e.g., Ω, kΩ, MΩ, etc.).
    """
    DIMENSIONAL_QUANTITY = ('MLL', 'TTTII') # M L^2  T^-3 I^-2
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "Ω": 1,        # base unit: ohms
        "ohm": 1,      # alias
        "kΩ": 1e3,     # kiloohms to ohms
        "kohm": 1e3,   # alias
        "MΩ": 1e6,     # megaohms to ohms
        "Mohm": 1e6,   # alias
        "GΩ": 1e9,     # gigaohms to ohms
        "Gohm": 1e9    # alias
    }


class Angle(UnitQuantity):
    """
    Represents an angular value with units (e.g. rad, deg).
    """
    DIMENSIONAL_QUANTITY = ('1', '1') # Angle has no dimension
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "rad": 1,               # base unit: radians
        "mrad": 1e-3,           # millradians to radians
        "deg": math.pi / 180    # degrees to radians
    }


class Frequency(UnitQuantity):
    """
    Represents a frequency value with units (e.g. Hz, kHz, MHz, or GHz).
    """
    DIMENSIONAL_QUANTITY = ('1', 'T')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "Hz": 1,        # base unit: hertz
        "kHz": 1e3,     # kilohertz to hertz
        "MHz": 1e6,     # megahertz to hertz
        "GHz": 1e9,     # gigahertz to hertz
        #"rpm": 1/60     # rotations per minute to hertz
    }


class SampleRate(UnitQuantity):
    """
    Represents a samples per second rate value with units (e.g. S/s, kS/s, MS/s, GS/s).

    Dimensionally equivalent to Frequency, but with slight modification to unit
    labels to benefit specific situations (e.g. digitizer rate).
    """
    DIMENSIONAL_QUANTITY = ('1', 'T')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "S/s": 1,       # base unit: samples per second
        "kS/s": 1e3,    # kilo samples per second to samples per second
        "MS/s": 1e6,    # mega samples per second to samples per second
        "GS/s": 1e9,    # giga kilo samples per second to samples per second
    }


class Time(UnitQuantity):
    """
    Represents a time value with units (e.g. s, ms, μs, ns, min, hr, days)
    """
    DIMENSIONAL_QUANTITY = ('T', '1')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "s": 1,         # base unit: seconds
        "ms": 1e-3,     # milliseconds to seconds
        "μs": 1e-6,     # microseconds to seconds
        "us": 1e-6,     # alias
        "ns": 1e-9,     # nanoseconds to seconds
        "min": 60.0,    # minutes to seconds
        "hr": 3600.0,   # hours to seconds
        "days": 86400.0 # days to seconds
    }


class Position(UnitQuantity):
    """
    Represents a spatial position value with units (e.g. m, cm, mm, μm, nm, km)
    """
    DIMENSIONAL_QUANTITY = ('L', '1')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "m": 1,         # base unit: meters
        "cm": 1e-2,     # centimeters to meters
        "mm": 1e-3,     # millimeters to meters
        "μm": 1e-6,     # micrometers to meters
        "um": 1e-6,     # alias
        "nm": 1e-9,     # nanometers to meters
        "km": 1e3       # kilometers to meters
    }


class Velocity(UnitQuantity):
    """
    Represents a velocity value with units (e.g. m/s, cm/s, mm/s, μm/s)
    """
    DIMENSIONAL_QUANTITY = ('L', 'T')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "m/s": 1,       # base unit: meters per second
        "cm/s": 1e-2,   # centimeters per second to meters per second
        "mm/s": 1e-3,   # millimeters per second to meters per second
        "μm/s": 1e-6,   # micrometers per second to meters per second
        "um/s": 1e-6    # alias
    }


class AngularVelocity(UnitQuantity):
    """
    Represents an angular velocity value with units (e.g. rad/s, deg/s)
    """
    DIMENSIONAL_QUANTITY = ('1', 'T')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "rad/s": 1,             # base unit: radians per second
        "deg/s": math.pi / 180  # degrees per second to radians per second
    }


class Acceleration(UnitQuantity):
    """
    Represents a velocity value with units (e.g. m/s^2, mm/s^2, μm/s^2)
    """
    DIMENSIONAL_QUANTITY = ('L', 'TT')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "m/s^2": 1,     # base unit: meters per second squared
        "mm/s^2": 1e-3, # millimeters per second squared to meters per second squared
        "μm/s^2": 1e-6, # micrometers per second squared to meters per second squared
        "um/s^2": 1e-6  # alias
    }


class AngularAcceleration(UnitQuantity):
    """
    Represents an angular velocity value with units (e.g. rad/s, deg/s)
    """
    DIMENSIONAL_QUANTITY = ('1', 'TT')
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "rad/s^2": 1,               # base unit: radians per second squared
        "deg/s^2": math.pi / 180    # degrees per second squared to radians per second squared
    }


class RangeWithUnits:
    """
    Represents a range with associated units and supports unit conversion.

    Attributes:
    - min (UnitQuantity): Minimum value as a UnitQuantity.
    - max (UnitQuantity): Maximum value as a UnitQuantity.
    - unit (str): The base unit of the range.
    """
    UNIT_QUANTITY_CLASS = UnitQuantity  # Define the UnitQuantity class to use in subclasses

    def __init__(self, min: str | float, max: str | float | None = None):
        """
        Initialize the range.

        Args:
            min_val (str | float): Could be one of:
                - A float or a string "X unit"
                - A single string in "±Xunit" form, e.g. "±5V"
            max_val (str | float | None): A float or string in "X unit" form, or None.

        Raises:
            ValueError: If the input format is invalid or the range is invalid
                       (min >= max).
        """
        # Case 1: The user gave us a single string that starts with ±
        if max is None and isinstance(min, str) and min.strip().startswith("±"):
            # Attempt to parse something like "±5V" or "± 5 V"
            min_str, max_str = self._parse_plus_minus_string(min)
            self._min = self.UNIT_QUANTITY_CLASS(min_str)
            self._max = self.UNIT_QUANTITY_CLASS(max_str)

        # Case 2: The user gave the original (min, max) form
        else:
            self._min = self.UNIT_QUANTITY_CLASS(min)
            self._max = self.UNIT_QUANTITY_CLASS(max)

        # Validate that min < max in the base unit
        if float(self._min) >= float(self._max):
            raise ValueError(f"Invalid range: min ({self._min}) must be less than max ({self._max}).")

        # Set the base unit as the first key from the allowed units and multipliers
        self.unit = next(iter(self.UNIT_QUANTITY_CLASS.ALLOWED_UNITS_AND_MULTIPLIERS))

    def _parse_plus_minus_string(self, pm_str: str) -> tuple[str, str]:
        """
        Given a string like '±5V' or '± 5 V',
        return ('-5 V', '+5 V') for usage in UnitQuantity.

        Raises:
            ValueError if it cannot parse the string properly.
        """
        # Example pattern to match ±, then optional spaces, a number, optional decimal,
        # optional spaces, then a unit (e.g. 'mV', 'V', etc.)
        pattern = r"^[±]\s*([+-]?\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$"
        match = re.match(pattern, pm_str.strip())
        if not match:
            raise ValueError(f"Unable to parse ± string '{pm_str}'.")

        numeric_part = match.group(1)  # e.g. '5' or '5.0'
        unit_part = match.group(2)     # e.g. 'V' or 'mV'

        # Construct the negative and positive strings for UnitQuantity
        min_str = f"-{numeric_part} {unit_part}"
        max_str = f"{numeric_part} {unit_part}"
        return (min_str, max_str)

    @property
    def min(self) -> UnitQuantity:
        """Get the minimum value in the base unit."""
        return self._min

    @property
    def max(self) -> UnitQuantity:
        """Get the maximum value in the base unit."""
        return self._max

    def within_range(self, value: UnitQuantity) -> bool:
        """
        Check whether a UnitQuantity is within the range.

        Args:
            value (UnitQuantity): The value to check.

        Returns:
            bool: True if the value is within the range, False otherwise.

        Raises:
            ValueError: If the value is not a UnitQuantity or if its units are incompatible.
        """
        if not isinstance(value, UnitQuantity):
            raise ValueError("Input must be a UnitQuantity.")

        # Normalize the input value to the base unit for comparison
        normalized_value = float(value)
        return float(self._min) <= normalized_value <= float(self._max)

    @property
    def range(self) -> UnitQuantity:
        """Get the range as the difference between max and min."""
        return self.UNIT_QUANTITY_CLASS(str(float(self.max) - float(self.min)) + " " + self.unit)

    def __str__(self) -> str:
        """Return a human-readable string representation of the range."""
        return f"{self._min} to {self._max}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._min}, {self._max})"
    
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.min == other.min and self.max == other.max
    
    def __hash__(self):
        return hash((self.min, self.max))


class AngleRange(RangeWithUnits):
    """
    Describes a range of angles with units (e.g., radians, degrees).
    """
    UNIT_QUANTITY_CLASS = Angle

    @property 
    def min_degrees(self) -> float:
        """Returns min in degrees optical."""
        return self.min * 180 / math.pi
    
    @property 
    def max_degrees(self) -> float:
        """Returns max in degrees optical."""
        return self.max * 180 / math.pi
    

class VoltageRange(RangeWithUnits):
    """
    Represents a range of voltages with units (e.g., V, mV, kV).
    """
    UNIT_QUANTITY_CLASS = Voltage
        

class PositionRange(RangeWithUnits):
    """
    Represents a position range with units (e.g. m, mm, μm, nm, km)
    """
    UNIT_QUANTITY_CLASS = Position


class TimeRange(RangeWithUnits):
    """
    Represents a time/duration range with units (e.g. s, ms, μs, ns, min, hr, days)
    """
    UNIT_QUANTITY_CLASS = Time
    

class FrequencyRange(RangeWithUnits):
    """
    Represents a position range with units (e.g. Hz, kHz, MHz, GHz)
    """
    UNIT_QUANTITY_CLASS = Frequency


class SampleRateRange(RangeWithUnits):
    """
    Represents a position range with units (e.g. S/s, kS/s, MS/s, GS/s)
    """
    UNIT_QUANTITY_CLASS = SampleRate


@dataclass
class ValueRange:
    min: int
    max: int

    @property
    def range(self) -> int:
        return 1 + self.max - self.min
    
    @property
    def recommended_dtype(self):
        """
        Returns the recommended Numpy data type to store data of this range.
        """
        if self.min < 0:
            # signed data types ...
            if self.max < 2**7:
                return np.int8
            elif self.max < 2**15:
                return np.int16
            elif self.max < 2**31:
                return np.int32
            elif self.max < 2**63:
                return np.int64
            else:
                return np.int128

        else: 
            # unsigned data types ..
            if self.min == False and self.max == True:
                return np.bool
            elif self.max < 2**8:
                return np.uint8
            elif self.max < 2**16:
                return np.uint16
            elif self.max < 2**32:
                return np.uint32
            elif self.max < 2**64:
                return np.uint64
            else:
                return np.uint128