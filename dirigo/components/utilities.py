import math
import re
from typing import Dict


class UnitQuantity(float):
    """
    Represents a single value with an associated unit and supports unit conversion.

    Attributes:
    - unit (str): The original unit of the quantity (e.g., "V", "mV").
    """
    ALLOWED_UNITS_AND_MULTIPLIERS: Dict[str, float] = None  # Define allowed units and their factors in subclasses

    def __new__(cls, quantity: str):
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
        instance._original_value = value
        instance._unit = unit
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
        pattern = r"^\s*([-+]?\d+(\.\d+)?)\s*(\w+)\s*$"
        match = re.match(pattern, quantity)

        if not match:
            raise ValueError(f"Invalid format for value with unit: '{quantity}'. Expected format: '<value> <unit>'.")

        value_str, _, unit = match.groups()
        return float(value_str), unit

    @property
    def unit(self) -> str:
        """Get the original unit."""
        return self._unit

    def __str__(self) -> str:
        """Return a human-readable string with the original value and unit."""
        return f"{self._original_value} {self._unit}"

    def __repr__(self):
        return str(self)


class Voltage(UnitQuantity):
    """
    Represents a voltage value with units (e.g., V, mV, kV, etc.).
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "V": 1,        # base unit: volts
        "mV": 1e-3,    # millivolts to volts
        "μV": 1e-6,    # microvolts to volts
        "nV": 1e-9,    # nanovolts to volts
        "kV": 1e3,     # kilovolts to volts
    }


class Angle(UnitQuantity):
    """
    Represents an angular value with units (e.g. rad, deg).
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "rad": 1,           # Base unit: radians
        "deg": math.pi/180  # degrees to radians
    }


class Frequency(UnitQuantity):
    """
    Represents a frequency value with units (e.g. Hz, kHz, MHz, GHz)
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "Hz": 1,        # base unit: hertz
        "kHz": 1e3,     # kilohertz to hertz
        "MHz": 1e6,     # megahertz to hertz
        "GHz": 1e9,     # gigahertz to hertz
    }


class Time(UnitQuantity):
    """
    Represents a time value with units (e.g. s, ms, μs, ns, min, hr, days)
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "s": 1,         # base unit: seconds
        "ms": 1e-3,     # milliseconds to seconds
        "μs": 1e-6,     # microseconds to seconds
        "ns": 1e-9,     # nanoseconds to seconds
        "min": 60.0,    # minutes to seconds
        "hr": 3600.0,   # hours to seconds
        "days": 86400.0 # days to seconds
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

    def __init__(self, min: str, max: str):
        """
        Initialize the range with strings specifying values and units.

        Args:
            min (str): Minimum value with unit (e.g., "100 mV").
            max (str): Maximum value with unit (e.g., "1 V").

        Raises:
            ValueError: If the input format is invalid or the range is invalid (min >= max).
        """
        # Parse min and max values using the UnitQuantity class
        self._min = self.UNIT_QUANTITY_CLASS(min)
        self._max = self.UNIT_QUANTITY_CLASS(max)

        # Validate that min < max in the base unit
        if float(self._min) >= float(self._max):
            raise ValueError(f"Invalid range: min ({self._min}) must be less than max ({self._max}).")

        # Set the base unit as the first key from the allowed units and multipliers
        self.unit = next(iter(self.UNIT_QUANTITY_CLASS.ALLOWED_UNITS_AND_MULTIPLIERS))

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
    def range(self) -> float:
        """Get the range as the difference between max and min."""
        return self.UNIT_QUANTITY_CLASS(str(self.max - self.min) + " " + self.unit)

    def __str__(self) -> str:
        """Return a human-readable string representation of the range."""
        return f"{self._min} to {self._max}"

    def __repr__(self) -> str:
        return f"RangeWithUnits({self._min}, {self._max})"


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
        


# # For testing
# if __name__ == "__main__":

#     angle_range = AngleRange(min="-6 deg", max="6 deg")

#     assert angle_range.within_range(Angle("1.2 deg"))