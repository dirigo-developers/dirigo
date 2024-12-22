import math
import re
from typing import Dict


class RangeWithUnits:
    """
    Describes a range with associated units and supports unit conversion.

    Attributes:
    - min (float): Minimum value in the range (converted to the base unit).
    - max (float): Maximum value in the range (converted to the base unit).
    - unit (str): The original unit of the range (e.g., "Hz", "kHz").
    """
    ALLOWED_UNITS_AND_MULTIPLIERS: Dict[str, float] = None  # Define allowed units and their factors in subclasses

    def __init__(self, min: str, max: str):
        """
        Initialize the range with strings specifying values and units.

        Args:
            min (str): Minimum value with unit (e.g., "-45 degrees").
            max (str): Maximum value with unit (e.g., "45 degrees").

        Raises:
            ValueError: If the input format is invalid or the units are not allowed.
        """
        self._min, unit_min = self._parse_value_with_unit(min)
        self._max, unit_max = self._parse_value_with_unit(max)

        if self.ALLOWED_UNITS_AND_MULTIPLIERS and unit_min not in self.ALLOWED_UNITS_AND_MULTIPLIERS:
            raise ValueError(f"Invalid unit '{unit_min}'. Allowed units are: {list(self.ALLOWED_UNITS_AND_MULTIPLIERS.keys())}.")
        if self.ALLOWED_UNITS_AND_MULTIPLIERS and unit_max not in self.ALLOWED_UNITS_AND_MULTIPLIERS:
            raise ValueError(f"Invalid unit '{unit_max}'. Allowed units are: {list(self.ALLOWED_UNITS_AND_MULTIPLIERS.keys())}.")

        # Convert values to base unit using the multiplier
        self._min *= self.ALLOWED_UNITS_AND_MULTIPLIERS[unit_min]
        self._max *= self.ALLOWED_UNITS_AND_MULTIPLIERS[unit_max]

        if self._min >= self._max:
            raise ValueError(f"Invalid range: min ({self._min}) must be less than max ({self._max}).")

    def _parse_value_with_unit(self, value: str) -> tuple[float, str]:
        """
        Parses a string containing a value and unit.

        Args:
            value (str): Input string (e.g., "-45 degrees").

        Returns:
            tuple[float, str]: Parsed value as a float and the unit as a string.

        Raises:
            ValueError: If the input string is not in the expected format.
        """
        pattern = r"^\s*([-+]?\d+(\.\d+)?)\s*(\w+)\s*$"
        match = re.match(pattern, value)

        if not match:
            raise ValueError(f"Invalid format for value with unit: '{value}'. Expected format: '<value> <unit>'.")

        value_str, _, unit = match.groups()
        return float(value_str), unit

    @property
    def min(self) -> float:
        """Get the minimum value in the base unit."""
        return self._min

    @property
    def max(self) -> float:
        """Get the maximum value in the base unit."""
        return self._max
    
    @property
    def unit(self) -> str:
        """Get the range base unit."""
        return next(iter(self.ALLOWED_UNITS_AND_MULTIPLIERS))

    def within_limits(self, value: float) -> bool:
        """Check whether a value is within the range."""
        return self._min <= value <= self._max

    @property
    def range(self) -> float:
        """Get the range as the difference between max and min."""
        return self._max - self._min


class FrequencyRange(RangeWithUnits):
    """
    Describes a range of frequencies with units (e.g., Hz, kHz, MHz).
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "Hz": 1,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
    }

    def __init__(self, min: str, max: str):
        super().__init__(min, max)


class AngleRange(RangeWithUnits):
    """
    Describes a range of angles with units (e.g., radians, degrees).
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "radians": 1,
        "degrees": math.pi / 180, 
    }

    def __init__(self, min: str, max: str):
        super().__init__(min, max)

        # Additional validation for angles
        if self.min_degrees < -90 or self.max_degrees > 90:
            raise ValueError(
                f"Angle range must be within the valid optical range (-90 to 90 degrees). "
                f"Received: min={self.min_degrees} degrees, max={self.max_degrees} degrees."
            )

    @property 
    def min_degrees(self):
        return self.min * 180 / math.pi
    
    @property 
    def max_degrees(self):
        return self.max * 180 / math.pi
    

class VoltageRange(RangeWithUnits):
    """
    Describes a range of voltages with units (e.g., V, mV, uV).
    """
    ALLOWED_UNITS_AND_MULTIPLIERS = {
        "V": 1,        # Base unit: Volts
        "mV": 1e-3,    # Millivolts to volts
        "uV": 1e-6,    # Microvolts to volts
    }

    def __init__(self, min: str, max: str):
        """
        Initialize the voltage range.

        Args:
            min (str): Minimum voltage with unit (e.g., "0 V").
            max (str): Maximum voltage with unit (e.g., "10 V").
        """
        super().__init__(min, max)
        

if __name__ == "__main__":
    v_range = VoltageRange(min="-1 V", max="1 V")
    None