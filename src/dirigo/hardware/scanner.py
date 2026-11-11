"""Validated installation parameters for resonant scanners."""

import math
from dataclasses import dataclass, field

from dirigo import units


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantScannerConfig:
    """Describe the installed scanner independently of its control hardware.

    Amplitudes are peak-to-peak optical scan angles about the scan center.
    For example, motion from -13 deg to +13 deg has an amplitude of 26 deg.

    Quantity fields require the indicated quantity types. Parsing TOML strings
    belongs to the configuration-loading boundary.

    Attributes:
        nominal_frequency: Declared oscillation frequency, not a measurement.
        max_amplitude: Maximum permitted peak-to-peak optical amplitude.
        settling_time: Declared wait after applying a new amplitude before
            acquisition may assume the scanner is ready; not measured feedback.
        frequency_tolerance: Maximum expected fractional deviation from nominal
            frequency. For example, 0.003 denotes 0.3 percent.

    Raises:
        TypeError: A quantity has the wrong type, or frequency_tolerance is not
            a plain integer or float.
        ValueError: A value is nonfinite or violates its physical constraints.
    """

    nominal_frequency: units.Frequency = field(
        metadata={
            "description": "Nominal scanner oscillation frequency.",
            "display_unit": "Hz",
        }
    )
    max_amplitude: units.Angle = field(
        metadata={
            "description": "Maximum permitted peak-to-peak optical scan angle.",
            "display_unit": "deg",
        }
    )
    settling_time: units.Time = field(
        default=units.Time("1 s"),
        metadata={
            "description": "Wait after applying an amplitude change.",
            "display_unit": "ms",
        },
    )
    frequency_tolerance: float = field(
        metadata={
            "description": (
                "Maximum fractional frequency deviation; 0.003 means 0.3 percent."
            ),
        }
    )

    def __post_init__(self) -> None:
        """Validate quantity types, finite values, and parameter constraints."""
        for name, expected_type in (
            ("nominal_frequency", units.Frequency),
            ("max_amplitude", units.Angle),
            ("settling_time", units.Time),
        ):
            value = getattr(self, name)

            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}.")

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if float(self.nominal_frequency) <= 0:
            raise ValueError("nominal_frequency must be positive.")

        if float(self.max_amplitude) <= 0:
            raise ValueError("max_amplitude must be positive.")

        if float(self.settling_time) < 0:
            raise ValueError("settling_time must be nonnegative.")

        tolerance = self.frequency_tolerance
        if isinstance(tolerance, (bool, units.Quantity)) or not isinstance(
            tolerance, (int, float)
        ):
            raise TypeError("frequency_tolerance must be a plain integer or float.")

        if not 0 <= tolerance < 1:
            raise ValueError("frequency_tolerance must satisfy 0 <= value < 1.")
