"""Validated installation parameters for resonant scanners."""

import math
from abc import abstractmethod
from dataclasses import dataclass, field

from dirigo import units
from dirigo.hardware.device import Device


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantScannerConfig:
    """Describe the installed scanner independently of its control hardware.

    Amplitudes are peak-to-peak optical scan angles about the scan center.
    For example, motion from -13 deg to +13 deg has an amplitude of 26 deg.

    Quantity fields require the indicated quantity types. Parsing TOML strings
    belongs to the configuration-loading boundary.

    Attributes:
        nominal_frequency: Declared oscillation frequency, not a measurement.
        max_amplitude: Vendor-specified maximum peak-to-peak optical amplitude.
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
            "description": "Vendor-specified maximum peak-to-peak optical amplitude.",
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


class ResonantScanner(Device):
    """Control the commanded optical amplitude of a resonant scanner.

    Construction stores configuration without issuing an amplitude command.
    The initial amplitude is unknown until a command succeeds.

    Amplitude commands do not wait for physical settling. Acquisition code
    decides when to proceed using config.settling_time.

    Calls must not overlap. Subclasses implement amplitude output and resource
    cleanup through the protected hooks.
    """

    def __init__(self, config: ResonantScannerConfig) -> None:
        """Store validated physical parameters without accessing hardware.

        Raises:
            TypeError: config is not a ResonantScannerConfig.
        """
        if not isinstance(config, ResonantScannerConfig):
            raise TypeError("config must be a ResonantScannerConfig.")

        self._config = config
        self._amplitude: units.Angle | None = None
        self._closed = False

    @property
    def config(self) -> ResonantScannerConfig:
        """Immutable physical parameters, available even after closure."""
        return self._config

    @property
    def amplitude(self) -> units.Angle | None:
        """Last successfully commanded peak-to-peak optical amplitude.

        Return None before the first successful command. This cached value
        remains available after closure and is not a measurement of motion.

        A failed command leaves this history unchanged, but does not establish
        the physical scanner's current state.
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: units.Angle) -> None:
        """Apply a finite optical amplitude from zero through max_amplitude.

        Update the cached amplitude only after the driver call succeeds.
        Do not wait for physical settling.

        Raises:
            RuntimeError: The scanner is closed.
            TypeError: value is not an Angle.
            ValueError: value is nonfinite or outside the configured bounds.

        Driver exceptions propagate without changing the cached amplitude.
        """
        if self._closed:
            raise RuntimeError("Scanner is closed.")

        if not isinstance(value, units.Angle):
            raise TypeError("amplitude must be an Angle.")

        if not math.isfinite(float(value)):
            raise ValueError("amplitude must be finite.")

        if not 0 <= float(value) <= float(self._config.max_amplitude):
            raise ValueError("amplitude must be between zero and max_amplitude.")

        self._write_amplitude(value)
        self._amplitude = value

    def close(self) -> None:
        """Disable amplitude commands and release owned resources once.

        Do not issue an amplitude command from this base implementation.
        Concrete drivers must document output behavior when resources are
        released.

        Closure is permanent even if cleanup raises. Subsequent calls do
        nothing; they do not retry cleanup or repeat its exception.
        """
        if self._closed:
            return

        self._closed = True
        self._close()

    def _snapshot_settings(self) -> dict[str, object]:
        """Select declared physical parameters and cached command history.

        These fields remain available after closure. The amplitude describes
        the last successful command, not measured scanner motion.
        """
        return {
            "nominal_frequency": self.config.nominal_frequency,
            "max_amplitude": self.config.max_amplitude,
            "settling_time": self.config.settling_time,
            "frequency_tolerance": self.config.frequency_tolerance,
            "commanded_amplitude": self.amplitude,
        }

    @abstractmethod
    def _write_amplitude(self, value: units.Angle) -> None:
        """Submit a validated amplitude command without waiting for settling."""

    @abstractmethod
    def _close(self) -> None:
        """Release owned resources, attempting remaining cleanup after failures.

        Support partially initialized resources. Public amplitude commands are
        already disabled when this hook runs.
        """
