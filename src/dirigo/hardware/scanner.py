"""Validated installation parameters for resonant scanners."""

import math
from abc import abstractmethod
from dataclasses import dataclass, field
from threading import RLock

from dirigo import units
from dirigo.hardware.device import Device
from dirigo.hardware.waveform import AngularWaveform


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


@dataclass(frozen=True, slots=True, kw_only=True)
class GalvoScannerConfig:
    """Physical specifications for one galvanometer scanner axis.

    angle_range is the inclusive vendor-specified optical deflection range,
    not the angular extent selected for a particular acquisition.
    """

    angle_range: units.AngleRange = field(
        metadata={
            "description": "Full-scale optical deflection range.",
            "display_unit": "deg",
        }
    )

    def __post_init__(self) -> None:
        """Require a typed optical range with finite endpoints."""
        if not isinstance(self.angle_range, units.AngleRange):
            raise TypeError("angle_range must be an AngleRange.")
        if not all(
            math.isfinite(float(endpoint))
            for endpoint in (self.angle_range.min, self.angle_range.max)
        ):
            raise ValueError("angle_range endpoints must be finite.")


class GalvoScanner(Device):
    """One galvo axis with a static optical-angle command.

    Construction leaves the output unchanged. Scalar commands support parking
    and alignment; buffered waveform execution is a separate operation.

    angle records the last successful static command, not measured position
    or instantaneous position during waveform output. Static commands and waveform
    reservations are serialized internally.
    """

    def __init__(self, config: GalvoScannerConfig) -> None:
        """Store physical specifications without commanding the scanner.

        Raises:
            TypeError: config is not a GalvoScannerConfig.
        """
        if not isinstance(config, GalvoScannerConfig):
            raise TypeError("config must be a GalvoScannerConfig.")
        self._config = config
        self._output_lock = RLock()
        self._output_owner: object | None = None
        self._angle: units.Angle | None = None
        self._closed = False

    @property
    def config(self) -> GalvoScannerConfig:
        """Physical specifications for this axis."""
        return self._config

    @property
    def angle(self) -> units.Angle | None:
        """Last successful static command, or None before any such command.

        This history does not describe instantaneous waveform position.
        It remains readable after closure.
        """
        return self._angle

    @angle.setter
    def angle(self, value: units.Angle) -> None:
        """Command a static optical angle without waiting for settling.

        The axis must be open and unreserved. Update command history only
        after the driver returns successfully. A driver error may leave
        physical output uncertain while the recorded history stays unchanged.

        Raises:
            RuntimeError: The scanner is closed or reserved for waveform output.
            TypeError: value is not an Angle.
            ValueError: value is nonfinite or outside angle_range.

        Driver exceptions propagate.
        """
        with self._output_lock:
            if self._closed:
                raise RuntimeError("Scanner is closed.")
            if self._output_owner is not None:
                raise RuntimeError("Scanner is reserved for waveform output.")
            if not isinstance(value, units.Angle):
                raise TypeError("angle must be an Angle.")
            if not math.isfinite(float(value)):
                raise ValueError("angle must be finite.")
            if not (
                float(self.config.angle_range.min)
                <= float(value)
                <= float(self.config.angle_range.max)
            ):
                raise ValueError("angle must be within angle_range.")

            self._write_angle(value)
            self._angle = value

    def validate_waveform(self, waveform: AngularWaveform) -> None:
        """Check every sample against this axis's inclusive optical limits.

        This read-only check is available even after closure. It does not
        issue output, change static-command history, or reserve the axis.
        Output-session timing and hardware capabilities are checked separately.

        Raises:
            TypeError: waveform is not an AngularWaveform.
            ValueError: Any sample is outside angle_range.
        """
        if not isinstance(waveform, AngularWaveform):
            raise TypeError("waveform must be an AngularWaveform.")

        samples = waveform.samples
        if float(samples.min()) < float(self.config.angle_range.min) or float(
            samples.max()
        ) > float(self.config.angle_range.max):
            raise ValueError("waveform samples must be within angle_range.")

    def close(self) -> None:
        """Release resources after all borrowing output sessions are closed.

        A reserved axis raises RuntimeError and remains open. Otherwise,
        closure is permanent even if cleanup fails, and repeated calls do
        nothing. Closure does not select a parking angle.
        """
        with self._output_lock:
            if self._closed:
                return
            if self._output_owner is not None:
                raise RuntimeError("Scanner is reserved for waveform output.")
            self._closed = True
            self._close()

    @abstractmethod
    def _write_angle(self, value: units.Angle) -> None:
        """Apply one validated static command with the output lock held."""

    @abstractmethod
    def _close(self) -> None:
        """Release resources owned by this driver without choosing an angle."""

    def _snapshot_settings(self) -> dict[str, object]:
        """Record the optical range and last successful static command."""
        return {
            "angle_range": {
                "min": self.config.angle_range.min,
                "max": self.config.angle_range.max,
            },
            "last_static_angle": self.angle,
        }

    def _reserve_output(self, owner: object) -> None:
        """Reserve this axis for one output session."""
        with self._output_lock:
            if self._closed:
                raise RuntimeError("Scanner is closed.")
            if self._output_owner is not None:
                raise RuntimeError("Scanner is reserved for waveform output.")
            self._output_owner = owner

    def _release_output(self, owner: object) -> None:
        """Release only the reservation belonging to this session."""
        with self._output_lock:
            if self._output_owner is owner:
                self._output_owner = None
