"""Optical calibration for point-scanning microscopy."""

import math
from dataclasses import dataclass
from typing import Literal

from dirigo import units


@dataclass(frozen=True, slots=True, kw_only=True)
class LaserScanningOptics:
    """Convert object-space extents to optical scanner-angle spans.

    Use the linear focal-length and relay-magnification model from v0.3.
    Axis corrections multiply the commanded angle for a requested extent.

    Angles are optical deflection angles. Passing a complete frame extent
    returns its complete angular span, without an additional factor of two.

    This object describes calibration and owns no hardware resources.
    """

    objective_focal_length: units.Length
    relay_magnification: float
    fast_axis_correction: float = 1.0
    slow_axis_correction: float = 1.0

    def __post_init__(self) -> None:
        """Require finite, positive optical parameters."""
        if not isinstance(self.objective_focal_length, units.Length):
            raise TypeError("objective_focal_length must be a Length.")

        focal_length = float(self.objective_focal_length)
        if not math.isfinite(focal_length) or focal_length <= 0:
            raise ValueError("objective_focal_length must be finite and positive.")

        for name in (
            "relay_magnification",
            "fast_axis_correction",
            "slow_axis_correction",
        ):
            value = getattr(self, name)

            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a plain number.")

            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")

    def _correction(self, axis: Literal["fast", "slow"]) -> float:
        """Select the calibration correction for one scanner axis."""
        if axis == "fast":
            return self.fast_axis_correction
        if axis == "slow":
            return self.slow_axis_correction
        raise ValueError("axis must be 'fast' or 'slow'.")

    def object_length_to_scan_angle(
        self,
        length: units.Length,
        *,
        axis: Literal["fast", "slow"],
    ) -> units.Angle:
        """Return the optical angle span covering a positive object extent."""
        if not isinstance(length, units.Length):
            raise TypeError("length must be a Length.")

        if not math.isfinite(float(length)) or float(length) <= 0:
            raise ValueError("length must be finite and positive.")

        value = (
            float(length)
            / float(self.objective_focal_length)
            * self.relay_magnification
            * self._correction(axis)
        )

        if not math.isfinite(value) or value <= 0:
            raise ValueError("Calculated scan angle must be finite and positive.")

        return units.Angle(value)

    def scan_angle_to_object_length(
        self,
        angle: units.Angle,
        *,
        axis: Literal["fast", "slow"],
    ) -> units.Length:
        """Return the positive object extent covered by an optical angle span."""
        if not isinstance(angle, units.Angle):
            raise TypeError("angle must be an Angle.")

        if not math.isfinite(float(angle)) or float(angle) <= 0:
            raise ValueError("angle must be finite and positive.")

        value = (
            float(angle)
            * float(self.objective_focal_length)
            / self.relay_magnification
            / self._correction(axis)
        )

        if not math.isfinite(value) or value <= 0:
            raise ValueError("Calculated object length must be finite and positive.")

        return units.Length(value)
