"""Nominal analog command scaling derived from vendor full-scale ranges."""

import math

from dirigo import units


def angle_voltage_coefficients(
    angle_range: units.AngleRange,
    command_voltage_range: units.VoltageRange,
    *,
    inverted: bool = False,
) -> tuple[float, float]:
    """Derive nominal voltage scaling from ordered vendor full-scale ranges.

    Return (volts_per_radian, voltage_at_zero), both as plain floats. Apply them
    as voltage = angle_in_radians * volts_per_radian + voltage_at_zero.

    For resonant control, angle_range is zero through the specified maximum
    peak-to-peak optical amplitude. For galvos, it is the specified range of
    instantaneous optical deflection.

    Normally, minimum angle maps to minimum voltage and maximum angle maps to
    maximum voltage. inverted=True swaps the voltage endpoints.

    Use the installed hardware's full-scale specifications, not the experiment's
    selected scan interval. The coefficients do not validate or clip individual
    samples; drivers must enforce command limits before hardware output.

    Args:
        angle_range: Vendor-specified angular input range using the appropriate
            optical angle convention.
        command_voltage_range: Scanner controller's specified command voltage
            range, distinct from the DAQ board's supported output limits.
        inverted: Whether increasing angular input decreases command voltage.

    Returns:
        Finite gain in volts per radian and offset in volts.

    Raises:
        TypeError: A range or inverted has an inappropriate type.
        ValueError: Range endpoints are nonfinite or usable finite coefficients
            cannot be represented.
    """
    if not isinstance(angle_range, units.AngleRange):
        raise TypeError("angle_range must be an AngleRange.")
    if not isinstance(command_voltage_range, units.VoltageRange):
        raise TypeError("command_voltage_range must be a VoltageRange.")
    if not isinstance(inverted, bool):
        raise TypeError("inverted must be a bool.")

    angle_min = float(angle_range.min)
    angle_max = float(angle_range.max)
    voltage_min = float(command_voltage_range.min)
    voltage_max = float(command_voltage_range.max)

    if not all(
        math.isfinite(value)
        for value in (angle_min, angle_max, voltage_min, voltage_max)
    ):
        raise ValueError("Full-scale range endpoints must be finite.")

    if inverted:
        voltage_min, voltage_max = voltage_max, voltage_min

    gain = (voltage_max - voltage_min) / (angle_max - angle_min)
    offset = voltage_min - gain * angle_min

    if not math.isfinite(gain) or gain == 0 or not math.isfinite(offset):
        raise ValueError(
            "Full-scale ranges require finite coefficients and nonzero gain."
        )

    return gain, offset
