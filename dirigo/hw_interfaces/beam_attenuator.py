from abc import ABC, abstractmethod

from dirigo import units
from dirigo.hw_interfaces.hw_interface import HardwareInterface



class BeamAttenuator(HardwareInterface, ABC):
    """
    Abstract base class for hardware that attenuates an optical beam.

    This interface is intended for downstream optical attenuation devices such as:
    - a motorized half-wave plate plus polarizer
    - a Pockels cell used in static attenuation mode

    The controlled quantity is a normalized transmission fraction in the
    interval [0, 1], where:

    - 0.0 means minimum transmitted power
    - 1.0 means maximum transmitted power for the configured optical path
    """

    attr_name = "beam_attenuator"

    DEFAULT_LIMITS = {"min": 0.0, "max": 1.0}

    def __init__(self, 
                 *, 
                 limits: dict | None = None, 
                 **kwargs
                 ):
        super().__init__(**kwargs)

        if limits is None:
            limits = self.DEFAULT_LIMITS

        if not (0.0 <= limits['min'] <= limits['max'] <= 1.0):
            raise ValueError(
                "`fraction_limits` must satisfy 0.0 <= min <= max <= 1.0 "
                f"(got {limits})"
            )

        self._fraction_limits = units.FloatRange(**limits)

    @property
    def fraction_limits(self) -> units.FloatRange:
        """
        Allowed commanded transmission fraction range.

        Usually this is (0.0, 1.0), but a concrete device may expose a smaller
        safe or calibrated operating range.
        """
        return self._fraction_limits

    @property
    @abstractmethod
    def fraction(self) -> float:
        """
        Current commanded transmission fraction.
        """
        ...

    @abstractmethod
    def set_fraction(self, fraction: float, blocking: bool = False) -> None:
        """
        Command a new transmission fraction.

        Parameters
        ----------
        fraction:
            Requested transmission fraction in the allowed interval.
        blocking:
            Whether to wait for the new setting to be reached.
        """
        ...
