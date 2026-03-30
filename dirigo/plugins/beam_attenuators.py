from math import asin, sin, sqrt

from dirigo import units
from dirigo.hw_interfaces.beam_attenuator import BeamAttenuator
from dirigo.hw_interfaces.stage import RotationStage


class HalfWavePlateAttenuator(BeamAttenuator):
    """
    Beam attenuator based on a motorized half-wave plate rotated in front of a polarizer.

    Parameters
    ----------
    min_transmission_position:
        Rotation-stage angle corresponding to minimum transmission.
    rotation_stage:
        Rotation stage controlling the half-wave plate.
    """

    def __init__(self, 
                 *,
                 min_transmission_position: str | units.Angle,
                 rotation_stage: RotationStage,
                 limits: dict | None = None, 
                 **kwargs
                 ):
        super().__init__(limits=limits, **kwargs)

        self._min_position = units.Angle(min_transmission_position)

        if not isinstance(rotation_stage, RotationStage):
            raise RuntimeError("WavePlateAttenuator requires a valid RotationStage reference")
        
        self._stage = rotation_stage

    @property
    def fraction(self) -> float:
        angle = self._stage.position - self._min_position
        return sin(2 * angle) ** 2

    def set_fraction(self, fraction: float, blocking: bool = False) -> None:
        fraction = float(fraction)

        if not self.fraction_limits.within_range(fraction):
            raise ValueError(
                f"Requested fraction {fraction} is outside allowed range "
                f"{self.fraction_limits.min} to {self.fraction_limits.max}."
            )
        
        angle = units.Angle( asin(sqrt(fraction)) / 2 )
        self._stage.move_to(self._min_position + angle, blocking)