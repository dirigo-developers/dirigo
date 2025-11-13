from pydantic import Field

from dirigo import units
from dirigo.devices.optics.interface import OpticsDevice, OpticsDeviceConfig



class LaserScanningOpticsConfig(OpticsDeviceConfig):
    objective_focal_length: units.Length = Field(
        ..., description="Objective lens effective focal length."
    )
    relay_magnification: float = Field(
        ..., description="Scanner → objective relay magnification."
    )


class LaserScanningOptics(OpticsDevice[LaserScanningOpticsConfig]):
    ConfigModel = LaserScanningOpticsConfig 

    def __init__(self, config: LaserScanningOpticsConfig) -> None:
        super().__init__(config)
        self._objective_focal_length = config.objective_focal_length
        self._relay_magnification = config.relay_magnification

    def scan_angle_to_object_position(self, 
                                      angle: units.Angle, 
                                      ) -> units.Length:
        """
        Return the focus position for a certain scanner angle (optical).
        """
        objective_angle = angle / self._relay_magnification 
        position = float(objective_angle) * self._objective_focal_length
        return units.Length(position)

    def object_position_to_scan_angle(self, 
                                      position: units.Length) -> units.Angle:
        """
        Return the scanner angle (optical) required for a certain focus position.
        """
        objective_angle = position / self._objective_focal_length
        angle = objective_angle * self._relay_magnification
        return units.Angle(angle)