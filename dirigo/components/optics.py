import dirigo


# TODO, rework as interface?

class LaserScanningOptics: 
    def __init__(self, objective_focal_length: str, relay_magnification: float):
        self._objective_focal_length = dirigo.Position(objective_focal_length)
        self._relay_magnification = float(relay_magnification)

    @property
    def objective_focal_length(self) -> dirigo.Position:
        """Returns the objective focal length."""
        return self._objective_focal_length
    
    @property
    def relay_magnification(self) -> float:
        """Returns the scan relay system (typically: scan lens + tube lens) 
        lateral magnification.
        """
        return self._relay_magnification

    def scan_angle_to_object_position(self, angle: dirigo.Angle) -> dirigo.Position:
        """
        Return the focus position for a certain scanner angle (optical).
        """
        objective_angle = angle / self.relay_magnification
        return dirigo.Position(objective_angle * self.objective_focal_length)

    def object_position_to_scan_angle(self, position: dirigo.Position) -> dirigo.Angle:
        """
        Return the scanner angle (optical) required for a certain focus position.
        """
        objective_angle = position / self.objective_focal_length
        angle = objective_angle * self.relay_magnification
        return dirigo.Angle(angle)

