import math


class LaserScanningOptics: # TODO, rework as interface?
    def __init__(self, objective_focal_length: str|float, relay_magnification: float):
        # validate focal length argument
        if isinstance(objective_focal_length, str):
            v, u = objective_focal_length.split()
            if u.lower() in ["mm", "millimeter", "millimeters"]:
                v = float(v) / 1000
            elif u.lower() in ["m", "meter", "meters"]:
                v = float(v)
            else:
                raise ValueError(
                    f"Expecting objective focal length units in mm or m, got {u}"
                )
        self._objective_focal_length = v

        self._relay_magnification = float(relay_magnification)

    @property
    def objective_focal_length(self):
        """Returns the objective focal length in meters."""
        return self._objective_focal_length
    
    @property
    def relay_magnification(self):
        """Returns the scan relay lateral magnification"""
        return self._relay_magnification

    def scan_angle_to_object_position(self, angle: float) -> float:
        """
        Return the focus position (in meters) for a certain scanner angle (in  
        degrees, optical).
        """
        angle_rad = math.pi * angle / 180
        objective_angle_rad = angle_rad / self.relay_magnification
        return objective_angle_rad * self.objective_focal_length

    def object_position_to_scan_angle(self, position: float) -> float:
        """
        Return the scanner angle (in degrees, optical) required for a certain 
        focus position (in meters).
        """
        objective_angle_rad = position / self.objective_focal_length
        angle_rad = objective_angle_rad * self.relay_magnification
        return 180 * angle_rad / math.pi


