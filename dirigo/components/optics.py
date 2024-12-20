from dataclasses import dataclass
import math

@dataclass
class LaserScanningOptics: # possibly this should be an interface
    objective_focal_length: float # meters
    relay_magnification: float # dimensionless
    # scan range limits?

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


