from typing import Optional
from pathlib import Path

from dirigo.components import units
from dirigo.components.io import load_line_width_calibration


"""
Notes:
Concerning distortion and magnification error, we assume that the scanner always
produces the correct angle. This may or may not be true, but besides the point.
Any corrections are applied at Optics class level.
"""

class LaserScanningOptics: 
    def __init__(self, 
                 objective_focal_length: str, 
                 relay_magnification: float) -> None:
        """
        fast_axis_correction (float): extends scan by factor to correct mag error
        """
        self._objective_focal_length = units.Position(objective_focal_length)
        self._relay_magnification = float(relay_magnification)

        # load line width calibration, if fails don't use calibration
        try:
            self._fast_axis_calibration = load_line_width_calibration()
        except:
            self._fast_axis_calibration = None

        self._slow_axis_calibration = None

    @property
    def objective_focal_length(self) -> units.Position:
        """Returns the objective focal length."""
        return self._objective_focal_length
    
    @property
    def relay_magnification(self) -> float:
        """Returns the scan relay system (typically: scan lens + tube lens) 
        lateral magnification.
        """
        return self._relay_magnification

    def scan_angle_to_object_position(self, 
                                      angle: units.Angle, 
                                      axis: Optional[str] = None
                                      ) -> units.Position:
        """
        Return the focus position for a certain scanner angle (optical).

        Specify axis ('fast', 'slow') to invoke correction factor.
        """
        objective_angle = angle / self.relay_magnification 
        position = float(objective_angle) * self.objective_focal_length
        return units.Position(position)

    def object_position_to_scan_angle(self, 
                                      position: units.Position,
                                      axis: Optional[str] = None) -> units.Angle:
        """
        Return the scanner angle (optical) required for a certain focus position.

        Specify axis ('fast', 'slow') to invoke correction factor.
        """
        if axis == "fast" and self._fast_axis_calibration:
            angle = self._fast_axis_calibration(float(position))
        elif axis == "slow" and self._slow_axis_calibration:
            angle = self._slow_axis_calibration(float(position))
        else:
            objective_angle = position / self.objective_focal_length
            angle = objective_angle * self.relay_magnification
            
        return units.Angle(angle)
       

class CameraOptics:
    """
    Optics to use with a parallel array of detectors, usually an image sensor.
    """
    def __init__(self, magnification: float | int, **kwargs):
        if not (isinstance(magnification, float) or isinstance(magnification, int) ):
            raise ValueError("Magnification must be a float or an integer")
        self._magnification = float(magnification)

    @property
    def magnification(self) -> float:
        return self._magnification


