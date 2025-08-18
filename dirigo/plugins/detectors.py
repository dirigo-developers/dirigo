from typing import Optional

from dirigo import units
from dirigo.hw_interfaces.detector import Detector



class GenericDetector(Detector):
    """
    A generic detector with no controllable or readable parameters.
    
    This plugin provides a means of documenting the detector's presence and 
    model number.
    """
    def __init__(self, model: Optional[str] = None, **kwargs):
        super().__init__()
        self._model = model

    @property
    def _name(self) -> str:
        if self._model:
            return f"{self._model} (Generic Detector)"
        else:
            return "Generic Detector"

    @property
    def enabled(self) -> bool:
        return True # assume on
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        raise NotImplementedError(
            f"{self._name} cannot be enabled/disabled in software."
        )
    
    @property
    def gain(self):
        """Switchable gain; raise NotImplementedError if fixed."""
        raise NotImplementedError(
            f"Gain is not readable for {self._name}"
        )

    @gain.setter
    def gain(self, value) -> None: 
        raise NotImplementedError(
            f"Gain is not adjustable for {self._name}"
        )
    
    @property
    def gain_range(self):
        raise NotImplementedError(
            f"Gain range is not readable for {self._name}"
        )

    @property
    def bandwidth(self) -> units.Frequency: 
        """Switchable bandwidth; raise NotImplementedError if fixed."""
        raise NotImplementedError(
            f"Bandwidth is not readable for {self._name}"
        )

    @bandwidth.setter
    def bandwidth(self, value: units.Frequency):
        raise NotImplementedError(
            f"Bandwidth is not adjustable for {self._name}"
        )