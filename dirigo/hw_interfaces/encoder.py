from abc import abstractmethod
from typing import Literal

import numpy as np

from dirigo.components import units
from dirigo.hw_interfaces.hw_interface import HardwareInterface


class LinearEncoder(HardwareInterface):
    """
    Abstraction of a linear encoder. Linear encoders can be used to either log
    stage axis position at triggered times (for post-hoc motion correction) or 
    to derive a linearized capture trigger (e.g. to provide to a line-scan
    camera). Use method start_logging() with read() for position capture and use 
    method start_triggering() for linearized trigger out. If either is not 
    implemented, raises NotImplementedError.
    """
    attr_name = "encoder"
    VALID_AXES = {'x', 'y', 'z'}

    def __init__(self, axis: str, **kwargs):
        self.axis = axis

    @property
    def axis(self):
        return self._axis
    
    @axis.setter
    def axis(self, new_axis: str):
        if new_axis in self.VALID_AXES:
            self._axis = new_axis
        else:
            raise ValueError(f"Error setting encoder axis: Got '{new_axis}'")

    @abstractmethod
    def start_logging(self):
        """Start encoder position logging."""
        pass

    @abstractmethod
    def read_positions(self, nsamples: int) -> np.ndarray:
        pass

    @abstractmethod
    def read_timestamps(self, nsamples: int) -> np.ndarray:
        pass

    @abstractmethod
    def start_triggering(self, 
                         distance_per_trigger: units.Position, 
                         direction: Literal['forward', 'reverse']):
        """Start encoder-derived trigger output."""
        pass

    @abstractmethod
    def stop(self):
        pass
    

class MultiAxisLinearEncoder(HardwareInterface):
    """
    Dirigo interface for an X, Y, and/or Z stage position encoders.
    """
    attr_name = "encoders" # more intuitive than typing hw.multi_axis_linear_encoder
    # TODO, device info?

    @property
    @abstractmethod
    def x(self) -> LinearEncoder:
        """
        Returns reference to the X axis encoder. Raises RuntimeException if 
        not available.
        """
        pass
    
    @property
    @abstractmethod
    def y(self) -> LinearEncoder:
        """
        Returns reference to the Y axis encoder. Raises RuntimeException if 
        not available.
        """
        pass

    @property
    @abstractmethod
    def z(self) -> LinearEncoder:
        """
        Returns reference to the Z axis encoder. Raises RuntimeException if 
        not available.
        """
        pass

    @abstractmethod
    def start_logging(self):
        """Starts all the available encoders."""
        pass

    @abstractmethod
    def read_positions(self, n: int) -> np.ndarray:
        """Reads n samples from all the available position encoders.
        
        Implementations should return an array of shape (n, [axes available]).
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops all the available encoders."""
        pass

