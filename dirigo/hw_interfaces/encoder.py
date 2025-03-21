from abc import ABC, abstractmethod

import numpy as np

from dirigo import units


class LinearEncoder(ABC):
    """
    Abstraction of a linear encoder. Linear encoders can be used to either log
    stage axis position at triggered times (for post-hoc motion correction) or 
    to derive a linearized capture trigger (e.g. to provide to a line-scan
    camera). Use method start_logging() with read() for position capture and use 
    method start_triggering() for linearized trigger out. If either is not 
    implemented, raises NotImplementedError.
    """
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
    def start_triggering(self, distance_per_trigger: units.Position):
        """Start encoder-derived trigger output."""
        pass

    @abstractmethod
    def stop(self):
        pass
    

class MultiAxisLinearEncoder(ABC):
    """
    Dirigo interface for an X, Y, and/or Z stage position encoders.
    """
    # TODO, device info?

    @property
    @abstractmethod
    def x(self) -> None | LinearEncoder:
        """If available, returns reference to the X axis encoder."""
        pass
    
    @property
    @abstractmethod
    def y(self) -> None | LinearEncoder:
        """If available, returns reference to the Y axis encoder."""
        pass

    @property
    @abstractmethod
    def z(self) -> None | LinearEncoder:
        """If available, returns reference to the Z axis encoder."""
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

