from abc import ABC, abstractmethod

import numpy as np



class LinearEncoder(ABC):
    """Abstraction of a linear encoder channel"""
    def __init__(self, axis: str, **kwargs):
        self._axis = axis # TODO, check valid axis?

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def read(self, nsamples: int) -> np.ndarray:
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
    def start(self):
        """Starts all the available encoders."""
        pass

    @abstractmethod
    def read(self, n: int) -> np.ndarray:
        """Reads n samples from all the available encoders.
        
        Implementations should return an array of shape (n, [axes available]).
        """
        pass

    @abstractmethod
    def stop(self):
        """Stops all the available encoders."""
        pass