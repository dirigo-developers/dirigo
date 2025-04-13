from dataclasses import dataclass
from abc import abstractmethod

import numpy as np

from dirigo import units 
from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces.acquisition import Acquisition


# TODO, 
# Must a Processor always be associated with an Acquisition? 
# Can Processors be cascaded?
# Limitation: currently needs to be linked to Acquisition specifically


# TODO, this is virtually the same as the digitizer.Buffer class, worth consolidating?
@dataclass
class ProcessedFrame():
    """Data class to encapsulate a processed frame and its metadata."""
    data: np.ndarray # Dimensions: Y, X, Channel
    timestamps: float | np.ndarray | None = None # should be one or more time points (in seconds since the start)
    positions: tuple[float] | np.ndarray | None = None # should be one or more sets of coordinates (x,y)


class Processor(Worker):
    """
    Dirigo interface for data processing worker thread.
    """
    def __init__(self, acquisition: Acquisition):
        """Stores the acquisition and spec in private attributes"""
        super().__init__()
        self._acq = acquisition
        self._spec = acquisition.spec
    
    @property
    @abstractmethod # Not sure this is absolutely needed for every subclass of this.
    def data_range(self) -> units.ValueRange:
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        pass