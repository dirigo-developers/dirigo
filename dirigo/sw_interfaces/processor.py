from dataclasses import dataclass

import numpy as np

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
        super().__init__()
        self._acq = acquisition
        self._spec = acquisition.spec
    
