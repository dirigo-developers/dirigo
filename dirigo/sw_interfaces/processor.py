from dataclasses import dataclass
from abc import abstractmethod
from typing import Self

import numpy as np

from dirigo import units 
from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces.acquisition import Acquisition



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
    def __init__(self, upstream: Acquisition | Self):
        """Stores the acquisition and spec in private attributes"""
        super().__init__()
        if isinstance(upstream, Acquisition):
            self._acq = upstream
            self._spec = upstream.spec
        elif isinstance(upstream, Processor):
            self._acq = upstream._acq
            self._spec = upstream._acq.spec
        else:
            raise ValueError("Upstream worker passed to Processor must be either an Acquisition or another Processor")
    
    @property
    @abstractmethod # Not sure this is absolutely needed for every subclass of this.
    def data_range(self) -> units.ValueRange:
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        pass