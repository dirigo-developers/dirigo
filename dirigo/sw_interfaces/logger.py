from abc import abstractmethod
from pathlib import Path

import numpy as np

from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces import Acquisition, Processor



class Logger(Worker):
    """Dirigo interface for data logging."""
    def __init__(self, upstream: Acquisition | Processor):
        """Instantiate with either an upstream Acquisition or Processor"""
        super().__init__() # sets up the thread and the publisher-subcriber interface
        
        if isinstance(upstream, Processor): # TODO refactor
            self._processor = upstream
            self._acquisition = upstream._acq
        elif isinstance(upstream, Acquisition):
            self._processor = None
            self._acquisition = upstream
        else:
            raise ValueError("Upstream Worker must be either an Acquisition or a Processor")

        self.basename: str = None
        self.save_path: Path = None # potentially this could be a kwarg
        self.frames_per_file: int = None
        # Track frames/buffers saved

    @abstractmethod
    def save_data(self, data: np.ndarray):
        """Save an increment of data"""
        pass

