from abc import abstractmethod

import numpy as np

from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces import Acquisition, Processor



class Logger(Worker):
    """Dirigo interface for data logging."""
    def __init__(self, acquisition: Acquisition = None, processor: Processor = None):
        """Instantiate with either an Acquisition or Processor"""
        super().__init__() # sets up the thread and the publisher-subcriber interface
        
        if (acquisition is not None) and (processor is not None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor, not both")
        elif (acquisition is None) and (processor is None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor.")
        self._acquisition = acquisition
        self._processor = processor

        self._save_path = None
        self._append = False
        # Track frames/buffers saved

    @abstractmethod
    def save_data(self, data: np.ndarray):
        """Save an increment of data"""
        pass

