from abc import abstractmethod

import numpy as np

from dirigo import io
from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct, Loader
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces.display import Display


class Logger(Worker):
    """Dirigo interface for data logging."""
    def __init__(self, 
                 upstream: Acquisition | Loader | Processor,
                 basename: str = "experiment",
                 ) -> None:
        """Instantiate with either an upstream Acquisition or Processor"""
        super().__init__("Logger") # sets up the thread and the publisher-subcriber interface
        
        if isinstance(upstream, (Processor, Display)): 
            self._processor = upstream
            self._acquisition = upstream._acquisition
        elif isinstance(upstream, (Acquisition, Loader)):
            self._processor = None
            self._acquisition = upstream
        else:
            raise ValueError("Upstream Worker must be either an Acquisition or a Processor")

        self.basename = basename
        self.save_path = io.data_path()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.last_saved_file_path = None
        
        self.frames_per_file: int = 1

    @abstractmethod
    def save_data(self, data: np.ndarray):
        """Save an increment of data"""
        pass

    # def _receive_product(self, block = True, timeout = None) -> AcquisitionProduct | ProcessorProduct:
    #     return super()._receive_product(block, timeout)

