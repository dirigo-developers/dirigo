from abc import abstractmethod
from typing import Optional

import numpy as np
from platformdirs import user_documents_path

from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces import Acquisition, Processor
from dirigo.components.io import SystemConfig


class Logger(Worker):
    """Dirigo interface for data logging."""
    def __init__(self, 
                 upstream: Acquisition | Processor,
                 basename: str = "experiment",
                 system_config: Optional[SystemConfig] = None
                 ) -> None:
        """Instantiate with either an upstream Acquisition or Processor"""
        super().__init__("Logger") # sets up the thread and the publisher-subcriber interface
        
        if isinstance(upstream, Processor): # TODO refactor
            self._processor = upstream
            self._acq = upstream._acq
        elif isinstance(upstream, Acquisition):
            self._processor = None
            self._acq = upstream
        else:
            raise ValueError("Upstream Worker must be either an Acquisition or a Processor")

        self.basename = basename
        self.save_path = user_documents_path() / "Dirigo"
        self.save_path.mkdir(parents=True, exist_ok=True)

        if isinstance(system_config, SystemConfig):
            self._system_config = system_config
        elif system_config is None:
            self._system_config = None
        else:
            raise ValueError(f"Expecting SystemConfig class, got {type(system_config)}")
        
        self.frames_per_file: int = None
        # Track frames/buffers saved

    @abstractmethod
    def save_data(self, data: np.ndarray):
        """Save an increment of data"""
        pass

