from abc import ABC, abstractmethod
import threading
import queue

from dirigo.sw_interfaces.acquisition import AcquisitionSpec, Acquisition


class Processor(threading.Thread, ABC):
    """
    Dirigo interface for data processing worker thread.
    """
    def __init__(self, acquisition: Acquisition, processed_queue: queue.Queue):
        super().__init__()
        self._acq = acquisition
        self._spec = acquisition.spec
        self._raw_queue = acquisition.data_queue # populated by Acquisition worker(s)
        self.processed_queue = processed_queue # populated by this worker, consumed by Display or Logging
    
    @abstractmethod
    def run(self):
        pass

