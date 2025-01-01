from abc import ABC, abstractmethod
import threading
import queue



class Processor(threading.Thread, ABC):
    """
    Dirigo interface for data processing worker thread.
    """
    def __init__(self, raw_queue: queue.Queue, processed_queue: queue.Queue):
        super().__init__()
        self._raw_queue = raw_queue # populated by Acquisition worker(s)
        self._processed_queue = processed_queue # populated by this worker, consumed by Display or Logging
    
    @abstractmethod
    def run(self):
        pass

