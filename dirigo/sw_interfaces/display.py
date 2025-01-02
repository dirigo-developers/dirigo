from abc import ABC, abstractmethod
import threading
import queue

from dirigo.sw_interfaces import Acquisition, Processor



class Display(threading.Thread, ABC):
    """
    Dirigo interface for display processing.
    """
    def __init__(self, display_queue: queue.Queue, acquisition: Acquisition = None, processor: Processor = None):
        # Instantiate with a display queue and either an Acquisition or Processor
        super().__init__()

        if (acquisition is not None) and (processor is not None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor, not both")
        elif acquisition is not None:
            self._data_queue = acquisition.data_queue
        elif processor is not None:
            self._data_queue = processor.processed_queue
        else:
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor.")

        self.display_queue = display_queue # GUIs will read from this queue


    @abstractmethod
    def run(self):
        pass