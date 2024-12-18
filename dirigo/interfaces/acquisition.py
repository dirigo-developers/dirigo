from abc import ABC, abstractmethod
import threading
import queue

from dirigo.components.hardware import Hardware



class Acquisition(threading.Thread, ABC):
    """
    Defines the interface for acquisition plugins
    """
    def __init__(self, hw:Hardware, data_queue:queue.Queue):
        super().__init__(self)
        self.hw = hw
        self.data_queue = data_queue
        self.running = False

    @abstractmethod
    def run(self):
        pass

    def stop(self):
        """Sets a flag to stop acquisition."""
        self.running = False


