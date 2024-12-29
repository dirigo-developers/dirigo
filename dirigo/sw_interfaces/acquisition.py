from abc import ABC, abstractmethod
import threading
import queue

from dirigo.components.hardware import Hardware



class Acquisition(threading.Thread, ABC):
    """
    Dirigo interface for data acquisition worker thread.
    """
    REQUIRED_RESOURCES: list[str] = None
    SPEC_LOCATION: str = None
    
    def __init__(self, hw: Hardware, data_queue: queue.Queue, spec):
        super().__init__()
        self.hw = hw
        self.spec = spec
        self.check_resources()
        self.data_queue = data_queue
        self.running = False
        self._stop_event = threading.Event()  # Event to signal thread termination
    
    def check_resources(self):
        if self.REQUIRED_RESOURCES is None: # No acquisition without any resources
            raise NotImplementedError(
                f"Acquisition must implement 'required_resources' attribute."
            )
        hw_items = self.hw.__dict__.items()

        for resource in self.REQUIRED_RESOURCES:
            present = any([isinstance(v, resource) for k,v in hw_items])
            if not present:
                raise RuntimeError(
                    f"Required resource, {resource} is not available as a "
                    f"resource."
                )
            
    @staticmethod
    @abstractmethod
    def get_specification():
        """Return the associated Specification object"""
        pass

    @abstractmethod
    def run(self):
        pass

    def stop(self, blocking=False):
        """Sets a flag to stop acquisition."""
        self._stop_event.set()

        if blocking:
            self.join()  # Wait for the thread to finish

