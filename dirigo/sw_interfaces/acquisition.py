from abc import ABC, abstractmethod
import threading
import queue

from dirigo.components.hardware import Hardware
from dirigo.components.io import load_toml



class AcquisitonSpec:
    """Marker class for an acquisition specification."""
    # TODO, are there any attributes that are constant for all acquisitions?
    pass


class Acquisition(threading.Thread, ABC):
    """
    Dirigo interface for data acquisition worker thread.
    """
    REQUIRED_RESOURCES: list[str] = None
    SPEC_LOCATION: str = None
    SPEC_OBJECT: AcquisitonSpec
    
    def __init__(self, hw: Hardware, data_queue: queue.Queue, spec: AcquisitonSpec):
        super().__init__()
        self.hw = hw
        self.check_resources()
        self.data_queue = data_queue
        self.spec = spec
        self._stop_event = threading.Event()  # Event to signal thread termination
    
    def check_resources(self):
        """Iterates through class attribute REQUIRED_RESOURCES to check if all 
        resources are present.

        Raises RuntimeError if any required resources are missing.
        """
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
            
    @classmethod
    def get_specification(cls, spec_name: str = "default"):
        """Return the associated Specification object"""
        spec_fn = spec_name + ".toml"
        spec = load_toml(cls.SPEC_LOCATION / spec_fn)
        return cls.SPEC_OBJECT(**spec)

    @abstractmethod
    def run(self):
        pass

    def stop(self, blocking=False):
        """Sets a flag to stop acquisition."""
        self._stop_event.set()

        if blocking:
            self.join()  # does not return until thread completes

