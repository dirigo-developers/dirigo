from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dirigo.sw_interfaces.worker import Worker
from dirigo.components.io import load_toml

if TYPE_CHECKING:
    from dirigo.components.hardware import Hardware 



class AcquisitionSpec:
    """Marker class for an acquisition specification."""
    pass


@dataclass
class AcquisitionBuffer:
    data: np.ndarray # Dimensions: Record, Sample, Channel
    timestamps: float | np.ndarray | None = None # should be one or more time points (in seconds since the start)
    positions: tuple[float] | np.ndarray | None = None # should be one or more sets of coordinates (x,y)


class Acquisition(Worker):
    """
    Dirigo interface for data acquisition worker thread.
    """
    REQUIRED_RESOURCES: list[str] = None # The first object in the list should be the data capture device (digitizer, camera, etc)
    SPEC_LOCATION: str = None
    SPEC_OBJECT: AcquisitionSpec
    
    def __init__(self, hw: 'Hardware', spec: AcquisitionSpec):
        super().__init__()
        self.hw = hw
        self.check_resources()
        self.spec = spec
    
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
    def get_specification(cls, spec_name: str = "default") -> AcquisitionSpec:
        """Return the associated Specification object"""
        spec_fn = spec_name + ".toml"
        spec = load_toml(cls.SPEC_LOCATION / spec_fn)
        return cls.SPEC_OBJECT(**spec)
    
    # Note that there is an abstractmethod, run() in the parent class. 
    # Acquisition subclasses must implement it.

    @property
    def data_acquisition_device(self):
        """Returns handle to the hardware resource actually acquiring data."""
        acq_device = self.REQUIRED_RESOURCES[0].__name__
        if acq_device == "Digitizer":
            return self.hw.digitizer
        elif acq_device == "LineScanCamera":
            return self.hw.line_scan_camera
        else:
            raise RuntimeError(f"Invalid data capture device: {acq_device}")
