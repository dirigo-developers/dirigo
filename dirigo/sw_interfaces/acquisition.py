from dirigo.sw_interfaces.worker import Worker
from dirigo.components.hardware import Hardware
from dirigo.components.io import load_toml



class AcquisitionSpec:
    """Marker class for an acquisition specification."""
    def __init__(self, nchannels: int = 1, **kwargs):
        if not isinstance(nchannels, int):
            raise ValueError("Acquisition spec parameter: `nchannels` must be an integer")
        if nchannels < 1:
            raise ValueError("`nchannels` must be 1 or greater")
        self.nchannels = nchannels


class Acquisition(Worker):
    """
    Dirigo interface for data acquisition worker thread.
    """
    REQUIRED_RESOURCES: list[str] = None
    SPEC_LOCATION: str = None
    SPEC_OBJECT: AcquisitionSpec
    
    def __init__(self, hw: Hardware, spec: AcquisitionSpec):
        super().__init__() # sets up Publisher and inbox (Queue) for this thread
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
    def get_specification(cls, spec_name: str = "default"):
        """Return the associated Specification object"""
        spec_fn = spec_name + ".toml"
        spec = load_toml(cls.SPEC_LOCATION / spec_fn)
        return cls.SPEC_OBJECT(**spec)
    
    # Note that there is an abstractmethod, run() in the parent class. 
    # Acquisition subclasses must implement it.

