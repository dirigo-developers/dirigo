from typing import TYPE_CHECKING
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from dirigo.sw_interfaces.worker import Worker, Product
from dirigo.components.io import load_toml, SystemConfig
from dirigo.components.units import RangeWithUnits

if TYPE_CHECKING:
    from dirigo.components.hardware import Hardware 



class AcquisitionSpec:
    """Base class for an acquisition specification ('spec')."""
    def to_dict(self) -> dict:
        """Return a JSON‑ready dict containing *public* instance fields."""
        def make_jsonable(obj):
            if obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            if isinstance(obj, RangeWithUnits):
                return {
                    "min": make_jsonable(obj.min),
                    "max": make_jsonable(obj.max),
                }
            if isinstance(obj, Mapping):
                return {k: make_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
                return [make_jsonable(v) for v in obj]
            return str(obj)                 # fallback

        return {
            name: make_jsonable(val)
            for name, val in self.__dict__.items()
            if not name.startswith("_")
        }


class AcquisitionProduct(Product):
    """
    Container for acquisition's product(s): (raw) data, timestamps (optional), 
    and positions (optional).
    
    Automatically returns itself to acquisition product pool when released by 
    all subscribing consumers (functionality of the Product base class).
    """
    # data: np.ndarray # Dimensions: Record, Sample, Channel
    # timestamps: float | np.ndarray | None = None # should be one or more time points (in seconds since the start)
    # positions: tuple[float] | np.ndarray | None = None # should be one or more sets of coordinates (x,y)
    __slots__ = ("data", "timestamps", "positions")
    def __init__(self, pool, data: np.ndarray, timestamps = None, positions = None):
        super().__init__(pool)
        self.data = data
        self.timestamps = timestamps
        self.positions = positions


class AcquisitionWorker(Worker):
    def init_product_pool(self, n, shape, dtype):
        for _ in range(n):
            acquisition_product = AcquisitionProduct(
                pool=self._product_pool,
                data=np.empty(shape, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(acquisition_product)

    def get_free_product(self) -> AcquisitionProduct:
        return self._product_pool.get()


class Acquisition(AcquisitionWorker):
    """
    Dirigo interface for data acquisition worker thread.
    """
    REQUIRED_RESOURCES: list[str] = None # The first object in the list should be the data capture device (digitizer, camera, etc)
    SPEC_LOCATION: str = None
    SPEC_OBJECT: AcquisitionSpec
    
    def __init__(self, 
                 hw: 'Hardware', 
                 system_config: SystemConfig, 
                 spec: AcquisitionSpec, 
                 thread_name: str = "Acquisition"):
        super().__init__(thread_name)
        self.hw = hw
        self.system_config = system_config
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
    
    # ! Acquisition subclasses must implement a run() method.

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
    

class Loader(AcquisitionWorker):
    """
    Loads and makes available saved data for post-hoc analyses. Mimics the 
    publishing behavior of Acquisition.
    """
    def __init__(self, file_path: str | Path, thread_name: str = "Loader"):
        super().__init__(thread_name)

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileExistsError(f"Could not find file: {file_path}")
        self._file_path = file_path