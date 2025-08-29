from typing import TYPE_CHECKING, Type, List, Optional
from collections.abc import Mapping, Sequence
from pathlib import Path
import uuid

import numpy as np

from dirigo.sw_interfaces.worker import Worker, Product
from dirigo.hw_interfaces.hw_interface import HardwareInterface
from dirigo.components.io import load_toml, SystemConfig
from dirigo.components.units import RangeWithUnits

if TYPE_CHECKING:
    from dirigo.components.hardware import Hardware#, NotConfiguredError, HardwareError
from dirigo.components.hardware import NotConfiguredError, HardwareError


class AcquisitionSpec:
    """Base class for an acquisition specification ('spec')."""

    def to_dict(self, skip: Optional[list[str]] = None) -> dict:
        """Return a JSON-ready dict containing *public* instance fields."""
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
            if not name.startswith("_") and (not skip or name not in skip)
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
        super().__init__(pool, data)
        self.timestamps = timestamps
        self.positions = positions
        self.data: np.ndarray


class AcquisitionWorker(Worker):
    Product = AcquisitionProduct

    def __init__(self, thread_name: str):
        super().__init__(thread_name)
        
        # Set the run ID -- note that there could be a collision of 2 or more
        # AcquisitionWorkers streams converge at a downstream Worker.
        self._dirigo_run_id = str(uuid.uuid4())

    def _get_free_product(self) -> AcquisitionProduct:
        return super()._get_free_product()


class Acquisition(AcquisitionWorker):
    """
    Dirigo interface for data acquisition worker thread.
    """
    required_resources: List[Type[HardwareInterface]] = [] # The first object should be the data capture device (digitizer, camera, etc)
    optional_resources: List[Type[HardwareInterface]] = []
    spec_location: Path
    Spec: Type[AcquisitionSpec] = AcquisitionSpec
    
    def __init__(self, 
                 hw: "Hardware", 
                 system_config: SystemConfig, 
                 spec: AcquisitionSpec, 
                 thread_name: str = "acquisition"):
        super().__init__(thread_name)
        self.hw = hw
        self.system_config = system_config
        self._check_resources()
        self.spec = spec

    def _check_resources(self) -> None:
        # Required: fail if not configured or if instantiation fails
        for iface in self.required_resources:
            try:
                _ = self._instantiate(iface)
            except NotConfiguredError as exc:
                raise RuntimeError(
                    f"Required {iface.__name__} not configured: {exc}"
                ) from exc
            except HardwareError as exc:
                raise RuntimeError(
                    f"Required {iface.__name__} failed to initialize: {exc}"
                ) from exc
        
        # Optional: ignore “not configured”, surface real failures
        for iface in self.optional_resources:
            try:
                _ = self._instantiate(iface)
            except NotConfiguredError:
                continue
            except HardwareError as exc:
                raise RuntimeError(
                    f"Optional {iface.__name__} failed to initialize: {exc}"
                ) from exc
            
    def _instantiate(self, iface: Type[HardwareInterface]):
        attr = iface.attr()
        try:
            return getattr(self.hw, attr) # triggers lazy instantiation
        except AttributeError:
            # typo in iface.attr(), or Hardware missing the property
            raise RuntimeError(f"Hardware has no attribute '{attr}' for {iface.__name__}")
            
    @classmethod
    def get_specification(cls, spec_name: str = "default") -> AcquisitionSpec:
        """Return the associated Specification object"""
        spec_fn = spec_name + ".toml"
        spec = load_toml(cls.spec_location / spec_fn)
        return cls.Spec(**spec)
    
    @property
    def data_acquisition_device(self):
        """
        Returns handle to the hardware resource actually acquiring data.

        Either class Digitizer, FrameGrabber, or Camera
        """
        acq_device = self.required_resources[0].__name__
        if acq_device == "Digitizer":
            return self.hw.digitizer
        elif acq_device == "FrameGrabber":
            return self.hw.frame_grabber
        elif acq_device == "Camera":
            return self.hw.camera
        else:
            raise RuntimeError(f"Invalid data capture device: {acq_device}")
    

class Loader(AcquisitionWorker):
    """
    Loads and makes available saved data for post-hoc analyses. Mimics the 
    publishing behavior of Acquisition.
    """
    def __init__(self, 
                 file_path: str | Path, 
                 thread_name: str = "loader"):
        super().__init__(thread_name)

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find file: {file_path}")
        self._file_path = file_path