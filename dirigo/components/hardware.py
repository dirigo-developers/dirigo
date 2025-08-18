from functools import lru_cache, cached_property
import importlib.metadata as im
from typing import TYPE_CHECKING

from dirigo.components.optics import LaserScanningOptics, CameraOptics
from dirigo.hw_interfaces.detector import DetectorSet, Detector

if TYPE_CHECKING:
    from dirigo.components.io import SystemConfig
    from dirigo.hw_interfaces.digitizer import Digitizer
    from dirigo.hw_interfaces.stage import MultiAxisStage
    from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder
    from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner, ObjectiveZScanner
    from dirigo.hw_interfaces.camera import FrameGrabber, LineCamera
    from dirigo.hw_interfaces.illuminator import Illuminator



@lru_cache
def _eps(group: str) -> dict[str, im.EntryPoint]:
    return {ep.name.lower(): ep for ep in im.entry_points().select(group=group)}


class HardwareError(RuntimeError): pass


class NotConfiguredError(HardwareError):
    def __init__(self, section: str):
        super().__init__(f"[{section}] missing in system_config.toml")
        self.section = section


class PluginNotFoundError(HardwareError):
    def __init__(self, group: str, type_name: str, available: list[str]):
        avail = ", ".join(available) or "none"
        super().__init__(f"No plugin '{type_name}' in entry-point group '{group}'. Available: {avail}")
        self.group, self.type_name, self.available = group, type_name, available


class PluginInitError(HardwareError):
    def __init__(self, cls, kwargs: dict):
        super().__init__(f"Failed to initialize {cls.__name__} with kwargs {kwargs}")
        self.cls, self.kwargs = cls, kwargs


class Hardware:
    """ 
    Loads hardware specified in system_config.toml and holds references to the 
    components.
    """
    def __init__(self, system_config: "SystemConfig") -> None:
        self._cfg = system_config

    # --- helper ---
    def _load(self, group: str, type_name: str, **kw):
        """
        Lazy-load the concrete driver class registered under *type_name*
        in entry-point *group*, instantiate it with **kw, and return it.
        """
        try:
            cls = _eps(group)[type_name.lower()].load()
        except KeyError as e:
            raise PluginNotFoundError(group, type_name, available=list(_eps(group)).copy()) from e
        try:
            return cls(**kw)
        except TypeError as e:
            raise PluginInitError(cls, kwargs=kw) from e
    
    # --- hardward devices (lazy loading) ---
    @cached_property
    def digitizer(self) -> "Digitizer":
        cfg = self._cfg.digitizer
        if cfg is None:
            raise NotConfiguredError("digitizer")
        type_name = cfg.pop("type")
        return self._load("dirigo_digitizers", type_name, **cfg)
    
    @cached_property
    def fast_raster_scanner(self) -> "FastRasterScanner":
        cfg = self._cfg.fast_raster_scanner
        if cfg is None:
            raise NotConfiguredError("fast raster scanner")
        type_name = cfg.pop("type")
        return self._load("dirigo_scanners", type_name, **cfg)

    @cached_property
    def slow_raster_scanner(self) -> "SlowRasterScanner":
        cfg = self._cfg.slow_raster_scanner
        if cfg is None:
            raise NotConfiguredError("slow raster scanner")
        type_name = cfg.pop("type")
        return self._load("dirigo_scanners", type_name,
            fast_scanner=self.fast_raster_scanner, **cfg)
    
    @cached_property
    def objective_z_scanner(self) -> "ObjectiveZScanner":
        cfg = self._cfg.objective_z_scanner
        if cfg is None:
            raise NotConfiguredError("objective z scanner")
        type_name = cfg.pop("type")
        return self._load("dirigo_scanners", type_name, **cfg)

    @cached_property
    def stages(self) -> "MultiAxisStage":
        cfg = self._cfg.stages
        if cfg is None:
            raise NotConfiguredError("stages")
        type_name = cfg.pop("type")
        return self._load("dirigo_stages", type_name, **cfg)
    
    @cached_property
    def encoders(self) -> "MultiAxisLinearEncoder":
        cfg = self._cfg.encoders
        if cfg is None:
            raise NotConfiguredError("encoders")
        type_name = cfg.pop("type")
        return self._load("dirigo_encoders", type_name, **cfg)

    @cached_property
    def detectors(self) -> DetectorSet[Detector]:
        cfg = self._cfg.detectors
        if cfg is None:
            raise NotConfiguredError("detectors")
        dset = DetectorSet()
        for key, det_cfg in cfg.items():
            type_name = det_cfg.pop("type")
            det = self._load("dirigo_detectors",
                             type_name,
                             **det_cfg,
                             fast_scanner=self.fast_raster_scanner)
            dset.append(det)
        return dset

    @cached_property
    def frame_grabber(self) -> "FrameGrabber":
        cfg = self._cfg.frame_grabber
        if cfg is None:
            raise NotConfiguredError("frame grabber")
        type_name = cfg.pop("type")
        return self._load("dirigo_frame_grabbers", type_name, **cfg)
    
    @cached_property
    def line_camera(self) -> "LineCamera": # this could just be camera
        cfg = self._cfg.line_camera
        if cfg is None:
            raise NotConfiguredError("line camera")
        type_name = cfg.pop("type")
        return self._load("dirigo_line_cameras", type_name, 
            frame_grabber=self.frame_grabber, **cfg)
    
    @cached_property
    def illuminator(self) -> "Illuminator":
        cfg = self._cfg.illuminator
        if cfg is None:
            raise NotConfiguredError("illuminator")
        type_name = cfg.pop("type")
        return self._load("dirigo_illuminators", type_name, **cfg)

    @cached_property
    def laser_scanning_optics(self) -> LaserScanningOptics:
        cfg = self._cfg.laser_scanning_optics
        if cfg is None:
            raise NotConfiguredError("laser scanning optics")
        #type_name = cfg.pop("type") TODO
        return LaserScanningOptics(**cfg)

    @cached_property
    def camera_optics(self) -> CameraOptics:
        cfg = self._cfg.camera_optics
        if cfg is None:
            raise NotConfiguredError("camera optics")
        # type_name = cfg.pop("type") TODO
        return CameraOptics(**cfg)
        
    # --- conveniences ---
    # TODO deprecate these methods to restore Hardware as a container/lazy loader
    @property
    def nchannels_enabled(self) -> int:
        """
        Returns the number channels currently enabled on the primary data 
        acquisition device.
        """
        if self._cfg.digitizer:
            return sum([channel.enabled for channel in self.digitizer.channels])
        elif self._cfg.line_camera:
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            raise RuntimeError("No channels available: lacking digitizer or camera")
        
    @property
    def nchannels_present(self) -> int:
        """
        Returns the number channels present on the primary data acquisition 
        device.
        """
        if self._cfg.digitizer:
            return len(self.digitizer.channels)
        elif self._cfg.line_camera:
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            raise RuntimeError("No channels available: lacking digitizer or camera")
        
    # TODO __repr__ method