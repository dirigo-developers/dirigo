from functools import lru_cache, cached_property
import importlib.metadata as im
from typing import Optional

from dirigo.components.io import SystemConfig
from dirigo.components.optics import LaserScanningOptics, CameraOptics
from dirigo.hw_interfaces.detector import DetectorSet, Detector
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.stage import MultiAxisStage
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder
from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner, ObjectiveZScanner
from dirigo.hw_interfaces.camera import FrameGrabber, LineCamera
from dirigo.hw_interfaces.illuminator import Illuminator



@lru_cache
def _eps(group: str) -> dict[str, im.EntryPoint]:
    return {ep.name.lower(): ep for ep in im.entry_points().select(group=group)}


class Hardware:
    """ 
    Loads hardware specified in system_config.toml and holds references to the 
    components.
    """
    def __init__(self, system_config: SystemConfig) -> None:
        self._cfg = system_config

    # --- helper ---
    def _load(self, group: str, type_name: str, **kw):
        """
        Lazy-load the concrete driver class registered under *type_name*
        in entry-point *group*, instantiate it with **kw, and return it.
        """
        try:
            cls = _eps(group)[type_name.lower()].load()   # import on demand
            return cls(**kw)
        except KeyError as e:
            raise ValueError(
                f"No plugin '{type_name}' in entry-point group '{group}'. "
                f"Available: {', '.join(_eps(group)) or 'none'}"
            ) from e
        
    def exists(self, resource_attribute_name: str) -> bool:
        return resource_attribute_name in self.__dict__
    
    # --- hardward devices (lazy loading) ---
    @cached_property
    def digitizer(self) -> Digitizer:
        cfg = self._cfg.digitizer
        assert cfg is not None
        return self._load("dirigo_digitizers", cfg["type"], **cfg)
    
    @cached_property
    def fast_raster_scanner(self) -> FastRasterScanner:
        cfg = self._cfg.fast_raster_scanner
        assert cfg is not None
        return self._load("dirigo_scanners", cfg["type"], **cfg)

    @cached_property
    def slow_raster_scanner(self) -> SlowRasterScanner:
        cfg = self._cfg.slow_raster_scanner
        assert cfg is not None
        return self._load("dirigo_scanners", cfg["type"],
            fast_scanner=self.fast_raster_scanner, **cfg)
    
    @cached_property
    def objective_z_scanner(self) -> ObjectiveZScanner:
        cfg = self._cfg.objective_z_scanner
        assert cfg is not None
        return self._load("dirigo_scanners", cfg["type"], **cfg)

    @cached_property
    def stages(self) -> MultiAxisStage:
        cfg = self._cfg.stages
        assert cfg is not None
        return self._load("dirigo_stages", cfg["type"], **cfg)
    
    @cached_property
    def encoders(self) -> MultiAxisLinearEncoder:
        cfg = self._cfg.encoders
        assert cfg is not None
        return self._load("dirigo_encoders", cfg["type"], **cfg)

    @cached_property
    def detectors(self) -> DetectorSet[Detector]:
        """Instantiate all detectors listed in the system-config file.

        Returns
        -------
        DetectorSet | None
            • A DetectorSet (possibly empty) if a [detectors] table exists  
            • None if the system-config omits the section entirely
        """
        cfg = self._cfg.detectors                         # a *table* or None
        assert cfg is not None

        dset = DetectorSet()
        for key, det_cfg in cfg.items():                  # preserves order in TOML
            det = self._load("dirigo_detectors",
                             det_cfg["type"],
                             **det_cfg,
                             fast_scanner=self.fast_raster_scanner)
            dset.append(det)

        return dset

    @cached_property
    def frame_grabber(self) -> FrameGrabber:
        cfg = self._cfg.frame_grabber
        assert cfg is not None
        return self._load("dirigo_frame_grabbers", cfg["type"], **cfg)
    
    @cached_property
    def line_camera(self) -> LineCamera: # this could just be camera
        cfg = self._cfg.line_camera
        assert cfg is not None
        return self._load("dirigo_line_cameras", cfg["type"], 
            frame_grabber=self.frame_grabber, **cfg)
    
    @cached_property
    def illuminator(self) -> Illuminator:
        cfg = self._cfg.illuminator
        assert cfg is not None
        return self._load("dirigo_illuminators", cfg["type"], **cfg)

    @cached_property
    def laser_scanning_optics(self) -> LaserScanningOptics:
        cfg = self._cfg.laser_scanning_optics
        assert cfg is not None
        return LaserScanningOptics(**cfg)

    @cached_property
    def camera_optics(self) -> CameraOptics:
        cfg = self._cfg.camera_optics
        assert cfg is not None
        return CameraOptics(**cfg)
        
    # --- conveniences ---
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
        
    # TODO: add close all method