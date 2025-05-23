from functools import cached_property
import importlib.metadata as im
from typing import Optional

from dirigo.components.io import SystemConfig
from dirigo.components.optics import LaserScanningOptics, CameraOptics
from dirigo.hw_interfaces.detector import DetectorSet, Detector
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.stage import MultiAxisStage
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder
from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner, ObjectiveZScanner
from dirigo.hw_interfaces.camera import FrameGrabber, LineScanCamera
from dirigo.hw_interfaces.illuminator import Illuminator



class Hardware:
    """ 
    Loads hardware specified in system_config.toml and holds references to the 
    components.
    """
    def __init__(self, system_config: SystemConfig) -> None:
        self._cfg = system_config

    # --- helper ---
    def _load(self, group: str, type_name: str, **kw):
        try:
            cls = im.entry_points(group=group)[type_name.lower()].load()
            return cls(**kw)
        except KeyError as e:
            raise ValueError(f"No plugin '{type_name}' in entry-point group "
                             f"'{group}'.") from e
        
    # --- hardward devices (lazy loading) ---
    @cached_property
    def digitizer(self) -> Digitizer | None:
        cfg = self._cfg.digitizer
        return None if cfg is None else self._load(
            "dirigo_digitizers", cfg["type"], **cfg)
    
    @cached_property
    def fast_raster_scanner(self) -> FastRasterScanner | None:
        cfg = self._cfg.fast_raster_scanner
        return None if cfg is None else self._load(
            "dirigo_scanners", cfg["type"], **cfg)

    @cached_property
    def slow_raster_scanner(self) -> SlowRasterScanner | None:
        cfg = self._cfg.slow_raster_scanner
        return None if cfg is None else self._load(
            "dirigo_scanners", cfg["type"],
            fast_scanner=self.fast_raster_scanner, **cfg)
    
    @cached_property
    def objective_z_scanner(self) -> ObjectiveZScanner | None:
        cfg = self._cfg.objective_z_scanner
        return None if cfg is None else self._load(
            "dirigo_scanners", cfg["type"], **cfg)

    @cached_property
    def stages(self) -> MultiAxisStage | None:
        cfg = self._cfg.stages
        return None if cfg is None else self._load(
            "dirigo_stages", cfg["type"], **cfg)
    
    @cached_property
    def encoders(self) -> MultiAxisLinearEncoder | None:
        cfg = self._cfg.encoders
        return None if cfg is None else self._load(
            "dirigo_encoders", cfg["type"], **cfg)

    @cached_property
    def detectors(self) -> Optional[DetectorSet[Detector]]:
        """Instantiate all detectors listed in the system-config file.

        Returns
        -------
        DetectorSet | None
            • A DetectorSet (possibly empty) if a [detectors] table exists  
            • None if the system-config omits the section entirely
        """
        cfg = self._cfg.detectors                         # a *table* or None
        if cfg is None:
            return None

        dset = DetectorSet()
        for key, det_cfg in cfg.items():                  # preserves order in TOML
            det = self._load("dirigo_detectors",
                             det_cfg["type"],
                             **det_cfg,
                             fast_scanner=self.fast_raster_scanner)
            dset.append(det)

        return dset

    @cached_property
    def frame_grabber(self) -> FrameGrabber | None:
        cfg = self._cfg.frame_grabber
        return None if cfg is None else self._load(
            "dirigo_frame_grabbers", cfg["type"], **cfg)
    
    @cached_property
    def line_scan_camera(self) -> LineScanCamera | None:
        cfg = self._cfg.line_scan_camera
        return None if cfg is None else self._load(
            "dirigo_line_scan_cameras", cfg["type"], 
            frame_grabber=self.frame_grabber, **cfg)
    
    @cached_property
    def illuminator(self) -> Illuminator | None:
        cfg = self._cfg.illuminator
        return None if cfg is None else self._load(
            "dirigo_illuminators", cfg["type"], **cfg)

    @cached_property
    def laser_scanning_optics(self) -> LaserScanningOptics | None:
        cfg = self._cfg.laser_scanning_optics
        return LaserScanningOptics(**cfg) if cfg else None

    @cached_property
    def camera_optics(self) -> CameraOptics | None:
        cfg = self._cfg.camera_optics
        return CameraOptics(**cfg) if cfg else None
        
    # --- conveniences ---
    @property
    def nchannels_enabled(self) -> int:
        """
        Returns the number channels currently enabled on the primary data 
        acquisition device.
        """
        if self._cfg.digitizer:
            return sum([channel.enabled for channel in self.digitizer.channels])
        elif self._cfg.line_scan_camera:
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            return None
        
    @property
    def nchannels_present(self) -> int:
        """
        Returns the number channels present on the primary data acquisition 
        device.
        """
        if self._cfg.digitizer:
            return len(self.digitizer.channels)
        elif self._cfg.line_scan_camera:
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            return None
        
