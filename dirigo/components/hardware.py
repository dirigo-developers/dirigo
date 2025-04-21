import importlib.metadata
from typing import Optional

from dirigo.components.io import SystemConfig
from dirigo.components.optics import LaserScanningOptics, CameraOptics
from dirigo.hw_interfaces.detector import DetectorSet
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
    # Note: the point of this class is to hold several 
    # hardware references, not to implement functionality
    def __init__(self, default_config: SystemConfig):
        if default_config.laser_scanning_optics is not None:
            self.laser_scanning_optics = LaserScanningOptics(
                **default_config.laser_scanning_optics
            )
        else:
            self.laser_scanning_optics = None
        
        if default_config.camera_optics is not None:
            self.camera_optics = CameraOptics(**default_config.camera_optics)
        else:
            self.camera_optics = None

        if default_config.detectors is not None:
            self.detectors = DetectorSet()
            for _, detector_config in default_config.detectors.items():
                detector = self._try_instantiate(
                    group="dirigo_detectors",
                    config=detector_config
                )
                self.detectors.append(detector)
        else:
            self.detectors = None

        self.digitizer: Digitizer = self._try_instantiate(
            group="dirigo_digitizers",
            config=default_config.digitizer
        )

        self.stage: MultiAxisStage = self._try_instantiate(
            group="dirigo_stages",
            config=default_config.stage
        )

        # motorized objective is considered a 'scanner' because it move the beam 
        # through the sample, as opposed to moving the sample (stage).
        self.objective_scanner: ObjectiveZScanner = self._try_instantiate(
            group="dirigo_scanners", 
            config=default_config.objective_scanner
        )

        self.encoders: MultiAxisLinearEncoder = self._try_instantiate(
            group="dirigo_encoders",
            config=default_config.encoders
        )

        self.fast_raster_scanner: FastRasterScanner = self._try_instantiate(
            group="dirigo_scanners",
            config=default_config.fast_raster_scanner
        )

        self.slow_raster_scanner: SlowRasterScanner = self._try_instantiate(
            group="dirigo_scanners",
            config=default_config.slow_raster_scanner,
            extra_args={"fast_scanner": self.fast_raster_scanner}
        )

        self.frame_grabber: FrameGrabber = self._try_instantiate(
            group="dirigo_frame_grabbers",
            config=default_config.frame_grabber
        )

        # frame grabber must be instantiated before line scan camera
        self.line_scan_camera: LineScanCamera = self._try_instantiate(
            group="dirigo_line_scan_cameras",
            config=default_config.line_scan_camera,
            extra_args={"frame_grabber": self.frame_grabber}
        )
        
        self.illuminator: Illuminator = self._try_instantiate(
            group="dirigo_illuminators",
            config=default_config.illuminator
        )



    def _try_instantiate(self, group: str, config: Optional[dict], extra_args=None):
        if config is None:
            return None
        
        if extra_args is None:
            extra_args = {}

        entry_pts = importlib.metadata.entry_points(group=group)
        for entry_point in entry_pts:
            if entry_point.name.lower() == config['type'].lower():
                # Dynamically load and return the plugin class
                ConcreteClass = entry_point.load()
                return ConcreteClass(**{**config, **extra_args})
            
        raise ValueError(f"No {group} plugin found for: {config['type']}")
    
    @property
    def nchannels_enabled(self) -> int:
        """
        Returns the number channels currently enabled on the primary data 
        acquisition device.
        """
        # TODO is this a good strategy? Are other devices multichannel?
        if hasattr(self, 'digitizer'):
            return sum([channel.enabled for channel in self.digitizer.channels])
        elif hasattr(self, 'camera'):
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            return 0 # or raise error?
        
    @property
    def nchannels_present(self) -> int:
        """
        Returns the number channels present on the primary data acquisition 
        device.
        """
        # TODO is this a good strategy? Are other devices multichannel?
        if hasattr(self, 'digitizer'):
            return len(self.digitizer.channels)
        elif hasattr(self, 'camera'):
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            return 0 # or raise error?
        
