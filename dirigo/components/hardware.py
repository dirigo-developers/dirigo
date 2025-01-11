import importlib.metadata
from dataclasses import dataclass

from dirigo.components.io import SystemConfig
from dirigo.components.optics import LaserScanningOptics
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.stage import MultiAxisStage
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder
from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner


@dataclass
class ValueRange:
    min: int
    max: int

    @property
    def range(self) -> int:
        return self.max - self.min


class Hardware:
    """ 
    Loads hardware specified in system_config.toml and holds references to the 
    components.
    """
    # Note: the point of this class is to hold several 
    # hardware references, not to implement functionality
    def __init__(self, default_config:SystemConfig):
        self.optics = LaserScanningOptics(**default_config.optics)

        self.digitizer: Digitizer = self.get_hardware_plugin(
            group="dirigo_digitizers",
            default_config=default_config.digitizer
        )

        # self.stage: MultiAxisStage = self.get_hardware_plugin(
        #     group="dirigo_stages",
        #     default_config=default_config.stage
        # )

        self.encoders: MultiAxisLinearEncoder = self.get_hardware_plugin(
            group="dirigo_encoders",
            default_config=default_config.encoders
        )

        self.fast_raster_scanner: FastRasterScanner = self.get_hardware_plugin(
            group="dirigo_scanners",
            default_config=default_config.fast_raster_scanner
        )

        self.slow_raster_scanner: SlowRasterScanner = self.get_hardware_plugin(
            group="dirigo_scanners",
            default_config=default_config.slow_raster_scanner
        )


    def get_hardware_plugin(self, group, default_config):
        entry_pts = importlib.metadata.entry_points(group=group)

        for entry_point in entry_pts:
            if entry_point.name.lower() == default_config['type'].lower():
                # Dynamically load and return the plugin class
                ConcreteClass = entry_point.load()
                return ConcreteClass(**default_config)
        raise ValueError(f"No {group} plugin found for: {default_config['type']}")
    
    @property
    def nchannels(self) -> int:
        """Returns the number channels present on the primary data acquisitoin 
        device.
        """
        # TODO is this a good strategy? Are other devices multichannel?
        if hasattr(self, 'digitizer'):
            return len(self.digitizer.channels)
        elif hasattr(self, 'camera'):
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            return 0 # or raise error?
        
    @property
    def data_range(self) -> ValueRange:
        """Returns the range: min (inclusive) - max (exclusive)"""
        if hasattr(self, 'digitizer'):
            bytes_per_sample = (self.digitizer.bit_depth-1) // 8 + 1
            return ValueRange( 
                min=0,
                max=2**(8*bytes_per_sample) # This is assuming that all digitizers use the most significant bits
            )
        elif hasattr(self, 'camera'):
            return 1 # monochrome only for now, but RGB cameras should be 3-channel
        else:
            return 0 # or raise error?
