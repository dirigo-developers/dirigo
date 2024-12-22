import importlib.metadata

from dirigo.components.io import SystemConfig
from dirigo.components.scanner import Scanner
from dirigo.components.optics import LaserScanningOptics
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.stage import Stage
from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner



class Hardware:
    """ 
    Object to hold hardware components.
    """
    # Note: the point of this class is to hold several 
    # hardware references, not to implement functionality
    def __init__(self, default_config:SystemConfig):
        self.scanner = Scanner(default_config.scanner) # OBSOLETE
        self.optics = LaserScanningOptics(**default_config.optics)

        self.digitizer:Digitizer = self.get_hardware_plugin(
            group="dirigo_digitizers",
            default_config=default_config.digitizer
        )

        self.stage:Stage = self.get_hardware_plugin(
            group="dirigo_stages",
            default_config=default_config.stage
        )

        self.fast_raster_scanner:FastRasterScanner = self.get_hardware_plugin(
            group="dirigo_scanners",
            default_config=default_config.fast_raster_scanner
        )

        self.slow_raster_scanner:SlowRasterScanner = self.get_hardware_plugin(
            group="dirigo_scanners",
            default_config=default_config.slow_raster_scanner
        )

    def get_hardware_plugin(self, group, default_config):
        entry_pts = importlib.metadata.entry_points(group=group)

        for entry_point in entry_pts:
            if entry_point.name.lower() == default_config['type'].lower():
                # Dynamically load and return the plugin class
                ConcreteClass = entry_point.load()
                return ConcreteClass(default_config)
        raise ValueError(f"No {group} plugin found for: {default_config['type']}")
