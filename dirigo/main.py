import importlib.metadata
from pathlib import Path

from platformdirs import user_config_dir

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces import Acquisition, Processor, Display, Logger
from dirigo.sw_interfaces.acquisition import AcquisitionSpec
from dirigo.sw_interfaces.display import DisplayPixelFormat
from dirigo.plugins.processors import RasterFrameProcessor
from dirigo.plugins.displays import FrameDisplay
from dirigo.plugins.loggers import TiffLogger



class Dirigo:
    """
    Dirigo is an API and collection of interfaces for customizable image data 
    acquisition. 
    """

    def __init__(self):
        # Retrieve system_config.toml
        config_dir = Path(user_config_dir(appauthor="Dirigo", appname="Dirigo"))
        if not config_dir.exists():
            raise FileNotFoundError(
                f"Could not find system configuration file. "
                f"Expected to find directory: {config_dir}"
            )
        self.sys_config = SystemConfig.from_toml(config_dir / "system_config.toml")

        self.hw = Hardware(self.sys_config) 

    @property
    def acquisition_types(self) -> set[str]:
        """Returns a set of the available acquisition types."""
        entry_pts = importlib.metadata.entry_points(group="dirigo_acquisitions")
        return {entry_pt.name for entry_pt in entry_pts}
    
    def acquisition_factory(self, 
                            type: str, 
                            spec: AcquisitionSpec = None, 
                            spec_name: str = "default") -> Acquisition:
        """Returns an initialized acquisition worker object."""

        # Dynamically load plugin class
        entry_pts = importlib.metadata.entry_points(group="dirigo_acquisitions")

        # Look for the specified plugin by name
        for entry_pt in entry_pts:
            if entry_pt.name == type:
                # Load and instantiate the plugin class
                try:
                    plugin_class: Acquisition = entry_pt.load()
                    
                    # Get the acquisition specification if necessary 
                    if spec:
                        pass
                    else:
                        spec = plugin_class.get_specification(spec_name)

                    # Instantiate and return the acquisition worker
                    return plugin_class(self.hw, spec)  
                
                except Exception as e:
                    raise RuntimeError(f"Failed to load Acquisition '{type}': {e}")
                                
        # If the plugin was not found, raise an error
        raise ValueError(
            f"Acquisition '{type}' not found in entry points."
        )
    
    def processor_factory(self, 
                          upstream_worker: Acquisition | Processor, 
                          auto_connect = True) -> Processor:
        # Dynamically load plugin class
        #entry_pts = importlib.metadata.entry_points(group="dirigo_processors")
        #TODO finish entry point loading

        processor = RasterFrameProcessor(upstream_worker)

        if auto_connect:
            upstream_worker.add_subscriber(processor)

        return processor
    
    def display_factory(self, 
                        upstream_worker: Acquisition | Processor, 
                        display_pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
                        auto_connect: bool = True
                        ) -> Display:
        
        display = FrameDisplay(upstream_worker, display_pixel_format)

        if auto_connect:
            upstream_worker.add_subscriber(display)

        return display
    
    def logger_factory(self, 
                       upstream_worker: Acquisition | Processor,
                       auto_connect: bool = True) -> Logger:
        logger = TiffLogger(upstream_worker)

        if auto_connect:
            upstream_worker.add_subscriber(logger)

        return logger
    
    def acquisition_spec(self, acquisition_type: str, spec_name: str = "default"):
        # Dynamically load plugin class
        entry_pts = importlib.metadata.entry_points(group="dirigo_acquisitions")

        # Look for the specified plugin by name
        for entry_pt in entry_pts:
            if entry_pt.name == acquisition_type:
                # Load and instantiate the plugin class
                try:
                    plugin_class: Acquisition = entry_pt.load()
                    
                    # Get the acquisition specification
                    return plugin_class.get_specification(spec_name)

                except Exception as e:
                    raise RuntimeError(f"Failed to load Acquisition '{type}': {e}")
                                
        # If the plugin was not found, raise an error
        raise ValueError(
            f"Acquisition '{type}' not found in entry points."
        )
    

# Run this as a test and to help debug
if __name__ == "__main__":

    diri = Dirigo()
    
    acquisition = diri.acquisition_factory('frame')
    # processor = diri.processor_factory(acquisition)
    # display = diri.display_factory(processor)
    # logging = diri.logger_factory(processor)
    # logging.frames_per_file = float('inf')    

    # processor.start()
    # display.start()
    # logging.start()
    acquisition.start()

    acquisition.join(timeout=100.0)

    print("Acquisition complete")
