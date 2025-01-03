import importlib.metadata
from pathlib import Path
import queue

from platformdirs import user_config_dir

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces import Acquisition, Processor, Display
from dirigo.plugins.processors import RasterFrameProcessor
from dirigo.plugins.displays import FrameDisplay



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

        self.hw = Hardware(self.sys_config) # TODO, should this be 'resources'?

        self.data_queue = queue.Queue() 
        self.processed_queue = queue.Queue()
        self.display_queue = queue.Queue()

    @property
    def acquisition_types(self) -> set[str]:
        """Returns a set of the available acquisition types."""
        entry_pts = importlib.metadata.entry_points(group="dirigo_acquisitions")
        return {entry_pt.name for entry_pt in entry_pts}
    
    def acquisition_factory(self, type: str, spec_name: str = "default") -> Acquisition:
        """Returns an initialized acquisition worker object."""
        self._flush_queue()

        # Dynamically load plugin class
        entry_pts = importlib.metadata.entry_points(group="dirigo_acquisitions")

        # Look for the specified plugin by name
        for entry_pt in entry_pts:
            if entry_pt.name == type:
                # Load and instantiate the plugin class
                try:
                    plugin_class: Acquisition = entry_pt.load()
                    
                    # Get the acquisition specification
                    spec = plugin_class.get_specification(spec_name)

                    # Instantiate and return the acquisition worker
                    return plugin_class(self.hw, self.data_queue, spec)  
                
                except Exception as e:
                    raise RuntimeError(f"Failed to load Acquisition '{type}': {e}")
                                
        # If the plugin was not found, raise an error
        raise ValueError(
            f"Acquisition '{type}' not found in entry points."
        )
    
    def processor_factory(self, acquisition: Acquisition) -> Processor:
        # Dynamically load plugin class
        #entry_pts = importlib.metadata.entry_points(group="dirigo_processors")
        #TODO finish entry point loading

        return RasterFrameProcessor(acquisition, self.processed_queue)
    
    def display_factory(self, processor: Processor = None, acquisition: Acquisition = None) -> Display:
        return FrameDisplay(self.display_queue, acquisition, processor)
    
    def _flush_queue(self):
        """Remove all items from the queue."""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        
if __name__ == "__main__":

    diri = Dirigo()
    
    acquisition = diri.acquisition_factory('frame')

    processor = diri.processor_factory(acquisition)

    display = diri.display_factory(processor)

    # TODO spawn Logging thread

    processor.start()
    display.start()
    acquisition.start()

    acquisition.join(timeout=10.0)

