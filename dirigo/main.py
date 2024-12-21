import importlib.metadata
from pathlib import Path
import queue

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces import Acquisition



class Dirigo:
    """
    Dirigo is an API and collection of interfaces for customizable image data 
    acquisition. 
    """

    def __init__(self):
        path = Path(__file__).parent.parent / "config/system_config.toml"
        self.sys_config = SystemConfig.from_toml(path)
        self.hw = Hardware(self.sys_config) # TODO, should this be 'resources'?

        self.data_queue = queue.Queue() # TODO, probably needs to be reset periodically


    def acquisition_factory(self, type: str, spec_name: str = "default") -> Acquisition:
        """Returns an initialized acquisition worker object."""
        self._flush_queue()

        # Dynamically load plugin class
        entry_pts = importlib.metadata.entry_points(group="dirigo_acquisitions")

        # Look for the specified plugin by name
        for ep in entry_pts:
            if ep.name == type:
                # Load and instantiate the plugin class
                try:
                    plugin_class:Acquisition = ep.load()
                    # Instantiate and return the acquisition worker
                    spec = plugin_class.get_specification(spec_name)
                    return plugin_class(self.hw, self.data_queue, spec)  
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load Acquisition '{type}': {e}"
                    )
        
        # If the plugin was not found, raise an error
        raise ValueError(
            f"Acquisition '{type}' not found in entry points."
        )
    
    def _flush_queue(self):
        """Remove all items from the queue."""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        
if __name__ == "__main__":

    diri = Dirigo()
    
    acq = diri.acquisition_factory('frame')

    None
