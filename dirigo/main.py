from pathlib import Path
import queue

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces import Acquisition

from dirigo.sw_interfaces.acquisition import LineAcquisition, LineAcquisitionSpec



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


    def acquisition_factory(self, type: str) -> Acquisition:
        """Returns an initialized acquisition worker object."""
        self._flush_queue()

        # Get the Acquisition specification
        spec = LineAcquisitionSpec(
            bidirectional_scanning=False,
            line_width=2e-3,
            pixel_size=2e-6,
            fill_fraction=0.9,
            lines_per_buffer=256,
            buffers_per_acquisition=4,
            buffers_allocated=4
        )

        # Initialize worker object
        acq_worker =  LineAcquisition(self.hw, self.data_queue, spec)
        return acq_worker
    
    def _flush_queue(self):
        """Remove all items from the queue."""
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        
if __name__ == "__main__":

    diri = Dirigo()
    
    acq = diri.acquisition_factory('line')

    None
