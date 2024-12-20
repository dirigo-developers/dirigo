from pathlib import Path
import queue

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces import Acquisition

from dirigo.sw_interfaces.acquisition import LineAcquisition, LineAcquisitionSpec

class Dirigo:
    """ Main coordinator """

    def __init__(self):
        path = Path(__file__).parent.parent / "config/system_config.toml"
        self.sys_config = SystemConfig.from_toml(path)
        self.hw = Hardware(self.sys_config) # TODO, should this be 'resources'?

        self.data_queue = queue.Queue() # TODO, probably needs to be reset periodically


    def prepare_acquisition(self, type) -> Acquisition:

        LineAcquisition(self.hw, self.data_queue)

        
if __name__ == "__main__":

    diri = Dirigo()
    a=None
