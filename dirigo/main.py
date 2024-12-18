from pathlib import Path
import queue

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware
#from dirigo.interfaces import Acquisition

class Dirigo:
    """ Main coordinator """

    def __init__(self):
        path = Path(__file__).parent.parent / "config/system_config.toml"
        self.sys_config = SystemConfig.from_toml(path)
        self.hw = Hardware(self.sys_config)

        self.data_queue = queue.Queue()


    # def prepare_acquisition(self, type) -> Acquisition:
    #     self.data_queue

        
if __name__ == "__main__":

    diri = Dirigo()
    a=1
