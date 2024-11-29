from pathlib import Path

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware


class Dirigo:
    """ Main coordinator """

    def __init__(self):
        path = Path(__file__).parent.parent / "config/system_config.toml"
        self.sys_config = SystemConfig.from_toml(path)
        self.hw = Hardware(self.sys_config)


    def prepare_acquisition(self):
        pass

        
if __name__ == "__main__":

    diri = Dirigo()
