import importlib
from pathlib import Path

from dirigo.components.io import SystemConfig
from dirigo.components.hardware import Hardware


class Dirigo:
    """ Main coordinator """

    def __init__(self):
        path = Path(__file__).parent.parent / "config/system_config.toml"
        self.sys_config = SystemConfig.from_toml(path)
        self.hw = Hardware(self.sys_config)

        self._load_plugins()


    def _load_plugins(self):
        # NOTE, it may be better to selectively load plugins if this gets large
        plugins_dir = Path(__file__).parent / 'plugins'
        for plugin_file in plugins_dir.glob('*.py'):
            module_name = f'plugins.{plugin_file.stem}'
            importlib.import_module(module_name)


if __name__ == "__main__":

    diri = Dirigo()
