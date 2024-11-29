import importlib.metadata

from dirigo.components.io import SystemConfig
from dirigo.components.scanner import Scanner


class Hardware:
    """ 
    Object to hold hardware components.
    """
    # Note: the point of this class is to hold several 
    # hardware references, not to implement functionality
    def __init__(self, default_config:SystemConfig):
        self.scanner = Scanner(default_config.scanner)

        self.digitizer = self.get_hardware_plugin(
            group="dirigo_digitizers",
            default_config=default_config.digitizer
        )

        self.stage = self.get_hardware_plugin(
            group="dirigo_stages",
            default_config=default_config.stage
        )

    def get_hardware_plugin(self, group, default_config):
        mnfr_name = default_config.manufacturer

        entry_points = importlib.metadata.entry_points(group=group)
        for entry_point in entry_points:
            if entry_point.name.lower() == mnfr_name.lower():
                # Dynamically load and return the plugin class
                ConcreteClass = entry_point.load()
                return ConcreteClass()
        raise ValueError(f"No {group} plugin found for: {mnfr_name}")
