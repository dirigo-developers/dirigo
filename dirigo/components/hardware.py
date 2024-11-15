
from dirigo.components.io import SystemConfig
from dirigo.components.scanner import Scanner
from dirigo.components.digitizer import Digitizer


class Hardware:
    """ 
    Object to hold hardware components. There are no methods.
    """
    # Note: the point of this class is to hold several 
    # hardware references, not to implement functionality
    def __init__(self, default_config:SystemConfig):
        self.scanner = Scanner(default_config.scanner)
        self.digitizer