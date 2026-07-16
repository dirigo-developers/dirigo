from abc import abstractmethod

from dirigo.hw_interfaces.hw_interface import HardwareInterface


class Shutter(HardwareInterface):
    """
    Dirigo shutter interface
    """
    attr_name = "shutter"
    
    def __init__(self):
        pass

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def is_open(self) -> bool:
        ...
