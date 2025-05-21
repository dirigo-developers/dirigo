from abc import ABC, abstractmethod

from dirigo.hw_interfaces.hw_interface import HardwareInterface


class Illuminator(HardwareInterface):
    """
    Dirigo illuminator interface
    """
    attr_name = "illuminator"
    
    def __init__(self):
        pass

    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass

    @abstractmethod
    def close(self):
        pass
    
    @property
    @abstractmethod
    def intensity(self):
        pass

    @intensity.setter
    @abstractmethod
    def intensity(self, new_value): # settable current/intensity
        pass