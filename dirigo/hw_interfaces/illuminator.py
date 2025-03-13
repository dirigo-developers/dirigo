from abc import ABC, abstractmethod



class Illuminator(ABC):
    """
    Dirigo illuminator interface
    """
    def __init__(self):
        pass

    @abstractmethod
    def turn_on(self):
        pass

    @abstractmethod
    def turn_off(self):
        pass
    
    @property
    @abstractmethod
    def intensity(self):
        pass

    @intensity.setter
    @abstractmethod
    def intensity(self, new_value): # settable current/intensity
        pass