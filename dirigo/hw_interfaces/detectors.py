from abc import ABC, abstractmethod
import typing




class Detector(ABC):
    def __init__(self):
        self.index: int # the detector number (0-indexed)

    @property
    @abstractmethod
    def enabled(self) -> bool:
        pass

    @enabled.setter
    @abstractmethod
    def enabled(self, state: bool):
        pass

    @property
    @abstractmethod
    def gain(self) -> typing.Any: # not obvious that there is a type hint
        """The switchable gain, if available.
        
        If the gain is not switchable, then setter should raise 
        NotImplementedError.
        """
        pass

    @gain.setter
    @abstractmethod
    def gain(self, gain):
        pass

    @property
    @abstractmethod
    def bandwidth(self) -> typing.Any: # not obvious that there is a type hint
        """The switchable bandwidth, if available.
        
        If the bandwidth is not switchable, then setter should raise 
        NotImplementedError.
        """
        pass

    @bandwidth.setter
    @abstractmethod
    def bandwidth(self, bandwidth):
        pass



# TODO, subclass list, add a get item
class Detectors(ABC):
    def __init__(self):
        self # How to make this [slice-able]?