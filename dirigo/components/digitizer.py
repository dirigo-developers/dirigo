from abc import ABC, abstractmethod

"""
Dirigo digitizer interface

Plugin modules for digitizers must implement concrete versions of classes:
SampleClock
Trigger
Channel
Digitizer

Finally, the Digitizer subclass must be registered as a plugin in PluginRegistry.

"""

class SampleClock(ABC):
    """Sample clock defining: source, rate, edge"""
    @property
    @abstractmethod
    def source(self):
        pass

    @source.setter
    @abstractmethod
    def source(self, value):
        pass

    @property
    @abstractmethod
    def rate(self):
        pass

    @rate.setter
    @abstractmethod
    def rate(self, value):
        pass

    @property
    @abstractmethod
    def edge(self):
        pass

    @edge.setter
    @abstractmethod
    def edge(self, value):
        pass
    

class Trigger(ABC):
    @property
    @abstractmethod
    def source(self):
        pass
    
    @source.setter
    @abstractmethod
    def source(self, value):
        pass

    @property
    @abstractmethod
    def slope(self):
        pass
    
    @slope.setter
    @abstractmethod
    def slope(self, value):
        pass

    @property
    @abstractmethod
    def level(self):
        pass
    
    @level.setter
    @abstractmethod
    def level(self, value):
        pass

    @property
    @abstractmethod
    def external_coupling(self):
        pass
    
    @external_coupling.setter
    @abstractmethod
    def external_coupling(self, value):
        pass

    @property
    @abstractmethod
    def external_range(self):
        pass
    
    @external_range.setter
    @abstractmethod
    def external_range(self, value):
        pass


class Channel(ABC):
    @property
    @abstractmethod
    def coupling(self):
        pass

    @coupling.setter
    @abstractmethod
    def coupling(self, value):
        pass

    @property
    @abstractmethod
    def impedance(self):
        pass

    @impedance.setter
    @abstractmethod
    def impedance(self, value):
        pass

    @property
    @abstractmethod
    def range(self):
        pass

    @range.setter
    @abstractmethod
    def range(self, value):
        pass


class AuxillaryIO(ABC):
    # Many boards have Aux IO capabililities, but not clear needs to be a requirement
    @abstractmethod
    def configure_mode(self, mode, **kwargs):
        pass

    @abstractmethod
    def read_input(self) -> bool:
        pass

    @abstractmethod
    def write_output(self, state:bool):
        pass


class Digitizer(ABC):
    """ Abstract digitizer class. """
    def __init__(self):
        self.sample_clock:SampleClock
        self.trigger:Trigger
        self.channels:list[Channel]
        self.aux_io:AuxillaryIO

    # runtime methods