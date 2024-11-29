from abc import ABC, abstractmethod

"""
Dirigo digitizer interface

Plugin modules for digitizers must implement concrete versions of classes:
Channel
SampleClock
Trigger
Acquire
AuxillaryIO [could be optional?]
Digitizer

Finally, the Digitizer subclass must be registered as a plugin in PluginRegistry.

"""

class ValidQuantityRange:
    def __init__(self, quantiy_min, quantiy_max):
        self.min = quantiy_min
        self.max = quantiy_max


class Channel(ABC):
    @property
    @abstractmethod
    def index(self):
        pass

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
    def coupling_options(self):
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
    def impedance_options(self):
        pass

    @property
    @abstractmethod
    def range(self):
        pass

    @range.setter
    @abstractmethod
    def range(self, value):
        pass

    @property
    @abstractmethod
    def range_options(self):
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        pass

    @enabled.setter
    @abstractmethod
    def enabled(self, value):
        pass


class SampleClock(ABC):
    """Sample clock defining: source, rate, edge"""
    @property
    @abstractmethod
    def source(self) -> str:
        pass

    @source.setter
    @abstractmethod
    def source(self, value):
        pass

    @property
    @abstractmethod
    def source_options(self) -> list[str]:
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
    def rate_options(self) -> list[str] | ValidQuantityRange:
        pass

    @property
    @abstractmethod
    def edge(self):
        pass

    @edge.setter
    @abstractmethod
    def edge(self, value):
        """Acquire sample on "Rising" or "Falling" edge of sample clock."""
        pass

    @property
    @abstractmethod
    def edge_options(self):
        pass
    

class Trigger(ABC):
    @property
    @abstractmethod
    def source(self) -> str:
        pass
    
    @source.setter
    @abstractmethod
    def source(self, value:str):
        pass

    @property
    @abstractmethod
    def source_options(self) -> list[str]:
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
    def slope_options(self):
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
    def level_min(self):
        pass

    @property
    @abstractmethod
    def level_max(self):  
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
    def external_coupling_options(self):
        pass

    @property
    @abstractmethod
    def external_range(self):
        pass
    
    @external_range.setter
    @abstractmethod
    def external_range(self, value):
        pass

    @property
    @abstractmethod
    def external_range_options(self):
        pass


class Acquire(ABC):
    def __init__(self):
        self._channels:list[Channel]

    @property
    def n_channels_enabled(self) -> int:
        return sum([c.enabled for c in self._channels])

    @property
    @abstractmethod
    def trigger_delay(self):
        pass

    @trigger_delay.setter
    @abstractmethod
    def trigger_delay(self, value):
        pass

    @property
    @abstractmethod
    def trigger_delay_resolution(self):
        pass

    @property
    @abstractmethod
    def pre_trigger_samples(self):
        pass

    @pre_trigger_samples.setter
    @abstractmethod
    def pre_trigger_samples(self, value):
        pass

    @property
    def pre_trigger_resolution(self):
        pass

    @property
    @abstractmethod
    def record_length(self):
        pass

    @record_length.setter
    @abstractmethod
    def record_length(self, value):
        pass

    @property
    @abstractmethod
    def record_length_minimum(self):
        pass

    @property
    @abstractmethod
    def record_length_resolution(self):
        pass

    @property
    @abstractmethod
    def records_per_buffer(self):
        pass

    @records_per_buffer.setter
    @abstractmethod
    def records_per_buffer(self, value):
        pass

    @property
    @abstractmethod
    def buffers_per_acquisition(self):
        pass

    @buffers_per_acquisition.setter
    @abstractmethod
    def buffers_per_acquisition(self, value):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
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
    @abstractmethod
    def __init__(self):
        self.channels:list[Channel]
        self.sample_clock:SampleClock
        self.trigger:Trigger
        self.acquire:Acquire
        self.aux_io:AuxillaryIO

    