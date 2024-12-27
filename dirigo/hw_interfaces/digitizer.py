from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

import dirigo
from dirigo.components.io import load_toml

"""
Dirigo digitizer interface.

This module defines an abstract interface for digitizers, enabling consistent 
integration of different hardware models into the Dirigo platform. Plugins 
implementing this interface should define concrete versions of the following 
classes:

- Channel: Represents individual input channels on the digitizer.
- SampleClock: Manages sampling rate, source, and clock edge settings.
- Trigger: Configures triggering parameters like source, slope, and level.
- Acquire: Controls acquisition settings and manages data buffering.
- AuxiliaryIO: Optional class for auxiliary input/output functionality.
- Digitizer: Top-level abstraction encapsulating the hardware interface.

Plugins must include a project entry point under the "dirigo_digitizers" group. 
Example configuration:

[project.entry-points."dirigo_digitizers"]
alazar = "dirigo_alazar:AlazarDigitizer"
"""
# Questions: 
# should we define a whole set of enumerations to be used with Dirigo?
# Currently uses strings (and sets of strings)
# Argument against: too many possibilities if we consider multiple vendors but 
# certain concepts like AC/DC, external/internal may be OK


class Channel(ABC):
    """Abstract base class for a digitizer's input channel configuration."""

    @property
    @abstractmethod
    def index(self) -> int:
        """Index of the channel (e.g., 0-based index)."""
        pass

    @property
    @abstractmethod
    def coupling(self) -> str:
        """Signal coupling mode (e.g., 'AC', 'DC')."""
        pass

    @coupling.setter
    @abstractmethod
    def coupling(self, coupling: str):
        """Set the signal coupling mode.
        
        Must match one of the options provided by `coupling_options`.
        """
        pass

    @property
    @abstractmethod
    def coupling_options(self) -> set[str]:
        """Set of available coupling modes."""
        pass

    @property
    @abstractmethod
    def impedance(self) -> str:
        """Input impedance setting (e.g., 50 Ohm, 1 MOhm)."""
        pass

    @impedance.setter
    @abstractmethod
    def impedance(self, impedance: str):
        """Set the input impedance.
        
        Must match one of the options provided by `impedance_options`.
        """
        pass

    @property
    @abstractmethod
    def impedance_options(self) -> set[str]:
        """Set of available input impedance modes."""
        pass

    @property
    @abstractmethod
    def range(self) -> str: 
        """Voltage range for the channel."""
        # should this return a dirigo.VoltageRange object?
        # arguments against: 
        # - digitizer input ranges are discrete options (if available at all)
        # - can't think of need for easier numerical access to range limits
        pass

    @range.setter
    @abstractmethod
    def range(self, range: str):
        """Set the voltage range for the channel.
        
        Must match one of the options provided by `range_options`.
        """
        pass

    @property
    @abstractmethod
    def range_options(self) -> set[str]:
        """Set of available voltage ranges."""
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Indicates whether the channel is enabled for acquisition."""
        pass

    @enabled.setter
    @abstractmethod
    def enabled(self, state: bool):
        """Enable or disable the channel."""
        pass


class SampleClock(ABC):
    """Abstract base class for configuring the digitizer's sampling clock."""

    @property
    @abstractmethod
    def source(self) -> str:
        """Source of the sample clock (e.g., internal, external)."""
        pass

    @source.setter
    @abstractmethod
    def source(self, source: str):
        """Set the source of the sample clock.
        
        Must match one of the options provided by `source_options`.
        """
        pass

    @property
    @abstractmethod
    def source_options(self) -> set[str]:
        """Set of supported clock sources."""
        pass

    @property
    @abstractmethod
    def rate(self) -> dirigo.Frequency:
        pass

    @rate.setter
    @abstractmethod
    def rate(self, value: dirigo.Frequency):
        pass
    
    @property
    @abstractmethod
    def rate_options(self) -> set[dirigo.Frequency]:
        pass

    @property
    @abstractmethod
    def edge(self) -> str:
        """Clock edge to use for sampling (e.g., rising, falling)."""
        pass

    @edge.setter
    @abstractmethod
    def edge(self, edge: str):
        """Set the clock edge for sampling.
        
        Must match one of the options provided by `edge_options`.
        """
        pass

    @property
    @abstractmethod
    def edge_options(self) -> set[str]:
        """Set of supported clock edge options."""
        pass
    

class Trigger(ABC):
    """Abstract base class for configuring the digitizer's trigger settings."""

    @property
    @abstractmethod
    def source(self) -> str:
        """Trigger source (e.g., internal, external)."""
        pass
    
    @source.setter
    @abstractmethod
    def source(self, source: str):
        """Set the trigger source.
        
        Must match one of the options provided by `source_options`.
        """
        pass

    @property
    @abstractmethod
    def source_options(self) -> set[str]:
        """Set of available trigger sources."""
        pass

    @property
    @abstractmethod
    def slope(self) -> str:
        """Trigger slope (e.g., rising, falling)."""
        pass
    
    @slope.setter
    @abstractmethod
    def slope(self, slope: str):
        """Set the trigger slope.
        
        Must match one of the options provided by `slope_options`.
        """
        pass

    @property
    @abstractmethod
    def slope_options(self) -> set[str]:
        """Set of supported trigger slopes."""
        pass

    @property
    @abstractmethod
    def level(self) -> dirigo.Voltage:
        """Trigger level in volts."""
        pass
    
    @level.setter
    @abstractmethod
    def level(self, level: dirigo.Voltage):
        """Set the trigger level."""
        pass

    @property
    @abstractmethod
    def level_limits(self) -> dirigo.VoltageRange:
        """Returns an object describing the supported trigger level range."""
        pass

    @property
    @abstractmethod
    def external_coupling(self) -> str:
        """Coupling mode for external trigger sources."""
        pass
    
    @external_coupling.setter
    @abstractmethod
    def external_coupling(self, coupling: str):
        """Set the coupling mode for external triggers.
        
        Must match one of the options provided by `external_coupling_options`.
        """
        pass

    @property
    @abstractmethod
    def external_coupling_options(self) -> set[str]:
        """Set of available external coupling modes."""
        pass

    @property
    @abstractmethod
    def external_range(self) -> str:
        """Voltage range for external triggers."""
        pass
    
    @external_range.setter
    @abstractmethod
    def external_range(self, range: str):
        """Set the voltage range for external triggers.
        
        Must match one of the options provided by `external_range_options`.
        """
        pass

    @property
    @abstractmethod
    def external_range_options(self) -> set[str]:
        """Set of available external trigger voltage ranges."""
        pass


class Acquire(ABC):
    """Abstract base class for managing digitizer acquisitions and buffering."""

    def __init__(self):
        self._channels: list[Channel]

    @property
    def n_channels_enabled(self) -> int:
        """Number of channels enabled for acquisition."""
        return sum([c.enabled for c in self._channels])

    @property
    @abstractmethod
    def trigger_delay(self) -> dirigo.Time:
        """Delay between trigger event and acquisition start."""
        # Arguably could be part of Trigger object, but put here because of role
        # in acquisition timing
        pass

    @trigger_delay.setter
    @abstractmethod
    def trigger_delay(self, delay: dirigo.Time):
        """Set the trigger delay."""
        pass

    @property
    @abstractmethod
    def trigger_delay_resolution(self) -> dirigo.Time:
        """Resolution of the trigger delay setting."""
        pass
    
    # TODO, feasible to merge the next 3 into trigger_delay?
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
    def record_length(self) -> int:
        """Record length in number samples.

        Use `record_duration` for specifying the same setting in terms of time. 
        """
        pass

    @property
    @abstractmethod
    def record_duration(self) -> dirigo.Time:
        """Record duration.

        Use `record_length` for specifying the same setting in terms of number 
        samples. 
        """
        pass

    @record_length.setter
    @abstractmethod
    def record_length(self, length: int):
        """Set the record length in number samples.
        
        Record length (in number of samples) must be greater than 
        `record_length_minimum` and divisible by `record_length_resolution`.
        """
        pass

    @property
    @abstractmethod
    def record_length_minimum(self) -> int:
        """Minimum record length."""
        pass

    @property
    @abstractmethod
    def record_length_resolution(self) -> int:
        """Resolution of the record length setting."""
        pass

    @property
    @abstractmethod
    def records_per_buffer(self) -> int:
        """Number of records per buffer.
        
        A buffer is a chunk of data to transfer from digitizer to host. See
        digitizer-specific documentation for tips on optimal settings.
        """
        pass

    @records_per_buffer.setter
    @abstractmethod
    def records_per_buffer(self, records: int):
        """Set the number of records per buffer."""
        pass

    @property
    @abstractmethod
    def buffers_per_acquisition(self) -> int:
        """Total number of buffers to acquire during an acquisition.
        
        Must be greater than or equal to `buffers_allocated`.
        """
        pass

    @buffers_per_acquisition.setter
    @abstractmethod
    def buffers_per_acquisition(self, buffers: int):
        """Set the total number of buffers to acquire during an acquisition."""
        pass

    @property
    @abstractmethod
    def buffers_allocated(self) -> int:
        """Number of buffers allocated in memory for acquisition.
        
        Must be less than or equal to `buffers_per_acquisition`.
        """
        pass

    @buffers_allocated.setter
    @abstractmethod
    def buffers_allocated(self, buffers: int):
        """Set the number of buffers to allocate for an acquisition."""
        pass

    @abstractmethod
    def start(self):
        """Start the acquisition process."""
        pass

    @property
    @abstractmethod
    def buffers_acquired(self) -> int:
        """Number of buffers acquired during the current acquisition."""
        pass

    @abstractmethod
    def get_next_completed_buffer(self, blocking: bool = True) -> np.ndarray:
        """Retrieve the next completed data buffer.

        Args:
            blocking (bool): Whether to block until a buffer is available.

        Returns:
            np.ndarray: The acquired data buffer.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the acquisition process."""
        pass

    
class AuxillaryIO(ABC):
    """Abstract base class for auxiliary input/output functionality."""

    @abstractmethod
    def configure_mode(self, mode, **kwargs):
        """Configure the auxiliary I/O mode.

        Args:
            mode: The I/O mode to configure.
            **kwargs: Additional parameters for configuration.
        """
        pass

    @abstractmethod
    def read_input(self) -> bool:
        """Read the state of an auxiliary digital input.

        Returns:
            bool: The state of the input (True for high, False for low).
        """
        pass

    @abstractmethod
    def write_output(self, state: bool):
        """Set the state of an auxiliary output.

        Args:
            state (bool): The desired output state (True for high, False for low).
        """
        pass



class Digitizer(ABC):
    """Abstract base class for digitizer hardware interface."""

    PROFILE_LOCATION = Path(__file__).parents[2] / "config/digitizer"

    @abstractmethod
    def __init__(self):
        self.sample_clock: SampleClock
        self.channels: list[Channel]
        self.trigger: Trigger
        self.acquire: Acquire
        self.aux_io: AuxillaryIO

    def load_profile(self, profile_name:str):
        """Load and apply a settings profile from a TOML file.

        Args:
            profile_name (str): Name of the profile to load (without file extension).
        """
        profile_fn = profile_name + ".toml"
        profile = load_toml(self.PROFILE_LOCATION / profile_fn)
        
        self.sample_clock.source = profile["sample_clock"]["clock_source"]
        self.sample_clock.rate = profile["sample_clock"]["rate"]
        self.sample_clock.edge = profile["sample_clock"]["edge"]

        for i, channel in enumerate(self.channels):
            channel.enabled = profile["channels"]["enabled"][i]
            channel.coupling = profile["channels"]["coupling"][i]
            channel.impedance = profile["channels"]["impedance"][i]
            channel.range = profile["channels"]["range"][i]

        self.trigger.external_range = profile["trigger"]["external_range"]
        self.trigger.external_coupling = profile["trigger"]["external_coupling"]
        self.trigger.source = profile["trigger"]["source"]
        self.trigger.slope = profile["trigger"]["slope"]
        self.trigger.level = 0
