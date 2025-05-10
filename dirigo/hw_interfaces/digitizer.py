from abc import ABC, abstractmethod
from functools import cached_property
from dataclasses import dataclass, asdict
from pathlib import Path
import math
from typing import Literal, List, Optional
import tomllib

from platformdirs import user_config_dir

from dirigo import units
from dirigo.sw_interfaces.acquisition import AcquisitionProduct

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


@dataclass(frozen=True, slots=True)
class SampleClockProfile:
    source: str
    rate: Optional[units.SampleRate] = None
    edge: Literal["Rising", "Falling"] = "Rising"

    @classmethod
    def from_dict(cls, d: dict) -> "SampleClockProfile":
        return cls(
            source = d["source"],
            rate = units.SampleRate(d["rate"]),
            edge = d["edge"]
        )


@dataclass(frozen=True, slots=True)
class ChannelProfile:
    enabled: bool
    inverted: bool
    coupling: Literal["AC", "DC"]
    impedance: units.Resistance
    range: units.VoltageRange

    @classmethod
    def list_from_dict(cls, d: dict) -> List["ChannelProfile"]:
        channel_list = []
        for ena, inv, cou, imp, ran in zip(d["enabled"], d["inverted"], d["coupling"], d["impedance"], d["range"]):
            channel = cls(
                enabled=ena, 
                inverted=inv,
                coupling=cou, 
                impedance=units.Resistance(imp), 
                range=units.VoltageRange(ran)
            )
            channel_list.append(channel)
        return channel_list


@dataclass(frozen=True, slots=True)
class TriggerProfile:
    source: str
    slope: str
    level: int
    external_range: str
    external_coupling: Literal["AC", "DC"]

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerProfile":
        return cls(
            source = d["source"],
            slope = d["slope"],
            level = d["level"],
            external_range = d["external_range"],
            external_coupling = d["external_coupling"]
        )


@dataclass(frozen=True, slots=True)
class DigitizerProfile:
    """Describes static parameters of the Digitizer."""
    sample_clock: SampleClockProfile
    channels: List[ChannelProfile]
    trigger: TriggerProfile
    
    def to_dict(self):
        d = asdict(self)
        # replace VoltageRange objects with simple dictionary for serialization
        for channel in d["channels"]:
            channel["range"] = channel["range"].to_dict()
        return d
     
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            sample_clock = SampleClockProfile.from_dict(d["sample_clock"]),
            channels     = ChannelProfile.list_from_dict(d["channels"]),
            trigger      = TriggerProfile.from_dict(d["trigger"]),
        )
    
    @classmethod
    def from_toml(cls, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find Digitizer profile at: {path}")
        
        with open(path, "rb") as fh:
            data = tomllib.load(fh)

        return cls.from_dict(data)


class Channel(ABC):
    """Abstract base class for a digitizer's input channel configuration."""
    def __init__(self, 
                 enabled: bool = False, 
                 inverted: bool = False):
        super().__init__() # Is this needed?

        self.enabled = enabled
        self.inverted = inverted

    @property
    def enabled(self) -> bool:
        """Indicates whether the channel is enabled for acquisition."""
        return self._enabled

    @enabled.setter
    def enabled(self, enable: bool):
        """Enable or disable the channel."""
        if not isinstance(enable, bool):
            raise ValueError("`inverted` must be set with a boolean")
        self._enabled = enable

    @property
    def inverted(self) -> bool:
        """
        Indicates whether the channel values should be inverted.
        
        Channel subclasses that don't support inverted inputs (i.e. edge 
        counting) should override setter method."""
        return self._inverted

    @inverted.setter
    def inverted(self, invert: bool):
        # For edge counting (e.g. photon counting) channels, override this with
        # empty method (with 'pass')
        if not isinstance(invert, bool):
            raise ValueError("`inverted` must be set with a boolean")
        self._inverted = invert

    @property
    @abstractmethod
    def index(self) -> int:
        """Index of the channel (0-based index)."""
        pass

    @property
    @abstractmethod
    def coupling(self) -> str:
        """Signal coupling mode (e.g., "AC", "DC")."""
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
        """Input impedance setting (e.g., "50 Ohm", "1 MOhm")."""
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
    def rate(self) -> units.SampleRate:
        pass

    @rate.setter
    @abstractmethod
    def rate(self, value: units.SampleRate):
        pass
    
    @property
    @abstractmethod
    def rate_options(self) -> set[units.SampleRate]:
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
    def level(self) -> units.Voltage:
        """Trigger level in volts."""
        pass
    
    @level.setter
    @abstractmethod
    def level(self, level: units.Voltage):
        """Set the trigger level."""
        pass

    @property
    @abstractmethod
    def level_limits(self) -> units.VoltageRange:
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
    def external_range(self) -> str: # return dirigo.VoltageRange?
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
    def trigger_delay_samples(self) -> int:
        """
        Delay between trigger event and acquisition start, in sample clock periods.
        
        Use `trigger_delay_duration` for the same setting in terms of time.
        """
        # Arguably could be part of Trigger object, but put here because of role
        # in acquisition timing
        pass

    @trigger_delay_samples.setter
    @abstractmethod
    def trigger_delay_samples(self, samples: int):
        """Set the trigger delay, in sample clock periods."""
        pass

    @property
    @abstractmethod
    def trigger_delay_duration(self) -> units.Time:
        """Delay between trigger event and acquisition start, in units of time.
        
        Use `trigger_delay_samples` for the same setting in terms of sample 
        clock periods. 
        """
        pass

    @property
    @abstractmethod
    def trigger_delay_sample_resolution(self) -> int:
        """Resolution of the trigger delay setting, in sample clock periods."""
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

        Use `record_duration` for the same setting in terms of time. 
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
    def record_duration(self) -> units.Time:
        """Record duration.

        Use `record_length` for the same setting in terms of number 
        samples. 
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

    @property
    @abstractmethod
    def timestamps_enabled(self) -> bool:
        """Enables timestamps on boards supporting them.
        
        If timestamps are not available, should raise NotImplementedError"""
        pass

    @timestamps_enabled.setter
    def timestamps_enabled(self, enable: bool):
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
    def get_next_completed_buffer(self, acq_buffer: AcquisitionProduct):
        """Retrieve the next completed data buffer.

        Args:
            acq_buffer (AcquisitionBuffer): Pre-allocated acquisition buffer to copy completed digitizer buffer
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the acquisition process."""
        pass

    @cached_property
    def _inverted_channels(self) -> list[bool]:
        return [chan.inverted for chan in self._channels if chan.enabled]

    
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

    PROFILE_LOCATION = Path(user_config_dir("Dirigo")) / "digitizer"

    @abstractmethod
    def __init__(self):
        self.profile: DigitizerProfile
        self.sample_clock: SampleClock
        self.channels: list[Channel]
        self.trigger: Trigger
        self.acquire: Acquire
        self.aux_io: AuxillaryIO

    def load_profile(self, profile_name: str):
        """Load and apply a settings profile from a TOML file.

        Args:
            profile_name (str): Name of the profile to load (without file extension).
        """
        profile_path = self.PROFILE_LOCATION / (profile_name + ".toml")
        self.profile = DigitizerProfile.from_toml(profile_path)
        
        self.sample_clock.source = self.profile.sample_clock.source
        self.sample_clock.rate = self.profile.sample_clock.rate
        self.sample_clock.edge = self.profile.sample_clock.edge

        for channel, channel_profile in zip(self.channels, self.profile.channels):
            channel.enabled = channel_profile.enabled
            # Digitizer doesn't need to know whether a channel is inverted
            channel.coupling = channel_profile.coupling
            channel.impedance = channel_profile.impedance
            channel.range = channel_profile.range

        self.trigger.external_range = self.profile.trigger.external_range
        self.trigger.external_coupling = self.profile.trigger.external_coupling
        self.trigger.source = self.profile.trigger.source
        self.trigger.slope = self.profile.trigger.slope
        self.trigger.level = self.profile.trigger.level

    @property
    @abstractmethod
    def data_range(self) -> units.IntRange:
        """
        Returns the range of values returned by the digitizer 
        
        The returned data range may exceed the bit depth, which can be useful
        for in-place averaging.
        """
        pass

    @property
    def bit_depth(self) -> int:
        """Returns the bit depth (sample resolution) of the digitizer.
        
        Requires data_range to be set up accurately in subclass."""
        return math.ceil(math.log2(self.data_range.range))
