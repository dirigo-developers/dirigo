from abc import ABC, abstractmethod
from functools import cached_property
from dataclasses import dataclass, asdict
from pathlib import Path
import math
from typing import Literal, List, TYPE_CHECKING
import tomllib
from enum import StrEnum, Enum

from platformdirs import user_config_dir

from dirigo.components import units
from dirigo.hw_interfaces.hw_interface import HardwareInterface
if TYPE_CHECKING:
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


# ---------- Digitizer enumerations ----------
class SampleClockSource(StrEnum):
    INTERNAL = "internal"
    EXTERNAL = "external"

class SampleClockEdge(StrEnum):
    RISING  = "rising"
    FALLING = "falling"

class ChannelCoupling(StrEnum):
    AC      = "ac"
    DC      = "dc"
    GROUND  = "ground"

class TriggerSource(StrEnum):
    INTERNAL  = "internal"
    EXTERNAL  = "external"
    CHANNEL_A = "channel_a"
    CHANNEL_B = "channel_b"
    CHANNEL_C = "channel_c"
    CHANNEL_D = "channel_d"
    # could do more ...

class TriggerSlope(StrEnum):
    RISING  = "rising"
    FALLING = "falling"

class ExternalTriggerCoupling(StrEnum):
    AC = "ac"
    DC = "dc"

class ExternalTriggerImpedance(StrEnum):
    HIGH = "high" # for "high-Z"
    # for 50 ohms, use a units.Resistance object instead

class ExternalTriggerRange(StrEnum):
    TTL = "ttl"
    # True voltage ranges should use the units.VoltageRange class

class AuxiliaryIOMode(StrEnum):
    DISABLE             = "disable"
    OUT_TRIGGER         = "out_trigger"
    OUT_PACER           = "out_pacer"
    OUT_DIGITAL         = "out_digital"
    IN_TRIGGER_ENABLE   = "in_trigger_enable"
    IN_DIGITAL          = "in_digital"


# ---------- Digitizer profiles ----------
@dataclass(frozen=True, slots=True)
class SampleClockProfile:
    source: SampleClockSource
    rate: units.SampleRate
    edge: SampleClockEdge = SampleClockEdge.RISING

    @classmethod
    def from_dict(cls, d: dict) -> "SampleClockProfile":
        rate = units.SampleRate(d["rate"]) if d.get("rate") is not None else None
        return cls(
            source = SampleClockSource(d["source"].lower()),
            rate = rate,
            edge = SampleClockEdge(d["edge"].lower())
        )


@dataclass(frozen=True, slots=True)
class ChannelProfile:
    enabled: bool
    inverted: bool
    coupling: ChannelCoupling
    impedance: units.Resistance
    range: units.VoltageRange
    offset: units.Voltage = units.Voltage("0 V")

    @classmethod
    def from_dict(cls, channel_profile_list: List[dict]) -> List["ChannelProfile"]:
        channel_list = []
        for channel_profile in channel_profile_list:
            coupling = channel_profile.get("coupling")
            if coupling is not None: 
                coupling = ChannelCoupling(channel_profile["coupling"].lower()) 

            impedance = channel_profile.get("impedance")
            if impedance is not None: 
                impedance = units.Resistance(channel_profile["impedance"]) 

            input_range = channel_profile.get("range")
            if isinstance(input_range, dict):
                # if dictionary with min, max keys
                input_range = units.VoltageRange(**input_range)
            elif isinstance(input_range, str):
                # if coming from toml file and using plus/minus, eg "±2 V"
                input_range = units.VoltageRange(input_range)

            offset = channel_profile.get("offset", units.Voltage("0 V"))
            offset = units.Voltage(offset)

            channel = cls(
                enabled     = channel_profile["enabled"], 
                inverted    = channel_profile.get("inverted", False),
                coupling    = coupling, 
                impedance   = impedance, 
                range       = input_range,
                offset      = offset 
            )
            channel_list.append(channel)
        return channel_list


@dataclass(frozen=True, slots=True)
class TriggerProfile:
    source: TriggerSource
    slope: TriggerSlope
    level: units.Voltage | None
    external_coupling: ExternalTriggerCoupling | None
    external_impedance: units.Resistance | ExternalTriggerImpedance | None
    external_range: units.VoltageRange | ExternalTriggerRange | None

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerProfile":
        if "level" in d.keys():
            level = units.Voltage(d["level"])
        else:
            level = None

        if "external_coupling" in d.keys():
            external_coupling = ExternalTriggerCoupling(d["external_coupling"].lower())
        else:
            external_coupling = None

        if "external_impedance" in d.keys():
            try:
                external_impedance = ExternalTriggerImpedance(d["external_impedance"].lower())
            except:
                external_impedance = units.Resistance(d["external_impedance"])
        else:
            external_impedance = None

        if "external_range" in d.keys():
            try:
                external_range = ExternalTriggerRange(d["external_range"].lower())
            except: 
                external_range = units.VoltageRange(d["external_range"])
        else:
            external_range = None

        return cls(
            source = TriggerSource(d["source"].lower()),
            slope = TriggerSlope(d["slope"].lower()),
            level = level,
            external_coupling = external_coupling or ExternalTriggerCoupling.DC,
            external_impedance = external_impedance,
            external_range = external_range,
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
            try:
                channel["range"] = channel["range"].to_dict()
            except AttributeError: # None type
                pass
        if isinstance(d["trigger"]["external_range"], units.VoltageRange):
            d["trigger"]["external_range"] = d["trigger"]["external_range"].to_dict()

        return d
     
    @classmethod
    def from_dict(cls, d: dict):
        return cls( 
            sample_clock = SampleClockProfile.from_dict(d["sample_clock"]),
            channels     = ChannelProfile.from_dict(d["channels"]),
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


class InputMode(StrEnum):
    ANALOG          = "analog"
    EDGE_COUNTING   = "edge counting" # e.g. photon counting


class StreamingMode(StrEnum):
    TRIGGERED   = "triggered"
    CONTINUOUS  = "continuous"


class Channel(ABC):
    """Abstract base class for a digitizer's input channel configuration."""
    def __init__(self, 
                 enabled: bool = False, 
                 inverted: bool = False):
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
            raise ValueError("`enabled` must be set with a boolean")
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
    def coupling(self) -> ChannelCoupling:
        """Signal coupling mode (e.g., "AC", "DC")."""
        pass

    @coupling.setter
    @abstractmethod
    def coupling(self, coupling: ChannelCoupling):
        """Set the signal coupling mode.
        
        Must match one of the options provided by `coupling_options`.
        """
        pass

    @property
    @abstractmethod
    def coupling_options(self) -> set[ChannelCoupling]:
        """Set of available coupling modes."""
        pass

    @property
    @abstractmethod
    def impedance(self) -> units.Resistance:
        """Input impedance setting (e.g., 50 Ohm, 1 MOhm)."""
        pass

    @impedance.setter
    @abstractmethod
    def impedance(self, impedance: units.Resistance):
        """Set the input impedance.
        
        Must match one of the options provided by `impedance_options`.
        """
        pass

    @property
    @abstractmethod
    def impedance_options(self) -> set[units.Resistance]:
        """Set of available input impedance modes."""
        pass

    @property
    @abstractmethod
    def range(self) -> units.VoltageRange: 
        """Voltage range for the channel."""
        pass

    @range.setter
    @abstractmethod
    def range(self, range: units.VoltageRange):
        """Set the voltage range for the channel.
        
        Must match one of the options provided by `range_options`.
        """
        pass

    @property
    @abstractmethod
    def range_options(self) -> set[units.VoltageRange]:
        """Set of available voltage ranges."""
        pass

    @property
    @abstractmethod
    def offset(self) -> units.Voltage:
        """DC offset voltage"""
        pass

    @offset.setter
    @abstractmethod
    def offset(self, offset: units.Voltage):
        """Set the DC offset voltage.

        Must be within range specified by `offset_range`
        """
        pass

    @property
    @abstractmethod
    def offset_range(self) -> units.VoltageRange:
        """Settable range for analog DC offset."""
        pass
    
    
class SampleClock(ABC):
    """Abstract base class for configuring the digitizer's sampling clock."""

    @property
    @abstractmethod
    def source(self) -> SampleClockSource:
        """Source of the sample clock (e.g., internal, external)."""
        pass

    @source.setter
    @abstractmethod
    def source(self, source: SampleClockSource):
        """Set the source of the sample clock.
        
        Must match one of the options provided by `source_options`.
        """
        pass

    @property
    @abstractmethod
    def source_options(self) -> set[SampleClockSource]:
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
    def edge(self) -> SampleClockEdge:
        """Clock edge to use for sampling (e.g., rising, falling)."""
        pass

    @edge.setter
    @abstractmethod
    def edge(self, edge: SampleClockEdge):
        """Set the clock edge for sampling.
        
        Must match one of the options provided by `edge_options`.
        """
        pass

    @property
    @abstractmethod
    def edge_options(self) -> set[SampleClockEdge]:
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
    def external_coupling(self) -> ExternalTriggerCoupling:
        """Coupling mode for external trigger sources."""
        pass
    
    @external_coupling.setter
    @abstractmethod
    def external_coupling(self, coupling: ExternalTriggerCoupling):
        """Set the coupling mode for external triggers.
        
        Must match one of the options provided by `external_coupling_options`.
        """
        pass

    @property
    @abstractmethod
    def external_coupling_options(self) -> set[ExternalTriggerCoupling]:
        """Set of available external coupling modes."""
        pass
    
    @property
    @abstractmethod
    def external_impedance(self) -> units.Resistance | ExternalTriggerImpedance: 
        pass

    @external_impedance.setter
    @abstractmethod
    def external_impedance(self, imp: units.Resistance | ExternalTriggerImpedance): 
        pass

    @property
    @abstractmethod
    def external_impedance_options(self) -> set[units.Resistance | ExternalTriggerImpedance]: 
        pass

    @property
    @abstractmethod
    def external_range(self) -> units.VoltageRange | ExternalTriggerRange: # return dirigo.VoltageRange?
        """Voltage range for external triggers."""
        pass
    
    @external_range.setter
    @abstractmethod
    def external_range(self, range: units.VoltageRange | ExternalTriggerRange):
        """Set the voltage range for external triggers.
        
        Must match one of the options provided by `external_range_options`.
        """
        pass

    @property
    @abstractmethod
    def external_range_options(self) -> set[units.VoltageRange | ExternalTriggerRange]:
        """Set of available external trigger voltage ranges."""
        pass


class Acquire(ABC):
    """Abstract base class for managing digitizer acquisitions and buffering."""

    def __init__(self):
        self._channels: tuple[Channel, ...]

    @property
    def n_channels_enabled(self) -> int:
        """Number of channels enabled for acquisition."""
        return sum([c.enabled for c in self._channels])
    
    @property
    @abstractmethod
    def trigger_offset(self) -> int:
        pass

    @trigger_offset.setter
    @abstractmethod
    def trigger_offset(self, value: int):
        pass

    @property
    @abstractmethod
    def trigger_offset_range(self) -> units.IntRange:
        pass

    @property
    @abstractmethod
    def pre_trigger_resolution(self) -> int:
        pass

    @property
    @abstractmethod
    def trigger_delay_resolution(self) -> int:
        pass

    @property
    @abstractmethod
    def record_length(self) -> int:
        """ Record length in number samples. """
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
        """Total number of buffers to acquire during an acquisition. -1 codes
        for unlimited.
        
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
    def get_next_completed_buffer(self, acq_buffer: "AcquisitionProduct"):
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

    
class AuxiliaryIO(ABC):
    """Abstract base class for auxiliary input/output functionality."""

    @abstractmethod
    def configure_mode(self, mode: AuxiliaryIOMode, **kwargs):
        """Configure the auxiliary I/O mode.

        Args:
            mode (AuxiliaryIOMode): The I/O mode to configure.
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


class Digitizer(HardwareInterface):
    """Abstract base class for digitizer hardware interface."""
    attr_name = "digitizer"
    PROFILE_LOCATION = Path(user_config_dir("Dirigo")) / "digitizer"

    @abstractmethod
    def __init__(self):
        self.input_mode: InputMode
        self.streaming_mode: StreamingMode
        self.profile: DigitizerProfile
        self.sample_clock: SampleClock
        self.channels: tuple[Channel]
        self.trigger: Trigger
        self.acquire: Acquire
        self.aux_io: AuxiliaryIO

    def load_profile(self, profile_name: str):
        """Load and apply a settings profile from a TOML file.

        Args:
            profile_name (str): Name of the profile to load (without file extension).
        """
        profile_path = self.PROFILE_LOCATION / (profile_name + ".toml")
        profile = DigitizerProfile.from_toml(profile_path)

        for channel, channel_profile in zip(self.channels, profile.channels):
            channel.enabled = channel_profile.enabled
            # NI X-series requires # channels enabled to check max (aggregate) sample rate
            # Teledyne also requires channels to be enabled (nof_records > 0) to not ignore some settings
        
        self.sample_clock.source = profile.sample_clock.source
        self.sample_clock.rate = profile.sample_clock.rate
        self.sample_clock.edge = profile.sample_clock.edge

        for channel, channel_profile in zip(self.channels, profile.channels):
            channel.coupling = channel_profile.coupling
            channel.impedance = channel_profile.impedance
            channel.range = channel_profile.range
            channel.offset = channel_profile.offset

        # Trigger settings
        self.trigger.source = profile.trigger.source
        if profile.trigger.external_coupling is not None:
            self.trigger.external_coupling = profile.trigger.external_coupling
        if profile.trigger.external_impedance is not None:
            self.trigger.external_impedance = profile.trigger.external_impedance
        if profile.trigger.external_range is not None:
            self.trigger.external_range = profile.trigger.external_range
        self.trigger.slope = profile.trigger.slope
        if profile.trigger.level is not None:
            self.trigger.level = profile.trigger.level

        # try, except AttributeError  for any of these?

        self.profile = profile

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
