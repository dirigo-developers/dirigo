import threading
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass, asdict
from enum import StrEnum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast, ClassVar

from platformdirs import user_config_dir

from dirigo.components import units
from dirigo.hw_interfaces.hw_interface import HardwareInterface
if TYPE_CHECKING:
    from dirigo.sw_interfaces.acquisition import AcquisitionProduct

__all__ = [
    # enums
    "ChannelCoupling", "ImpedanceMode", "InputMode", "StreamingMode",
    "SampleClockSource", "SampleClockEdge", 
    "TriggerSource", "TriggerSlope", "ExternalTriggerCoupling",
    "AuxiliaryIOMode",
    # profiles
    "ChannelProfile", "SampleClockProfile", "TriggerProfile", "DigitizerProfile",
    # ABCs
    "Channel", "SampleClock", "Trigger", "Acquire", "AuxiliaryIO", "Digitizer",
]

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
- AuxiliaryIO: Optional class for auxiliary input/output (GPIO) functionality.
- Digitizer: Top-level abstraction encapsulating the hardware interface.

Plugins must include a project entry point under the "dirigo_digitizers" group. 
Example configuration:

[project.entry-points."dirigo_digitizers"]
alazar = "dirigo_alazar:AlazarDigitizer"
"""


# ---------- Digitizer enumerations ----------
class ChannelCoupling(StrEnum):
    """Enumerate allowed analog input coupling modes."""
    AC      = "ac"
    DC      = "dc"
    GROUND  = "ground"

class ImpedanceMode(StrEnum):
    """Enumerate non-numeric impedance presets (e.g., high-Z)."""
    HIGH = "high" # for "high-Z"
    # for 50 ohms, use a units.Resistance object instead

class InputMode(StrEnum):
    """Select the input data modality (analog or edge counting)."""
    ANALOG          = "analog"
    EDGE_COUNTING   = "edge counting" # e.g. photon counting

class StreamingMode(StrEnum):
    """Select the acquisition streaming mode (triggered or continuous)."""
    TRIGGERED   = "triggered"   # e.g. Alazar, Teledyne
    CONTINUOUS  = "continuous"  # e.g. NI which can sync with analog out
    
class SampleClockSource(StrEnum):
    """Enumerate supported sample clock sources."""
    INTERNAL = "internal"
    EXTERNAL = "external"

class SampleClockEdge(StrEnum):
    """Enumerate valid sampling edges for the sample clock."""
    RISING  = "rising"
    FALLING = "falling"

class TriggerSource(StrEnum):
    """Enumerate valid trigger sources (internal, external, or channels)."""
    INTERNAL  = "internal"
    EXTERNAL  = "external"
    CHANNEL_A = "channel_a"
    CHANNEL_B = "channel_b"
    CHANNEL_C = "channel_c"
    CHANNEL_D = "channel_d"

class TriggerSlope(StrEnum):
    """Enumerate valid trigger edge polarities."""
    RISING  = "rising"
    FALLING = "falling"

class ExternalTriggerCoupling(StrEnum):
    """Enumerate coupling options for the external trigger input."""
    AC = "ac"
    DC = "dc"

class AuxiliaryIOMode(StrEnum):
    """Enumerate auxiliary I/O modes for trigger/pacer/digital lines."""
    DISABLE             = "disable"
    OUT_TRIGGER         = "out_trigger"
    OUT_PACER           = "out_pacer"
    OUT_DIGITAL         = "out_digital"
    IN_TRIGGER_ENABLE   = "in_trigger_enable"
    IN_DIGITAL          = "in_digital"


# ---------- Digitizer profiles ----------
@dataclass(frozen=True, slots=True)
class SampleClockProfile:
    """Parseable, serializable description of a desired sample-clock setup."""
    source: SampleClockSource
    rate: units.SampleRate
    edge: SampleClockEdge = SampleClockEdge.RISING

    @classmethod
    def from_dict(cls, d: dict) -> "SampleClockProfile":
        return cls(
            source = SampleClockSource(str(d["source"]).lower()),
            rate = units.SampleRate(d["rate"]),
            edge = SampleClockEdge(str(d["edge"]).lower())
        )


@dataclass(frozen=True, slots=True)
class ChannelProfile:
    """Parseable, serializable description of a single input channel setup."""
    enabled: bool
    coupling: ChannelCoupling | None
    impedance: units.Resistance | ImpedanceMode | None
    input_range: units.VoltageRange | None
    offset: units.Voltage | None = units.Voltage("0 V") 
    inverted: bool = False

    @classmethod
    def parse(cls, d: dict) -> "ChannelProfile":
        coupling_raw = d.get("coupling")
        if coupling_raw is None:
            coupling = None
        else:
            coupling = ChannelCoupling(str(coupling_raw).lower())

        imp_raw = d.get("impedance")
        impedance: units.Resistance | ImpedanceMode | None
        if imp_raw is None:
            impedance = None
        else:
            s = str(imp_raw).lower()
            impedance = ImpedanceMode(s) if s in (e.value for e in ImpedanceMode) \
                else units.Resistance(str(imp_raw))

        rng = d.get("range")
        if isinstance(rng, dict):
            input_range = units.VoltageRange(rng['min'], rng['max'])
        elif isinstance(rng, str):
            input_range = units.VoltageRange(rng)
        else:
            input_range = None

        offset = units.Voltage(d.get("offset", "0 V"))

        return cls(
            enabled     = bool(d["enabled"]),
            inverted    = bool(d.get("inverted", False)),
            coupling    = coupling,
            impedance   = impedance,
            input_range = input_range,
            offset      = offset,
        )

    @classmethod
    def list_from(cls, items: Iterable[dict]) -> list["ChannelProfile"]:
        return [cls.parse(item) for item in items]


@dataclass(frozen=True, slots=True)
class TriggerProfile:
    """Parseable, serializable description of trigger configuration."""
    source: TriggerSource
    slope: TriggerSlope
    level: units.Voltage | None
    external_coupling: ExternalTriggerCoupling | None
    external_impedance: units.Resistance | ImpedanceMode | None
    external_range: units.VoltageRange | None

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerProfile":
        level = units.Voltage(cast(str, d["level"])) if "level" in d else None

        external_coupling = (
            ExternalTriggerCoupling(str(d["external_coupling"]).lower())
            if "external_coupling" in d else None
        )

        if "external_impedance" in d:
            raw = str(d["external_impedance"]).lower()
            if raw in (e.value for e in ImpedanceMode):
                external_impedance = ImpedanceMode(raw)
            else:
                external_impedance = units.Resistance(cast(str, d["external_impedance"]))
        else:
            external_impedance = None

        if "external_range" in d:
            if isinstance(d["external_range"], dict):
                external_range = units.VoltageRange(
                    min=d["external_range"]["min"], 
                    max=d["external_range"]["max"]
                )
            else:
                external_range = units.VoltageRange(d["external_range"])
        else:
            external_range = None

        return cls(
            source              = TriggerSource(str(d["source"]).lower()),
            slope               = TriggerSlope(str(d["slope"]).lower()),
            level               = level,
            external_coupling   = external_coupling or ExternalTriggerCoupling.DC,
            external_impedance  = external_impedance,
            external_range      = external_range,
        )


@dataclass(frozen=True, slots=True)
class DigitizerProfile:
    """Aggregate profile of sample clock, channels, and trigger for a device."""
    sample_clock: SampleClockProfile
    channels: list[ChannelProfile]
    trigger: TriggerProfile
    
    def to_dict(self) -> dict:
        d = asdict(self)
        
        # Replace VoltageRanges with dicts
        for ch in d["channels"]:
            vr = ch.get("input_range")
            if isinstance(vr, units.VoltageRange):
                ch["input_range"] = vr.to_dict()

        trig = d["trigger"]
        vr = trig.get("external_range")
        if isinstance(vr, units.VoltageRange):
            trig["external_range"] = vr.to_dict()
        return d
     
    @classmethod
    def from_dict(cls, d: dict):
        return cls( 
            sample_clock = SampleClockProfile.from_dict(d["sample_clock"]),
            channels     = ChannelProfile.list_from(cast(Iterable[dict], d["channels"])),
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
    """Configure and query a single input channel's analog settings."""
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
    def coupling_options(self) -> Collection[ChannelCoupling]:
        """Set of available coupling modes."""
        pass

    @property
    @abstractmethod
    def impedance(self) -> units.Resistance | ImpedanceMode:
        """Input impedance setting (e.g., 50 Ohm, "High")."""
        pass

    @impedance.setter
    @abstractmethod
    def impedance(self, impedance: units.Resistance | ImpedanceMode):
        """Set the input impedance.
        
        Must match one of the options provided by `impedance_options`.
        """
        pass

    @property
    @abstractmethod
    def impedance_options(self) -> Collection[units.Resistance | ImpedanceMode]:
        """Set of available input impedance modes."""
        pass

    @property
    @abstractmethod
    def input_range(self) -> units.VoltageRange: 
        """Voltage range for the channel."""
        pass

    @input_range.setter
    @abstractmethod
    def input_range(self, range: units.VoltageRange):
        """Set the voltage range for the channel.
        
        Must match one of the options provided by `range_options`.
        """
        pass

    @property
    @abstractmethod
    def range_options(self) -> Collection[units.VoltageRange]:
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
    """Configure and query the digitizer's sampling clock."""
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
    def source_options(self) -> Collection[SampleClockSource]:
        """Set of supported clock sources."""
        pass

    @property
    @abstractmethod
    def rate(self) -> units.SampleRate:
        """Current sampling rate."""
        pass

    @rate.setter
    @abstractmethod
    def rate(self, value: units.SampleRate):
        """Set the sampling rate."""
        pass
    
    @property
    @abstractmethod
    def rate_options(self) -> Collection[units.SampleRate | units.SampleRateRange]:
        """Supported sampling rates."""
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
    def edge_options(self) -> Collection[SampleClockEdge]:
        """Set of supported clock edge options."""
        pass
    

class Trigger(ABC):
    """Configure and query the digitizer's trigger behavior."""
    @property
    @abstractmethod
    def source(self) -> TriggerSource:
        """Trigger source (e.g., internal, external)."""
        pass
    
    @source.setter
    @abstractmethod
    def source(self, source: TriggerSource):
        """Set the trigger source.
        
        Must match one of the options provided by `source_options`.
        """
        pass

    @property
    @abstractmethod
    def source_options(self) -> Collection[TriggerSource]:
        """Set of available trigger sources."""
        pass

    @property
    @abstractmethod
    def slope(self) -> TriggerSlope:
        """Trigger slope (e.g., rising, falling)."""
        pass
    
    @slope.setter
    @abstractmethod
    def slope(self, slope: TriggerSlope):
        """Set the trigger slope.
        
        Must match one of the options provided by `slope_options`.
        """
        pass

    @property
    @abstractmethod
    def slope_options(self) -> Collection[TriggerSlope]:
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
    def external_coupling_options(self) -> Collection[ExternalTriggerCoupling]:
        """Set of available external coupling modes."""
        pass
    
    @property
    @abstractmethod
    def external_impedance(self) -> units.Resistance | ImpedanceMode: 
        """Impedance of the external trigger input (numeric resistance or mode)."""
        pass

    @external_impedance.setter
    @abstractmethod
    def external_impedance(self, imp: units.Resistance | ImpedanceMode): 
        """Set the external trigger input impedance.
        
        Must match one of `external_impedance_options`."""
        pass

    @property
    @abstractmethod
    def external_impedance_options(self) -> Collection[units.Resistance | ImpedanceMode]: 
        """Supported external trigger impedances."""
        pass

    @property
    @abstractmethod
    def external_range(self) -> units.VoltageRange: # return dirigo.VoltageRange?
        """Voltage range for external triggers."""
        pass
    
    @external_range.setter
    @abstractmethod
    def external_range(self, range: units.VoltageRange):
        """Set the voltage range for external triggers.
        
        Must match one of the options provided by `external_range_options`.
        """
        pass

    @property
    @abstractmethod
    def external_range_options(self) -> Collection[units.VoltageRange]:
        """Set of available external trigger voltage ranges."""
        pass


class Acquire(ABC):
    """Manage acquisition parameters, lifecycle, and buffer handoff."""
    def __init__(self):
        self._channels: tuple[Channel, ...]
        self._active = threading.Event()

    @property
    def n_channels_enabled(self) -> int:
        """Number of channels enabled for acquisition."""
        return sum(c.enabled for c in self._channels)
    
    @property
    @abstractmethod
    def trigger_delay(self) -> int:
        """Delay in samples relative to the trigger: negative = pre-trigger, non-negative = post-trigger."""
        pass

    @trigger_delay.setter
    @abstractmethod
    def trigger_delay(self, value: int):
        """
        Set the trigger-relative delay (samples). Must lie within `trigger_delay_range`
        and be a multiple of `pre_trigger_delay_step` (if negative) or `post_trigger_delay_step` 
        (if ≥0).
        """
        pass

    @property
    @abstractmethod
    def trigger_delay_range(self) -> units.IntRange:
        """Valid delay range (inclusive) for `trigger_delay`."""
        pass

    @property
    @abstractmethod
    def pre_trigger_delay_step(self) -> int:
        """Minimum granularity (in samples) for negative `trigger_delay` values."""
        pass

    @property
    @abstractmethod
    def post_trigger_delay_step(self) -> int:
        """Minimum granularity (in samples) for non-negative `trigger_delay` values."""
        pass

    @property
    @abstractmethod
    def record_length(self) -> int:
        """Record length in number samples."""
        pass

    @record_length.setter
    @abstractmethod
    def record_length(self, length: int):
        """Set the record length in number samples.
        
        Record length (in number of samples) must be greater than 
        `record_length_minimum` and divisible by `record_length_step`.
        """
        pass

    @property
    @abstractmethod
    def record_length_minimum(self) -> int:
        """Minimum record length."""
        pass

    @property
    @abstractmethod
    def record_length_step(self) -> int:
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
    @abstractmethod
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
            acq_buffer (AcquisitionProduct): Pre-allocated acquisition buffer to 
            copy completed digitizer buffer
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
    """Configure and control auxiliary I/O lines for triggers and digital I/O."""
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
    """Top-level digitizer interface composed of clock, channels, trigger, and I/O."""
    attr_name: ClassVar[str] = "digitizer"
    PROFILE_LOCATION: ClassVar[Path] = Path(user_config_dir("Dirigo")) / "digitizer"

    @abstractmethod
    def __init__(self):
        self.input_mode: InputMode
        self.streaming_mode: StreamingMode
        self.profile: DigitizerProfile
        self.sample_clock: SampleClock
        self.channels: tuple[Channel, ...]
        self.trigger: Trigger
        self.acquire: Acquire
        self.aux_io: AuxiliaryIO

    def load_profile(self, profile_name: str):
        """Load and apply a settings profile from a TOML file.

        Args:
            profile_name (str): Name of the profile to load (no file extension).
        """
        profile_path = self.PROFILE_LOCATION / (profile_name + ".toml")
        profile = DigitizerProfile.from_toml(profile_path)

        if len(self.channels) != len(profile.channels):
            raise ValueError(
                f"Profile defines {len(profile.channels)} channels, "
                f"device has {len(self.channels)}."
            )
        for channel, channel_profile in zip(self.channels, profile.channels):
            channel.enabled = channel_profile.enabled
            # NI X-series requires # channels enabled to check max (aggregate) sample rate
            # Teledyne also requires channels to be enabled (nof_records > 0) to not ignore some settings
        
        self.sample_clock.source = profile.sample_clock.source
        self.sample_clock.rate = profile.sample_clock.rate
        self.sample_clock.edge = profile.sample_clock.edge
        
        for channel, channel_profile in zip(self.channels, profile.channels):
            if channel_profile.coupling is not None:
                channel.coupling = channel_profile.coupling
            if channel_profile.impedance is not None:
                channel.impedance = channel_profile.impedance
            if channel_profile.input_range is not None:
                channel.input_range = channel_profile.input_range
            if channel_profile.offset is not None:
                channel.offset = channel_profile.offset

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
    @abstractmethod
    def bit_depth(self) -> int:
        """Returns the bit depth (sample resolution) of the digitizer."""
        pass
    
    @property
    def is_active(self) -> bool:
        """Digitizer activity status."""
        return self.acquire._active.is_set()
