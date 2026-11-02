"""Digitizer configuration, acquisition, and data-transfer contracts.

Drivers own device state and enforce configuration locking between prepare()
and stop(). Calls are sequential; preparation and execution may use different
threads. Concurrent SDK access is not required by this interface.

Option getters describe settings supported by the driver in the current
configuration. Unsupported values raise ValueError; changes while prepared
raise RuntimeError. Getters may return cached effective settings rather than
querying hardware on every access.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt

from dirigo import units

from .hardware import Device

__all__ = [
    "Acquire",
    "AuxiliaryIO",
    "AuxiliaryIOMode",
    "CaptureInfo",
    "CaptureLayout",
    "Channel",
    "ChannelCoupling",
    "ChannelProfile",
    "Digitizer",
    "DigitizerProfile",
    "ExternalTriggerCoupling",
    "ExternalTriggerRange",
    "ImpedanceMode",
    "InputMode",
    "SampleArray",
    "SampleClock",
    "SampleClockEdge",
    "SampleClockProfile",
    "SampleClockSource",
    "StreamingMode",
    "Trigger",
    "TriggerProfile",
    "TriggerSlope",
    "TriggerSource",
]


class ChannelCoupling(StrEnum):
    """Analog input coupling modes."""

    AC = "ac"
    DC = "dc"
    GROUND = "ground"


class ImpedanceMode(StrEnum):
    """Non-numeric impedance presets."""

    HIGH = "high"


class InputMode(StrEnum):
    """Input data modality."""

    ANALOG = "analog"
    EDGE_COUNTING = "edge counting"


class StreamingMode(StrEnum):
    """Acquisition streaming mode."""

    TRIGGERED = "triggered"
    CONTINUOUS = "continuous"


class SampleClockSource(StrEnum):
    """Sampling clock source."""

    INTERNAL = "internal"
    EXTERNAL = "external"


class SampleClockEdge(StrEnum):
    """Sampling clock edge."""

    RISING = "rising"
    FALLING = "falling"


class TriggerSource(StrEnum):
    """Trigger source."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    CHANNEL_A = "channel_a"
    CHANNEL_B = "channel_b"
    CHANNEL_C = "channel_c"
    CHANNEL_D = "channel_d"


class TriggerSlope(StrEnum):
    """Trigger edge polarity."""

    RISING = "rising"
    FALLING = "falling"


class ExternalTriggerCoupling(StrEnum):
    """External trigger coupling."""

    AC = "ac"
    DC = "dc"


class ExternalTriggerRange(StrEnum):
    """Named external trigger input modes."""

    TTL = "ttl"


class AuxiliaryIOMode(StrEnum):
    """Auxiliary I/O modes."""

    DISABLE = "disable"
    OUT_TRIGGER = "out_trigger"
    OUT_PACER = "out_pacer"
    OUT_DIGITAL = "out_digital"
    IN_TRIGGER_ENABLE = "in_trigger_enable"
    IN_DIGITAL = "in_digital"


@dataclass(frozen=True, slots=True)
class SampleClockProfile:
    """Requested clock settings."""

    source: SampleClockSource
    rate: units.SampleRate
    edge: SampleClockEdge = SampleClockEdge.RISING


@dataclass(frozen=True, slots=True)
class ChannelProfile:
    """Requested channel settings.

    None leaves an optional setting unchanged.
    """

    enabled: bool
    coupling: ChannelCoupling | None = None
    impedance: units.Resistance | ImpedanceMode | None = None
    input_range: units.VoltageRange | None = None
    offset: units.Voltage | None = None
    inverted: bool = False


@dataclass(frozen=True, slots=True)
class TriggerProfile:
    """Requested trigger settings.

    None leaves an optional setting unchanged.
    """

    source: TriggerSource
    slope: TriggerSlope
    level: units.Voltage | None = None
    external_coupling: ExternalTriggerCoupling | None = None
    external_impedance: units.Resistance | ImpedanceMode | None = None
    external_range: units.VoltageRange | ExternalTriggerRange | None = None


@dataclass(frozen=True, slots=True)
class DigitizerProfile:
    """Requested settings, distinct from an effective-state snapshot.

    Profile parsing, file lookup, and application live outside the ABC.
    """

    sample_clock: SampleClockProfile
    channels: tuple[ChannelProfile, ...]
    trigger: TriggerProfile


type SampleArray = npt.NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class CaptureLayout:
    """The layout fixed by prepare() and valid until stop().

    shape is (records_per_buffer, record_length, enabled_channel_count).
    All dimensions are positive.

    channel_indices contains distinct physical channel indices in the
    order represented by the final array axis.

    dtype is the exact native-endian integer storage dtype.
    """

    shape: tuple[int, int, int]
    channel_indices: tuple[int, ...]
    dtype: np.dtype[np.integer[Any]]


@dataclass(frozen=True, slots=True, eq=False)
class CaptureInfo:
    """Information accompanying one successfully transferred complete batch.

    timestamps is a float64 array of first-sample times for each record,
    in seconds relative to the first sample of the first acquired record.

    Its shape is (records_per_buffer,). It is required when timestamps are
    enabled, and None otherwise.

    The driver must not modify it after returning or expose memory whose
    lifetime depends on an SDK buffer.

    Frozen fields do not make the array immutable; treat it as read-only.
    """

    timestamps: npt.NDArray[np.float64] | None = None


class Channel(ABC):
    """Configure one input channel.

    Acquisition-affecting settings are locked while acquisition is prepared.
    """

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether this channel is enabled for acquisition."""

    @enabled.setter
    @abstractmethod
    def enabled(self, enable: bool) -> None:
        """Enable or disable this channel."""

    @property
    @abstractmethod
    def inverted(self) -> bool:
        """Whether channel polarity inversion is selected."""

    @inverted.setter
    @abstractmethod
    def inverted(self, invert: bool) -> None:
        """Select polarity inversion; reject True when unsupported."""

    @property
    @abstractmethod
    def index(self) -> int:
        """Zero-based physical channel index."""

    @property
    @abstractmethod
    def coupling(self) -> ChannelCoupling:
        """Configured channel coupling."""

    @coupling.setter
    @abstractmethod
    def coupling(self, coupling: ChannelCoupling) -> None:
        """Select a supported channel coupling."""

    @property
    @abstractmethod
    def coupling_options(self) -> Collection[ChannelCoupling]:
        """Supported channel coupling modes."""

    @property
    @abstractmethod
    def impedance(self) -> units.Resistance | ImpedanceMode:
        """Configured input impedance or impedance mode."""

    @impedance.setter
    @abstractmethod
    def impedance(self, impedance: units.Resistance | ImpedanceMode) -> None:
        """Select a supported input impedance."""

    @property
    @abstractmethod
    def impedance_options(self) -> Collection[units.Resistance | ImpedanceMode]:
        """Supported input impedances for the current configuration."""

    @property
    @abstractmethod
    def input_range(self) -> units.VoltageRange:
        """Configured analog input voltage range."""

    @input_range.setter
    @abstractmethod
    def input_range(self, value: units.VoltageRange) -> None:
        """Select a supported input voltage range with positive span."""

    @property
    @abstractmethod
    def range_options(self) -> Collection[units.VoltageRange]:
        """Supported input ranges for the current configuration."""

    @property
    @abstractmethod
    def offset(self) -> units.Voltage:
        """Configured analog DC offset."""

    @offset.setter
    @abstractmethod
    def offset(self, offset: units.Voltage) -> None:
        """Set an analog DC offset within offset_options."""

    @property
    @abstractmethod
    def offset_options(self) -> Collection[units.Voltage] | units.VoltageRange:
        """Discrete supported offsets or a continuous adjustable range."""


class SampleClock(ABC):
    """Configure and query the digitizer's sampling clock."""

    @property
    @abstractmethod
    def source(self) -> SampleClockSource:
        """Configured clock source."""

    @source.setter
    @abstractmethod
    def source(self, source: SampleClockSource) -> None:
        """Select a supported clock source."""

    @property
    @abstractmethod
    def source_options(self) -> Collection[SampleClockSource]:
        """Supported clock sources for the current configuration."""

    @property
    @abstractmethod
    def rate(self) -> units.SampleRate:
        """Effective configured sample rate.

        An external rate may be declared rather than measured.
        """

    @rate.setter
    @abstractmethod
    def rate(self, value: units.SampleRate) -> None:
        """Select a supported sampling rate."""

    @property
    @abstractmethod
    def rate_options(self) -> Collection[units.SampleRate] | units.SampleRateRange:
        """Discrete supported rates or a continuous inclusive range."""

    @property
    @abstractmethod
    def edge(self) -> SampleClockEdge:
        """Configured sampling edge."""

    @edge.setter
    @abstractmethod
    def edge(self, edge: SampleClockEdge) -> None:
        """Select a supported sampling edge."""

    @property
    @abstractmethod
    def edge_options(self) -> Collection[SampleClockEdge]:
        """Supported sampling edges."""


class Trigger(ABC):
    """Configure and query the digitizer's trigger behavior."""

    @property
    @abstractmethod
    def source(self) -> TriggerSource:
        """Configured trigger source."""

    @source.setter
    @abstractmethod
    def source(self, source: TriggerSource) -> None:
        """Select a supported trigger source."""

    @property
    @abstractmethod
    def source_options(self) -> Collection[TriggerSource]:
        """Supported trigger sources for the current configuration."""

    @property
    @abstractmethod
    def slope(self) -> TriggerSlope:
        """Configured trigger slope."""

    @slope.setter
    @abstractmethod
    def slope(self, slope: TriggerSlope) -> None:
        """Select a supported trigger slope."""

    @property
    @abstractmethod
    def slope_options(self) -> Collection[TriggerSlope]:
        """Supported trigger slopes."""

    @property
    @abstractmethod
    def level(self) -> units.Voltage:
        """Configured trigger threshold."""

    @level.setter
    @abstractmethod
    def level(self, level: units.Voltage) -> None:
        """Set a trigger threshold within level_options."""

    @property
    @abstractmethod
    def level_options(self) -> Collection[units.Voltage] | units.VoltageRange:
        """Discrete supported trigger levels or a continuous range."""

    @property
    @abstractmethod
    def external_coupling(self) -> ExternalTriggerCoupling:
        """Configured external-trigger coupling."""

    @external_coupling.setter
    @abstractmethod
    def external_coupling(self, coupling: ExternalTriggerCoupling) -> None:
        """Select supported external-trigger coupling."""

    @property
    @abstractmethod
    def external_coupling_options(self) -> Collection[ExternalTriggerCoupling]:
        """Supported external-trigger coupling modes."""

    @property
    @abstractmethod
    def external_impedance(self) -> units.Resistance | ImpedanceMode:
        """Configured external-trigger impedance or mode."""

    @external_impedance.setter
    @abstractmethod
    def external_impedance(self, imp: units.Resistance | ImpedanceMode) -> None:
        """Select a supported external-trigger impedance."""

    @property
    @abstractmethod
    def external_impedance_options(
        self,
    ) -> Collection[units.Resistance | ImpedanceMode]:
        """Supported external-trigger impedances."""

    @property
    @abstractmethod
    def external_range(self) -> units.VoltageRange | ExternalTriggerRange:
        """Configured external-trigger voltage range or named mode."""

    @external_range.setter
    @abstractmethod
    def external_range(
        self,
        value: units.VoltageRange | ExternalTriggerRange,
    ) -> None:
        """Select a supported external-trigger range or named mode."""

    @property
    @abstractmethod
    def external_range_options(
        self,
    ) -> Collection[units.VoltageRange | ExternalTriggerRange]:
        """Supported external-trigger ranges or named modes."""


class Acquire(ABC):
    """Manage one acquisition at a time for the owning digitizer.

    Configuration is mutable before prepare(). From successful preparation
    until stop(), acquisition-affecting changes, including child component
    settings, raise RuntimeError.

    stop() releases per-run resources. Device.close() releases the persistent
    device connection. Methods need not support concurrent calls.
    """

    @property
    @abstractmethod
    def trigger_delay(self) -> int:
        """Trigger-relative delay in samples.

        Negative values select pre-trigger acquisition.
        Nonnegative values select post-trigger acquisition.
        """

    @trigger_delay.setter
    @abstractmethod
    def trigger_delay(self, value: int) -> None:
        """Set a delay within trigger_delay_range.

        Negative values must follow pre_trigger_delay_step.
        Nonnegative values must follow post_trigger_delay_step.
        """

    @property
    @abstractmethod
    def trigger_delay_range(self) -> units.IntRange:
        """Inclusive valid trigger-delay bounds."""

    @property
    @abstractmethod
    def pre_trigger_delay_step(self) -> int:
        """Trigger-delay granularity for negative delays, in samples."""

    @property
    @abstractmethod
    def post_trigger_delay_step(self) -> int:
        """Trigger-delay granularity for nonnegative delays, in samples."""

    @property
    @abstractmethod
    def record_length(self) -> int:
        """Number of samples per record."""

    @record_length.setter
    @abstractmethod
    def record_length(self, length: int) -> None:
        """Set a record length satisfying its minimum and step constraints."""

    @property
    @abstractmethod
    def record_length_minimum(self) -> int:
        """Minimum supported record length."""

    @property
    @abstractmethod
    def record_length_step(self) -> int:
        """Record-length granularity."""

    @property
    @abstractmethod
    def records_per_buffer(self) -> int:
        """Number of records in one complete logical batch."""

    @records_per_buffer.setter
    @abstractmethod
    def records_per_buffer(self, records: int) -> None:
        """Set a positive number of records per batch."""

    @property
    @abstractmethod
    def buffers_per_acquisition(self) -> int:
        """Number of complete batches, or -1 for unlimited acquisition."""

    @buffers_per_acquisition.setter
    @abstractmethod
    def buffers_per_acquisition(self, buffers: int) -> None:
        """Set a positive batch count, or -1 for unlimited acquisition."""

    @property
    @abstractmethod
    def buffers_allocated(self) -> int:
        """SDK transfer-buffer count, separate from worker output-pool capacity."""

    @buffers_allocated.setter
    @abstractmethod
    def buffers_allocated(self, buffers: int) -> None:
        """Set a transfer-buffer count satisfying driver-specific constraints."""

    @property
    @abstractmethod
    def timestamps_enabled(self) -> bool:
        """Whether record timestamps will accompany transferred batches.

        Unsupported devices report False.
        """

    @timestamps_enabled.setter
    @abstractmethod
    def timestamps_enabled(self, enable: bool) -> None:
        """Select timestamp capture.

        Raise NotImplementedError if enabling unsupported timestamps.
        Disabling timestamps must be supported.
        """

    @property
    @abstractmethod
    def buffers_acquired(self) -> int:
        """Complete batches successfully delivered by read_into().

        Reset by prepare() and retained after stop().
        """

    @property
    @abstractmethod
    def n_channels_enabled(self) -> int:
        """Current number of enabled channels."""

    @property
    @abstractmethod
    def layout(self) -> CaptureLayout:
        """Layout established by successful preparation.

        Raise RuntimeError before preparation or after stop().
        """

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """True after successful start until stop().

        This does not establish that triggers are arriving.
        """

    @abstractmethod
    def prepare(self) -> None:
        """Validate settings, establish layout, and prepare transfer resources.

        Require at least one enabled channel. Reset buffers_acquired to zero.
        Freeze settings for this run without starting sampling.

        Allocate what the SDK permits before start(); unavoidable allocation
        may occur during start().

        Raise ValueError for invalid configuration and RuntimeError if
        already prepared or started.

        After any failure, stop() must remain safe.
        """

    @abstractmethod
    def start(self) -> None:
        """Start the prepared acquisition, after all workers have prepared.

        Raise RuntimeError if unprepared, already started, or starting fails.

        After failure, stop() must remain safe even when is_active is False.
        """

    @abstractmethod
    def read_into(
        self,
        destination: SampleArray,
        *,
        timeout: units.Time | None = None,
    ) -> CaptureInfo | None:
        """Copy the next complete batch into caller-owned sample storage.

        Require a writable, C-contiguous array with exactly layout.shape
        and layout.dtype. Validate before consuming data.

        Do not alter destination's shape or dtype, or retain it after this
        call.

        Fill all records and enabled channels in corresponding event order.
        Return CaptureInfo and increment buffers_acquired once on success.

        Return None after all batches of a finite acquisition are delivered.
        Finite completion does not replace the required stop() cleanup.

        timeout must be finite and nonnegative, or None for unlimited waiting.
        Zero checks without waiting. The total waiting budget spans the
        whole batch; it must not restart for each channel or SDK buffer.

        After TimeoutError, destination contents are unspecified: do not
        publish them. Preserve partial progress internally so retry does
        not silently lose records or change batch ordering.

        Return or repost SDK buffers as required, including after copying
        or metadata decoding fails. Returned metadata must outlive SDK
        buffers.

        Raises:
            TypeError: An argument has an invalid type.
            ValueError: The array layout or timeout value is invalid.
            TimeoutError: A complete batch is unavailable within the budget.
            RuntimeError: Invalid lifecycle state or capture failure.

        Overflow or lost channel alignment is a failure, not normal finite
        completion.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop capture, release per-run resources, and invalidate layout.

        Safe before start(), after partial preparation, after failed start(),
        and after finite completion. Repeated calls are safe.

        Unfreeze settings and retain buffers_acquired until the next
        prepare(). Keep the device open and reusable.

        Concurrent invocation with read_into() is not required.
        """


class AuxiliaryIO(ABC):
    """Optional auxiliary I/O component owned by its digitizer.

    Acquisition-affecting changes are locked while prepared.
    Advertise only modes implemented by the driver.
    """

    @property
    @abstractmethod
    def mode_options(self) -> Collection[AuxiliaryIOMode]:
        """Implemented auxiliary I/O modes."""

    @abstractmethod
    def configure_mode(
        self,
        mode: AuxiliaryIOMode,
        **kwargs: object,
    ) -> None:
        """Select a supported mode using driver-documented parameters."""

    @abstractmethod
    def read_input(self) -> bool:
        """Read digital input; raise NotImplementedError if unsupported."""

    @abstractmethod
    def write_output(self, state: bool) -> None:
        """Write digital output; raise NotImplementedError if unsupported."""


class Digitizer(Device):
    """A device owning its channels, clock, trigger, and capture component.

    Component identities remain stable until close(). Constructors establish
    these attributes; concrete driver classes implement Device.close().

    close() stops any run and releases owned SDK resources, leaving the device
    object unusable. It does not imply powering off the hardware.

    One acquisition controls a digitizer at a time. Collection lookup does
    not provide exclusive ownership.

    Calls may move from the preparation thread to the source thread, but
    must not overlap.
    """

    input_mode: InputMode
    streaming_mode: StreamingMode

    channels: tuple[Channel, ...]
    sample_clock: SampleClock
    trigger: Trigger
    acquire: Acquire
    aux_io: AuxiliaryIO | None

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        """Hardware measurement resolution, distinct from storage width."""

    @property
    @abstractmethod
    def data_range(self) -> units.IntRange:
        """Inclusive bounds for sample codes delivered to consumers."""

    @property
    def is_active(self) -> bool:
        """Whether an acquisition has started and still requires stopping."""
        return self.acquire.is_active
