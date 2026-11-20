"""Reusable digitizer devices, immutable profiles, and capture construction.

Profiles describe requested acquisition settings. Capture sessions snapshot
those requests, resolve capture geometry, and own each acquisition's resources.

Device construction and capture construction are passive. Explicit constraint
resolution and capture preparation may access hardware.

Calls into one digitizer remain sequential, although preparation and execution
may occur on different threads. Capture reservations prevent conflicting
configuration changes until the owning session closes.
"""

from abc import abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from os import PathLike
from threading import RLock
from typing import TYPE_CHECKING

from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.device import Device

if TYPE_CHECKING:
    from dirigo.hardware.capture_session import DigitizerCaptureSession

__all__ = [
    "CaptureInfo",
    "CaptureLayout",
    "ChannelCoupling",
    "ChannelProfile",
    "Digitizer",
    "DigitizerProfile",
    "ExternalTriggerCoupling",
    "ExternalTriggerRange",
    "ImpedanceMode",
    "SampleArray",
    "SampleClockEdge",
    "SampleClockProfile",
    "SampleClockSource",
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


@dataclass(frozen=True, slots=True)
class SampleClockProfile:
    """Requested clock settings."""

    source: SampleClockSource
    rate: units.SampleRate
    edge: SampleClockEdge = SampleClockEdge.RISING


@dataclass(frozen=True, slots=True)
class ChannelProfile:
    """Requested channel configuration for one capture.

    None selects the driver's documented default for an optional setting.
    Drivers must reject requested values they cannot honor.
    """

    enabled: bool
    coupling: ChannelCoupling | None = None
    impedance: units.Resistance | ImpedanceMode | None = None
    input_range: units.VoltageRange | None = None
    offset: units.Voltage | None = None
    inverted: bool = False


@dataclass(frozen=True, slots=True)
class TriggerProfile:
    """Requested record-trigger configuration for one capture.

    Optional electrical settings use documented driver defaults when None.
    Drivers must reject unsupported sources, levels, or electrical settings.
    """

    source: TriggerSource
    slope: TriggerSlope
    level: units.Voltage | None = None
    external_coupling: ExternalTriggerCoupling | None = None
    external_impedance: units.Resistance | ImpedanceMode | None = None
    external_range: units.VoltageRange | ExternalTriggerRange | None = None


@dataclass(frozen=True, slots=True)
class DigitizerProfile:
    """Stored defaults for constructing independent capture sessions.

    Profiles contain sampling, channel, and trigger settings. Record
    geometry and acquisition length are configured on each capture session.

    Drivers resolve omitted optional settings to their defaults and reject
    explicitly requested settings they cannot support.
    """

    sample_clock: SampleClockProfile
    channels: tuple[ChannelProfile, ...]
    trigger: TriggerProfile


class Digitizer(Device):
    """Own a reusable digitizer connection and construct single-use captures.

    create_capture() accepts complete capture settings or snapshots the stored
    profile. Profile-based sessions receive their remaining geometry through
    configure(). Native acquisition programming belongs to capture preparation.

    A session reserves this logical digitizer during explicit constraint
    resolution or preparation and retains that reservation until close().
    The reservation covers this input operation, rather than an entire physical
    DAQ card shared with other logical devices.

    Stored-profile changes must hold _capture_lock and call
    _require_configurable(). Existing sessions retain their captured request.

    Close borrowing sessions before closing the digitizer. A reserved device
    rejects closure and remains open. Concrete drivers call super().__init__()
    and implement _close() for their persistent resources.
    """

    def __init__(self) -> None:
        """Initialize lifecycle bookkeeping without accessing hardware."""
        super().__init__()
        self._profile: DigitizerProfile | None = None
        self._capture_lock = RLock()
        self._capture_owner: DigitizerCaptureSession | None = None
        self._closed = False

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        """Hardware measurement resolution, distinct from storage width."""

    @property
    @abstractmethod
    def data_range(self) -> units.IntRange:
        """Inclusive bounds for sample codes delivered to consumers."""

    @property
    def profile(self) -> DigitizerProfile:
        """Stored defaults used by future profile-based capture sessions."""
        with self._capture_lock:
            if self._closed:
                raise RuntimeError("Digitizer is closed.")
            if self._profile is None:
                raise RuntimeError("No digitizer profile has been configured.")
            return self._profile

    @profile.setter
    def profile(self, value: DigitizerProfile) -> None:
        """Replace defaults without changing existing capture sessions."""
        from dirigo.hardware.profiles import validate_digitizer_profile

        validate_digitizer_profile(value)

        with self._capture_lock:
            self._require_configurable()
            self._profile = value

    @staticmethod
    def read_profile(
        path: str | PathLike[str],
    ) -> DigitizerProfile:
        """Read stored defaults without constructing or accessing hardware.

        Drivers may override this to parse additional vendor-specific settings.
        """
        from dirigo.hardware.profiles import load_digitizer_profile

        return load_digitizer_profile(path)

    @property
    def is_active(self) -> bool:
        """Whether the reserved session has started and has not stopped.

        True does not establish that triggers or samples are arriving.
        """
        with self._capture_lock:
            owner = self._capture_owner
            return owner is not None and owner.is_active

    def close(self) -> None:
        """Release persistent resources after borrowing sessions have closed.

        Closure is permanent even if the driver hook fails. Repeated calls do
        nothing. A reservation error leaves the device open.
        """
        with self._capture_lock:
            if self._closed:
                return
            if self._capture_owner is not None:
                raise RuntimeError("Digitizer is reserved for capture.")
            self._closed = True
            self._close()

    def _require_configurable(self) -> None:
        """Check usability while the caller holds _capture_lock."""
        if self._closed:
            raise RuntimeError("Digitizer is closed.")
        if self._capture_owner is not None:
            raise RuntimeError("Digitizer is reserved for capture.")

    def _reserve_capture(self, owner: DigitizerCaptureSession) -> None:
        """Reserve this input operation for one capture session."""
        with self._capture_lock:
            self._require_configurable()
            self._capture_owner = owner

    def _release_capture(self, owner: DigitizerCaptureSession) -> None:
        """Release only the reservation belonging to the supplied session."""
        with self._capture_lock:
            if self._capture_owner is owner:
                self._capture_owner = None

    @abstractmethod
    def _close(self) -> None:
        """Release persistent resources owned by this logical digitizer."""

    @property
    @abstractmethod
    def channel_count(self) -> int:
        """Number of inputs, or raise RuntimeError until discovery is available."""

    def create_capture(
        self,
        settings: CaptureSettings | None = None,
    ) -> DigitizerCaptureSession:
        """Construct an unprepared capture without accessing hardware.

        With explicit settings, retain the existing complete-request API.

        With no settings, snapshot the stored profile. The returned session
        requires configure() before prepare().
        """
        if settings is not None and not isinstance(settings, CaptureSettings):
            raise TypeError("settings must be CaptureSettings or None.")

        with self._capture_lock:
            if self._closed:
                raise RuntimeError("Digitizer is closed.")

            if settings is not None:
                self._validate_channel_indices(settings.channel_indices)
                return self._create_capture(settings)

            profile = self.profile
            indices = tuple(
                index
                for index, channel in enumerate(profile.channels)
                if channel.enabled
            )
            self._validate_channel_indices(indices)
            return self._create_profile_capture(profile)

    @abstractmethod
    def _create_capture(self, settings: CaptureSettings) -> DigitizerCaptureSession:
        """Validate backend support and construct an unprepared session."""

    def _create_profile_capture(
        self,
        profile: DigitizerProfile,
    ) -> DigitizerCaptureSession:
        """Construct a session whose geometry will be configured later."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support stored profiles yet."
        )

    def _settings_from_profile(
        self,
        profile: DigitizerProfile,
        *,
        record_length: int,
        records_per_batch: int,
        batch_count: int | None,
        timestamps_enabled: bool,
        trigger_offset_samples: int,
    ) -> CaptureSettings:
        """Resolve a profile and geometry into complete driver settings.

        Perform validation only. Do not access hardware or reserve resources.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support stored profiles yet."
        )

    def _validate_channel_indices(self, indices: tuple[int, ...]) -> None:
        """Reject indices beyond known inputs before constructing a session.

        Drivers requiring discovery may defer this check until preparation.
        They must then validate physical bounds before configuring inputs.
        CaptureSettings has already validated tuple contents and uniqueness.
        """
        count = self.channel_count
        if any(index >= count for index in indices):
            raise ValueError("channel_indices exceed the available inputs.")
