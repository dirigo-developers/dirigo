"""Raster-frame synchronization contracts, independent of vendor hardware."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING

from dirigo.hardware.synchronization import SyncBackend

if TYPE_CHECKING:
    from dirigo.hardware.capture_session import DigitizerCaptureSession
    from dirigo.hardware.scanner import BeamScanner, GalvoScanner
    from dirigo.hardware.waveform import AngularWaveform


class RasterFrameSyncState(StrEnum):
    """Lifecycle of one synchronization session.

    Capture completion is reported by the capture session. Synchronization
    remains STARTED until explicitly stopped or closed.
    """

    NEW = "new"
    CONFIGURED = "configured"
    PREPARED = "prepared"
    STARTED = "started"
    STOPPED = "stopped"
    CLOSED = "closed"


class RasterFrameSyncSession(ABC):
    """Coordinate one acquisition while borrowing capture and scanner devices.

    The worker owns this session and its capture session separately.

    Construction and configuration perform no hardware operations.
    prepare() prepares capture and synchronization resources without starting
    capture or timed scanner output. start() arms or starts participants in
    dependency order.

    stop() cancels activity without waiting for unfinished scanner waveforms.
    close() also releases resources owned by synchronization.

    Neither operation closes the borrowed capture or scanner devices. The
    worker must close capture after closing synchronization.

    Calls are serial. Preparation and execution may use different threads.
    """

    @property
    @abstractmethod
    def state(self) -> RasterFrameSyncState:
        """Current synchronization lifecycle state."""

    @abstractmethod
    def configure(
        self,
        *,
        records_per_frame: int,
        frame_count: int | None,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
    ) -> None:
        """Configure frame grouping and scanner output without hardware I/O.

        Each waveform represents one frame. None requests unlimited frames;
        a backend may reject unsupported acquisition modes.
        """

    @abstractmethod
    def prepare(self) -> None:
        """Prepare configured participants without starting acquisition."""

    @abstractmethod
    def start(self) -> None:
        """Arm and start participants in dependency order."""

    @abstractmethod
    def stop(self) -> None:
        """Stop activity, retaining reservations until resources are closed."""

    @abstractmethod
    def close(self) -> None:
        """Stop and release owned resources, leaving borrowed objects open.

        Permit calls after partial preparation or startup failure.
        Closure is permanent and idempotent.
        """


class RasterFrameSync(SyncBackend):
    """A configured backend that creates independent raster sync sessions."""

    @abstractmethod
    def create_session(
        self,
        *,
        fast_scanner: BeamScanner,
        slow_scanner: BeamScanner,
        capture: DigitizerCaptureSession,
    ) -> RasterFrameSyncSession:
        """Validate participants and construct a passive session."""
