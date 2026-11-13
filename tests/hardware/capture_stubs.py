"""Recording digitizer and capture-session fakes shared by core tests."""

import numpy as np

from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.capture_session import DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer


class RecordingDigitizer(Digitizer):
    """Provide persistent-device bookkeeping without an SDK connection."""

    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0
        self.configure_calls = 0
        self.close_failure: RuntimeError | None = None

    @property
    def bit_depth(self) -> int:
        """Report a fixed measurement resolution for this test device."""
        return 16

    @property
    def data_range(self) -> units.IntRange:
        """Report signed sample-code limits."""
        return units.IntRange(-32768, 32767)

    def configure(self) -> None:
        """Exercise the guard a driver uses for public configuration changes."""
        with self._capture_lock:
            self._require_configurable()
            self.configure_calls += 1

    def _close(self) -> None:
        """Record persistent cleanup and optionally fail."""
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure

    @property
    def channel_count(self) -> int:
        """Expose two physical inputs for capture-factory tests."""
        return 2

    def _create_capture(self, settings: CaptureSettings) -> DigitizerCaptureSession:
        """Construct a recording session using the requested batch geometry."""
        session = RecordingSession(
            self,
            batch_count=settings.batch_count,
            timestamps_enabled=settings.timestamps_enabled,
        )
        session.prepared_layout = CaptureLayout(
            shape=(
                settings.records_per_batch,
                settings.record_length,
                len(settings.channel_indices),
            ),
            channel_indices=settings.channel_indices,
            dtype=np.dtype(np.int16),
        )
        return session


class RecordingSession(DigitizerCaptureSession):
    """Record driver hooks and allow injected failures at lifecycle boundaries."""

    def __init__(
        self,
        digitizer: RecordingDigitizer,
        *,
        batch_count: int | None = None,
        timestamps_enabled: bool = False,
    ) -> None:
        super().__init__(
            digitizer,
            batch_count=batch_count,
            timestamps_enabled=timestamps_enabled,
        )
        self.result = CaptureInfo()
        self.read_timeouts: list[float | None] = []
        self.events: list[str] = []
        self.failures: dict[str, RuntimeError] = {}
        self.prepared_layout = CaptureLayout(
            shape=(2, 64, 2),
            channel_indices=(1, 0),
            dtype=np.dtype(np.int16),
        )

    def _read_into(
        self, destination: SampleArray, *, timeout: float | None
    ) -> CaptureInfo:
        """Fill one fake batch and return its configured capture metadata."""
        self._step("read")
        self.read_timeouts.append(timeout)
        destination.fill(0)
        return self.result

    def _step(self, name: str) -> None:
        """Record a hook and raise its configured failure, if any."""
        self.events.append(name)
        if name in self.failures:
            raise self.failures[name]

    def _prepare_capture(self) -> CaptureLayout:
        """Return the test-controlled effective layout."""
        self._step("prepare")
        return self.prepared_layout

    def _start_capture(self) -> None:
        """Record starting or arming capture."""
        self._step("start")

    def _stop_capture(self) -> None:
        """Record stopping a potentially started capture."""
        self._step("stop")

    def _close_capture(self) -> None:
        """Record release of this session's per-run resources."""
        self._step("close")
