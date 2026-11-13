"""Single-use capture sessions borrowing reusable digitizers."""

import math
from abc import ABC, abstractmethod
from enum import StrEnum
from numbers import Real

import numpy as np

from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray
from dirigo.hardware.digitizer import Digitizer


class CaptureState(StrEnum):
    """Lifecycle states of a single-use digitizer capture session."""

    NEW = "new"
    PREPARED = "prepared"
    STARTED = "started"
    COMPLETED = "completed"
    STOPPED = "stopped"
    CLOSED = "closed"


class DigitizerCaptureSession(ABC):
    """Own one capture run while borrowing a logical digitizer.

    Construction performs no hardware operations or reservation. prepare()
    reserves the digitizer and establishes a layout without starting capture.
    start() starts or arms the run; STARTED does not establish trigger arrival.

    stop() cancels an unfinished run but retains its reservation and resources
    until close(). Closure is permanent and idempotent. A successful layout
    remains available after stopping or closing as immutable descriptive data.

    Session methods must be called serially. The digitizer must outlive every
    session that borrows it. Driver hooks own per-run resources only.

    batch_count limits delivery to a positive number of complete batches;
    None permits unlimited delivery. COMPLETED means all requested batches
    were delivered. The SDK run still requires stopping and cleanup.
    """

    def __init__(
        self,
        digitizer: Digitizer,
        *,
        batch_count: int | None = None,
        timestamps_enabled: bool = False,
    ) -> None:
        """Validate delivery settings without reserving or configuring hardware."""
        if not isinstance(digitizer, Digitizer):
            raise TypeError("digitizer must be a Digitizer.")
        if batch_count is not None:
            if not isinstance(batch_count, int) or isinstance(batch_count, bool):
                raise TypeError("batch_count must be an integer or None.")
            if batch_count <= 0:
                raise ValueError("batch_count must be positive.")
        if not isinstance(timestamps_enabled, bool):
            raise TypeError("timestamps_enabled must be a bool.")

        self._batch_count = batch_count
        self._timestamps_enabled = timestamps_enabled
        self._batches_acquired = 0
        self._digitizer = digitizer
        self._state = CaptureState.NEW
        self._layout: CaptureLayout | None = None
        self._reserved = False

    @property
    def digitizer(self) -> Digitizer:
        """The borrowed digitizer; ownership remains with its caller."""
        return self._digitizer

    @property
    def state(self) -> CaptureState:
        """Last lifecycle state observed by this session."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether a started SDK run still requires stopping."""
        return self._state in (CaptureState.STARTED, CaptureState.COMPLETED)

    @property
    def layout(self) -> CaptureLayout:
        """Prepared layout, retained after close; unavailable before preparation."""
        if self._layout is None:
            raise RuntimeError(
                "Capture layout is unavailable before successful preparation."
            )
        return self._layout

    def prepare(self) -> None:
        """Reserve input resources and establish layout without starting capture."""
        self._require_state(CaptureState.NEW)
        try:
            self._digitizer._reserve_capture(self)
            self._reserved = True
            layout = self._prepare_capture()
            if not isinstance(layout, CaptureLayout):
                raise TypeError("_prepare_capture() must return a CaptureLayout.")
            self._layout = layout
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        self._state = CaptureState.PREPARED

    def start(self) -> None:
        """Start or arm once, returning without waiting for incoming triggers."""
        self._require_state(CaptureState.PREPARED)
        self._state = CaptureState.STARTED
        try:
            self._start_capture()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def stop(self) -> None:
        """Cancel capture while retaining resources until close()."""
        try:
            self._stop_unfinished()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def _stop_unfinished(self) -> None:
        """Stop a started run, including partial start or completed delivery."""
        if self._state in (CaptureState.STOPPED, CaptureState.CLOSED):
            return
        started = self.is_active
        self._state = CaptureState.STOPPED
        if started:
            self._stop_capture()

    def close(self) -> None:
        """Attempt stop and cleanup, release the reservation, and close permanently.

        A session that never acquired its reservation invokes no driver hooks.
        Preserve the first cleanup failure and attach additional errors as notes.
        """
        if self._state is CaptureState.CLOSED:
            return
        if not self._reserved:
            self._state = CaptureState.CLOSED
            return

        failure: BaseException | None = None
        try:
            self._stop_unfinished()
        except BaseException as exc:
            failure = exc
        try:
            self._close_capture()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Capture cleanup also failed: {exc!r}")
        finally:
            self._digitizer._release_capture(self)
            self._reserved = False
            self._state = CaptureState.CLOSED
        if failure is not None:
            raise failure

    def _require_state(self, expected: CaptureState) -> None:
        """Reject lifecycle operations outside their required state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Session must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Preserve an operation error while attempting all owned cleanup."""
        try:
            self.close()
        except BaseException as cleanup:
            notes = tuple(getattr(cleanup, "__notes__", ()))
            failure.add_note(f"Capture cleanup also failed: {cleanup!r}")
            for note in notes:
                failure.add_note(note)

    @abstractmethod
    def _prepare_capture(self) -> CaptureLayout:
        """Prepare per-run resources and return effective layout without starting."""

    @abstractmethod
    def _start_capture(self) -> None:
        """Establish backend readiness without waiting for record-trigger arrival.

        On successful return, a coordinator may enable the trigger producer.
        An exception may indicate a partial start and requires cleanup.
        """

    @abstractmethod
    def _stop_capture(self) -> None:
        """Stop capture, including after a partially failed start."""

    @abstractmethod
    def _close_capture(self) -> None:
        """Release owned run resources, including after partial preparation."""

    @property
    def batch_count(self) -> int | None:
        """Requested complete-batch count, or None for unlimited delivery."""
        return self._batch_count

    @property
    def timestamps_enabled(self) -> bool:
        """Whether every delivered batch must include record timestamps."""
        return self._timestamps_enabled

    @property
    def batches_acquired(self) -> int:
        """Complete batches successfully returned, retained after stop or close."""
        return self._batches_acquired

    def read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None = None,
    ) -> CaptureInfo | None:
        """Deliver one complete batch into caller-owned sample storage.

        Require STARTED or COMPLETED state and a writable, C-contiguous array
        with exactly layout.shape and layout.dtype. Validate arguments before
        invoking the driver, including when returning finite exhaustion.

        timeout is a total waiting budget in seconds: None waits indefinitely
        and zero polls. TimeoutError leaves capture active and its delivered
        count unchanged. Destination contents are then unspecified; do not
        publish them. A retry may supply a different destination array.

        Return CaptureInfo on success, or None once batch_count batches have
        been delivered. Completion retains input resources until close().
        Unexpected driver or result-validation failures close the session.
        Argument-validation errors leave it usable without consuming data.
        """
        if self._state is not CaptureState.COMPLETED:
            self._require_state(CaptureState.STARTED)

        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, Real):
                raise TypeError("timeout must be a real number of seconds or None.")
            timeout = float(timeout)
            if not math.isfinite(timeout) or timeout < 0:
                raise ValueError("timeout must be finite and nonnegative.")
        self._validate_destination(destination)

        if self._state is CaptureState.COMPLETED:
            return None

        try:
            info = self._read_into(destination, timeout=timeout)
            if not isinstance(info, CaptureInfo):
                raise TypeError("_read_into() must return a CaptureInfo.")

            timestamps = info.timestamps
            if self._timestamps_enabled:
                if timestamps is None:
                    raise RuntimeError("Capture must provide timestamps when enabled.")
                if timestamps.size != self.layout.records_per_batch:
                    raise RuntimeError(
                        "Capture timestamp count must match records_per_batch."
                    )
            elif timestamps is not None:
                raise RuntimeError("Capture returned timestamps when disabled.")
        except TimeoutError:
            raise
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

        self._batches_acquired += 1
        if (
            self._batch_count is not None
            and self._batches_acquired == self._batch_count
        ):
            self._state = CaptureState.COMPLETED
        return info

    def _validate_destination(self, destination: SampleArray) -> None:
        """Reject unusable caller storage before any driver data is consumed."""
        if not isinstance(destination, np.ndarray):
            raise TypeError("destination must be a NumPy array.")
        if destination.shape != self.layout.shape:
            raise ValueError("destination shape must match capture layout.")
        if destination.dtype != self.layout.dtype:
            raise ValueError("destination dtype must match capture layout.")
        if not destination.flags.writeable:
            raise ValueError("destination must be writable.")
        if not destination.flags.c_contiguous:
            raise ValueError("destination must be C-contiguous.")

    @abstractmethod
    def _read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        """Fill one complete batch within a single total waiting budget.

        Return CaptureInfo only after filling the whole destination. Do not
        change its shape or dtype, or retain it after returning or raising.

        On TimeoutError, retain partial progress in driver-owned storage or
        still-owned SDK buffers. A retry can supply a different destination;
        preserve batch ordering and do not silently discard records.

        Return or repost SDK buffers as required, including after copy or
        metadata-decoding errors. Unexpected SDK exhaustion or data loss must
        raise an error. Finite exhaustion is managed by the base batch counter.
        """
