"""Single-use grouped galvo waveform-output sessions."""

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

from dirigo.hardware.scanner import GalvoScanner
from dirigo.hardware.waveform import AngularWaveform


class OutputState(StrEnum):
    """Lifecycle states of a single-use output session."""

    NEW = "new"
    PREPARED = "prepared"
    STARTED = "started"
    COMPLETED = "completed"
    STOPPED = "stopped"
    CLOSED = "closed"


class GalvoOutputSession(ABC):
    """Own one finite output run while borrowing its participating axes.

    Construction validates and copies the binding map without reserving axes
    or accessing hardware. Waveforms must have identical nominal sample rates
    and sample counts.

    prepare() reserves the axes until close(), including after completion or
    stop(). Session methods must be called serially. For cooperative
    cancellation, use bounded wait() calls before calling stop().

    Subclasses implement output hooks using _waveforms, _sample_rate, and
    _sample_count. Constructors must remain free of hardware operations.
    Hardware devices must outlive their sessions.
    """

    def __init__(
        self,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
    ) -> None:
        """Validate the complete group without changing device state."""
        if not isinstance(waveforms, Mapping):
            raise TypeError("waveforms must be a mapping.")
        bindings = dict(waveforms)
        if not bindings:
            raise ValueError("At least one scanner waveform is required.")

        for axis, waveform in bindings.items():
            if not isinstance(axis, GalvoScanner):
                raise TypeError("Waveform keys must be GalvoScanner instances.")
            axis.validate_waveform(waveform)

        first = next(iter(bindings.values()))
        self._sample_rate = first.sample_rate
        self._sample_count = first.samples.size
        for waveform in bindings.values():
            if waveform.sample_rate != self._sample_rate:
                raise ValueError("Waveforms must have the same sample_rate.")
            if waveform.samples.size != self._sample_count:
                raise ValueError("Waveforms must have the same sample count.")

        self._waveforms = MappingProxyType(bindings)
        self._reserved: list[GalvoScanner] = []
        self._state = OutputState.NEW

    @property
    def state(self) -> OutputState:
        """Last observed lifecycle state; wait() observes finite completion."""
        return self._state

    def prepare(self) -> None:
        """Reserve every axis and prepare output without starting the run.

        Preparation failure closes the session and releases acquired
        reservations, preserving the original exception.
        """
        self._require_state(OutputState.NEW)
        try:
            for axis in self._waveforms:
                axis._reserve_output(self)
                self._reserved.append(axis)
            self._prepare_output()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        self._state = OutputState.PREPARED

    def start(self) -> None:
        """Start or arm the prepared run exactly once."""
        self._require_state(OutputState.PREPARED)
        self._state = OutputState.STARTED
        try:
            self._start_output()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def wait(self, timeout: float | None = None) -> bool:
        """Observe completion, waiting at most timeout seconds.

        None permits an unbounded wait. A finite timeout must be nonnegative.
        Return True on completion, or False on timeout or after stop().
        A timeout leaves the session started and reservations held.
        """
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative.")
        if self._state is OutputState.COMPLETED:
            return True
        if self._state is OutputState.STOPPED:
            return False
        self._require_state(OutputState.STARTED)
        try:
            completed = self._wait_output(timeout)
        except BaseException as failure:
            self._close_after_failure(failure)
            raise
        if completed:
            self._state = OutputState.COMPLETED
        return completed

    def stop(self) -> None:
        """Cancel an unfinished run, retaining reservations until close().

        Repeated calls and calls after completion or closure do nothing.
        No static angle is commanded. Failure closes the session and
        preserves the original exception.
        """
        try:
            self._stop_unfinished()
        except BaseException as failure:
            self._close_after_failure(failure)
            raise

    def _stop_unfinished(self) -> None:
        """Cancel the run, invoking the output hook only after start."""
        if self._state in (
            OutputState.COMPLETED,
            OutputState.STOPPED,
            OutputState.CLOSED,
        ):
            return
        started = self._state is OutputState.STARTED
        self._state = OutputState.STOPPED
        if started:
            self._stop_output()

    def close(self) -> None:
        """Stop output, attempt cleanup, and release every axis reservation.

        Cleanup is attempted even if stopping fails. Preserve the first
        failure and attach any additional cleanup error as a note.
        Closure is permanent and idempotent.
        """
        if self._state is OutputState.CLOSED:
            return
        failure: BaseException | None = None
        try:
            self._stop_unfinished()
        except BaseException as exc:
            failure = exc
        try:
            self._close_output()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Output cleanup also failed: {exc!r}")
        finally:
            for axis in reversed(self._reserved):
                axis._release_output(self)
            self._reserved.clear()
            self._state = OutputState.CLOSED
        if failure is not None:
            raise failure

    def _require_state(self, expected: OutputState) -> None:
        """Reject an operation outside its required lifecycle state."""
        if self._state is not expected:
            raise RuntimeError(
                f"Session must be {expected.value}; it is {self._state.value}."
            )

    def _close_after_failure(self, failure: BaseException) -> None:
        """Release resources without replacing an operation's original error."""
        try:
            self.close()
        except BaseException as cleanup:
            notes = tuple(getattr(cleanup, "__notes__", ()))
            failure.add_note(f"Output cleanup also failed: {cleanup!r}")
            for note in notes:
                failure.add_note(note)

    @abstractmethod
    def _prepare_output(self) -> None:
        """Prepare resources and preload samples without starting output."""

    @abstractmethod
    def _start_output(self) -> None:
        """Start or arm output; failure may indicate a partial start."""

    @abstractmethod
    def _wait_output(self, timeout: float | None) -> bool:
        """Wait for finite completion, returning False on timeout."""

    @abstractmethod
    def _stop_output(self) -> None:
        """Stop output, including after a partially failed start."""

    @abstractmethod
    def _close_output(self) -> None:
        """Release owned resources, including after partial preparation."""
