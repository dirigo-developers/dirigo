"""Test bounded capture reads, delivery counts, and metadata validation."""

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray
from dirigo.hardware.capture_session import CaptureState

from .capture_stubs import RecordingDigitizer, RecordingSession


@pytest.mark.parametrize(
    ("batch_count", "timestamps_enabled", "error_type", "message"),
    [
        (True, False, TypeError, "batch_count must be an integer"),
        (1.5, False, TypeError, "batch_count must be an integer"),
        (0, False, ValueError, "batch_count must be positive"),
        (-1, False, ValueError, "batch_count must be positive"),
        (None, 1, TypeError, "timestamps_enabled must be a bool"),
    ],
)
def test_invalid_delivery_settings_do_not_reserve_device(
    batch_count: object,
    timestamps_enabled: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid delivery limits and flags during task-free construction."""
    digitizer = RecordingDigitizer()
    try:
        with pytest.raises(error_type, match=message):
            RecordingSession(
                digitizer,
                batch_count=cast(int | None, batch_count),
                timestamps_enabled=cast(bool, timestamps_enabled),
            )
        digitizer.configure()
    finally:
        digitizer.close()


def test_finite_capture_completes_delivery_and_still_requires_cleanup() -> None:
    """Deliver exactly the requested batches without stopping the SDK run early.

    Further reads return None without entering the driver or modifying storage.
    Completion keeps the device reserved and active until stop; the delivered
    count and layout remain available after closing.
    """
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer, batch_count=2)
    try:
        assert session.batch_count == 2
        assert not session.timestamps_enabled
        assert session.batches_acquired == 0
        session.prepare()
        session.start()
        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)

        assert session.read_into(destination) is session.result
        assert session.state is CaptureState.STARTED
        assert session.batches_acquired == 1
        assert session.read_into(destination) is session.result
        assert session.state is CaptureState.COMPLETED
        assert session.batches_acquired == 2
        assert digitizer.is_active

        destination.fill(-99)
        assert session.read_into(destination, timeout=0.0) is None
        np.testing.assert_array_equal(destination, -99)
        assert session.events == ["prepare", "start", "read", "read"]

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.configure()

        session.stop()
        assert not digitizer.is_active
    finally:
        session.close()
        digitizer.close()

    assert session.batches_acquired == 2
    assert session.events == ["prepare", "start", "read", "read", "stop", "close"]


def _read_only_destination() -> SampleArray:
    """Return correctly shaped storage whose write permission is disabled."""
    destination = np.empty((2, 64, 2), dtype=np.int16)
    destination.setflags(write=False)
    return destination


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (object, TypeError, "must be a NumPy array"),
        (lambda: np.empty((1, 64, 2), dtype=np.int16), ValueError, "shape"),
        (lambda: np.empty((2, 64, 2), dtype=np.float64), ValueError, "dtype"),
        (_read_only_destination, ValueError, "writable"),
        (
            lambda: np.empty((2, 128, 2), dtype=np.int16)[:, ::2, :],
            ValueError,
            "C-contiguous",
        ),
    ],
)
def test_invalid_destination_consumes_nothing_and_leaves_capture_usable(
    factory: Callable[[], object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject unusable caller storage before consuming or counting a batch."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    try:
        session.prepare()
        session.start()

        with pytest.raises(error_type, match=message):
            session.read_into(cast(SampleArray, factory()))

        assert session.events == ["prepare", "start"]
        assert session.batches_acquired == 0
        assert session.state is CaptureState.STARTED

        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)
        assert session.read_into(destination) is session.result
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize(
    ("timeout", "error_type"),
    [
        (-1.0, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (True, TypeError),
        ("0.1", TypeError),
    ],
)
def test_invalid_timeout_does_not_consume_data(
    timeout: object,
    error_type: type[Exception],
) -> None:
    """Reject invalid waiting budgets without entering or closing the driver."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    try:
        session.prepare()
        session.start()
        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)

        with pytest.raises(error_type, match="timeout must be"):
            session.read_into(destination, timeout=cast(float, timeout))

        assert session.events == ["prepare", "start"]
        assert session.state is CaptureState.STARTED
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("timeout", [None, 0.0, 0.1, units.Time("100 ms")])
def test_unlimited_capture_forwards_wait_budget_and_keeps_delivering(
    timeout: float | None,
) -> None:
    """Forward seconds or unlimited waiting without imposing a finite batch limit."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    try:
        session.prepare()
        session.start()
        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)

        for _ in range(3):
            assert session.read_into(destination, timeout=timeout) is session.result

        expected = None if timeout is None else float(timeout)
        assert session.read_timeouts == [expected] * 3
        assert session.batch_count is None
        assert session.batches_acquired == 3
        assert session.state is CaptureState.STARTED
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize(
    ("enabled", "result", "error_type", "message"),
    [
        (True, CaptureInfo(), RuntimeError, "provide timestamps"),
        (True, CaptureInfo([0.0]), RuntimeError, "timestamp count"),
        (False, CaptureInfo([0.0, 0.1]), RuntimeError, "timestamps when disabled"),
        (False, object(), TypeError, "must return a CaptureInfo"),
        (False, None, TypeError, "must return a CaptureInfo"),
    ],
)
def test_invalid_driver_result_closes_without_counting_delivery(
    enabled: bool,
    result: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject malformed results and timestamp mismatches after driver transfer.

    Storage may have been written, but the batch is not deliverable. Close the
    session and release its reservation without incrementing the public count.
    """
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer, timestamps_enabled=enabled)
    session.result = cast(CaptureInfo, result)
    try:
        session.prepare()
        session.start()
        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)

        with pytest.raises(error_type, match=message):
            session.read_into(destination)

        assert session.state is CaptureState.CLOSED
        assert session.batches_acquired == 0
        assert session.events == ["prepare", "start", "read", "stop", "close"]
        digitizer.configure()
    finally:
        session.close()
        digitizer.close()


def test_enabled_timestamps_are_returned_with_a_complete_batch() -> None:
    """Accept detached timestamps matching the prepared record count."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer, batch_count=1, timestamps_enabled=True)
    session.result = CaptureInfo([0.0, 0.001])
    try:
        session.prepare()
        session.start()
        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)

        assert session.read_into(destination) is session.result
        assert session.timestamps_enabled
        assert session.batches_acquired == 1
        assert session.state is CaptureState.COMPLETED
    finally:
        session.close()
        digitizer.close()


class PartialSession(RecordingSession):
    """Stage one batch internally and time out after copying its first record."""

    def __init__(self, digitizer: RecordingDigitizer) -> None:
        super().__init__(digitizer, batch_count=1)
        self._pending: SampleArray | None = None
        self._timed_out = False

    def _prepare_capture(self) -> CaptureLayout:
        """Allocate private staging after obtaining the fake driver's layout."""
        layout = super()._prepare_capture()
        self._pending = np.arange(np.prod(layout.shape), dtype=layout.dtype).reshape(
            layout.shape
        )
        return layout

    def _read_into(
        self, destination: SampleArray, *, timeout: float | None
    ) -> CaptureInfo:
        """Preserve pending data independently of the supplied destination."""
        self._step("read")
        self.read_timeouts.append(timeout)
        assert self._pending is not None

        if not self._timed_out:
            destination[0] = self._pending[0]
            self._timed_out = True
            raise TimeoutError("Batch is incomplete.")

        np.copyto(destination, self._pending)
        self._pending = None
        return CaptureInfo()

    def _close_capture(self) -> None:
        """Release private staging and record normal driver cleanup."""
        self._pending = None
        super()._close_capture()


def test_timeout_retry_can_use_different_storage_without_losing_records() -> None:
    """Leave capture active after a partial transfer and retry into another lease.

    Overwrite the first destination after timeout, as a recycled worker buffer
    might be overwritten. The retry must still deliver the original complete
    batch from driver-owned staging and count it exactly once.
    """
    digitizer = RecordingDigitizer()
    session = PartialSession(digitizer)
    try:
        session.prepare()
        session.start()
        first = np.full(session.layout.shape, -99, dtype=session.layout.dtype)

        with pytest.raises(TimeoutError, match="incomplete"):
            session.read_into(first, timeout=0.05)

        assert session.state is CaptureState.STARTED
        assert session.batches_acquired == 0
        assert session.events == ["prepare", "start", "read"]
        first.fill(-777)

        second = np.empty_like(first)
        assert session.read_into(second, timeout=0.1) is not None
        expected = np.arange(second.size, dtype=second.dtype).reshape(second.shape)

        np.testing.assert_array_equal(second, expected)
        np.testing.assert_array_equal(first, -777)
        assert session.read_timeouts == [0.05, 0.1]
        assert session.batches_acquired == 1
        assert session.state is CaptureState.COMPLETED
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_read_failure_preserves_primary_error_and_releases_device(
    cleanup_fails: bool,
) -> None:
    """Preserve delivered counts and the read error through failed cleanup."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    failure = RuntimeError("Read failed.")
    if cleanup_fails:
        session.failures["stop"] = RuntimeError("Stop failed.")
        session.failures["close"] = RuntimeError("Cleanup failed.")
    try:
        session.prepare()
        session.start()
        destination = np.empty(session.layout.shape, dtype=session.layout.dtype)
        assert session.read_into(destination) is session.result

        session.failures["read"] = failure
        with pytest.raises(RuntimeError, match="Read failed") as caught:
            session.read_into(destination)

        assert caught.value is failure
        assert session.state is CaptureState.CLOSED
        assert session.batches_acquired == 1
        if cleanup_fails:
            assert any("Stop failed" in note for note in failure.__notes__)
            assert any("Cleanup failed" in note for note in failure.__notes__)
        digitizer.configure()
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("phase", ["new", "prepared", "stopped", "closed"])
def test_read_requires_a_started_or_completed_session(phase: str) -> None:
    """Reject data transfer outside the active delivery lifecycle."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    try:
        if phase != "new":
            session.prepare()
        if phase == "stopped":
            session.start()
            session.stop()
        if phase == "closed":
            session.close()

        destination = np.empty((2, 64, 2), dtype=np.int16)
        with pytest.raises(RuntimeError, match="must be started"):
            session.read_into(destination)
        assert "read" not in session.events
    finally:
        session.close()
        digitizer.close()
