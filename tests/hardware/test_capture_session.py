"""Test capture lifecycle, exclusive ownership, and failure cleanup."""

from collections.abc import Callable
from typing import cast

import pytest

from dirigo.hardware.capture import CaptureLayout
from dirigo.hardware.capture_session import CaptureState

from .capture_stubs import RecordingDigitizer, RecordingSession


def test_unused_session_does_not_reserve_or_call_driver_hooks() -> None:
    """Construct and discard a session while leaving its borrowed device usable."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)

    assert session.digitizer is digitizer
    assert session.state is CaptureState.NEW
    assert not session.is_active
    assert not digitizer.is_active
    with pytest.raises(RuntimeError, match="layout is unavailable"):
        _ = session.layout
    digitizer.configure()

    session.close()
    session.close()
    session.stop()

    assert session.state is CaptureState.CLOSED
    assert session.events == []
    assert digitizer.close_calls == 0
    digitizer.configure()
    digitizer.close()


def test_lifecycle_retains_reservation_until_close_and_preserves_layout() -> None:
    """Reserve through stop, retain descriptive layout, and allow a fresh run.

    Prepared and stopped sessions still block configuration and device closure.
    Closing releases ownership without closing the device, and the same layout
    remains readable. A new session can then use that digitizer.
    """
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    try:
        session.prepare()
        layout = session.layout
        assert layout is session.prepared_layout
        assert session.state is CaptureState.PREPARED
        assert not digitizer.is_active

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.configure()
        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.close()

        session.start()
        assert session.is_active
        assert digitizer.is_active
        session.stop()
        session.stop()

        assert session.state is CaptureState.STOPPED
        assert not digitizer.is_active
        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.configure()
    finally:
        session.close()

    assert session.events == ["prepare", "start", "stop", "close"]
    assert session.layout is layout
    assert digitizer.close_calls == 0
    digitizer.configure()
    with pytest.raises(RuntimeError, match="must be new"):
        session.prepare()

    following = RecordingSession(digitizer)
    try:
        following.prepare()
        following.start()
    finally:
        following.close()
        digitizer.close()


@pytest.mark.parametrize("started", [False, True])
def test_close_stops_only_started_captures(started: bool) -> None:
    """Always release prepared resources, stopping only after a start request."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    session.prepare()
    if started:
        session.start()

    session.close()
    session.close()

    expected = (
        ["prepare", "start", "stop", "close"] if started else ["prepare", "close"]
    )
    assert session.events == expected
    assert session.state is CaptureState.CLOSED
    assert not digitizer.is_active
    digitizer.configure()
    digitizer.close()


def test_competing_session_cannot_disturb_existing_owner() -> None:
    """Reject a second capture without invoking hooks or releasing another owner.

    The first session is active. The contender must close without native
    cleanup, and even an attempted release under the wrong identity must leave
    the first session's reservation intact.
    """
    digitizer = RecordingDigitizer()
    owner = RecordingSession(digitizer)
    contender = RecordingSession(digitizer)
    try:
        owner.prepare()
        owner.start()
        with pytest.raises(RuntimeError, match="reserved"):
            contender.prepare()

        assert contender.state is CaptureState.CLOSED
        assert contender.events == []
        digitizer._release_capture(contender)
        assert digitizer.is_active
        assert owner.events == ["prepare", "start"]

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.close()
    finally:
        contender.close()
        owner.close()
        digitizer.close()


@pytest.mark.parametrize("operation", ["prepare", "start", "stop"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_operation_failure_closes_session_and_preserves_primary_error(
    operation: str,
    cleanup_fails: bool,
) -> None:
    """Release ownership after partial preparation, partial start, or stop failure.

    A failed start may still need stopping. Inject additional stop and cleanup
    failures to prove that their diagnostics survive without replacing the
    original operation error. Closing the failed session again is harmless.
    """
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    failure = RuntimeError("Capture operation failed.")
    session.failures[operation] = failure
    if cleanup_fails:
        session.failures["close"] = RuntimeError("Capture cleanup failed.")
        if operation == "start":
            session.failures["stop"] = RuntimeError("Capture stop failed.")

    if operation != "prepare":
        session.prepare()
    if operation == "stop":
        session.start()

    action: Callable[[], None] = {
        "prepare": session.prepare,
        "start": session.start,
        "stop": session.stop,
    }[operation]

    with pytest.raises(RuntimeError, match="Capture operation failed") as caught:
        action()

    assert caught.value is failure
    assert session.state is CaptureState.CLOSED
    assert not digitizer.is_active
    assert session.events.count("close") == 1
    if cleanup_fails:
        assert any("Capture cleanup failed" in note for note in failure.__notes__)
        if operation == "start":
            assert any("Capture stop failed" in note for note in failure.__notes__)

    session.close()
    assert session.events.count("close") == 1
    digitizer.configure()
    digitizer.close()


@pytest.mark.parametrize("stop_fails", [False, True])
def test_close_failure_still_releases_the_digitizer(stop_fails: bool) -> None:
    """Attempt native cleanup after stop failure and preserve the first error."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    cleanup = RuntimeError("Cleanup failed.")
    stopping = RuntimeError("Stop failed.")
    session.failures["close"] = cleanup
    if stop_fails:
        session.failures["stop"] = stopping

    session.prepare()
    session.start()

    with pytest.raises(RuntimeError) as caught:
        session.close()

    assert caught.value is (stopping if stop_fails else cleanup)
    assert session.events == ["prepare", "start", "stop", "close"]
    assert session.state is CaptureState.CLOSED
    if stop_fails:
        assert any("Cleanup failed" in note for note in stopping.__notes__)

    session.close()
    digitizer.configure()
    digitizer.close()


def test_invalid_prepared_layout_rolls_back_the_session() -> None:
    """Reject a broken driver return value and clean up its prepared resources."""
    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)
    session.prepared_layout = cast(CaptureLayout, object())

    with pytest.raises(TypeError, match="must return a CaptureLayout"):
        session.prepare()

    assert session.events == ["prepare", "close"]
    assert session.state is CaptureState.CLOSED
    with pytest.raises(RuntimeError, match="layout is unavailable"):
        _ = session.layout

    digitizer.configure()
    digitizer.close()


@pytest.mark.parametrize("close_fails", [False, True])
def test_device_close_is_permanent_and_idempotent(close_fails: bool) -> None:
    """End device usability once, including when persistent cleanup fails."""
    digitizer = RecordingDigitizer()
    if close_fails:
        digitizer.close_failure = RuntimeError("Device cleanup failed.")
        with pytest.raises(RuntimeError, match="Device cleanup failed"):
            digitizer.close()
    else:
        digitizer.close()
    digitizer.close()

    assert digitizer.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        digitizer.configure()

    session = RecordingSession(digitizer)
    with pytest.raises(RuntimeError, match="closed"):
        session.prepare()

    assert session.state is CaptureState.CLOSED
    assert session.events == []


def test_start_requires_preparation_and_constructor_requires_digitizer() -> None:
    """Reject incompatible devices and starting outside the prepared state."""
    with pytest.raises(TypeError, match="digitizer must be a Digitizer"):
        RecordingSession(cast(RecordingDigitizer, object()))

    digitizer = RecordingDigitizer()
    session = RecordingSession(digitizer)

    with pytest.raises(RuntimeError, match="must be prepared"):
        session.start()
    session.close()
    with pytest.raises(RuntimeError, match="must be prepared"):
        session.start()

    digitizer.close()
