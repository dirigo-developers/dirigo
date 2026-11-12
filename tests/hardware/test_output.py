"""Test grouped finite output, axis ownership, and failure cleanup."""

from collections.abc import Callable, Mapping
from typing import cast

import pytest

from dirigo import units
from dirigo.hardware.output import GalvoOutputSession, OutputState
from dirigo.hardware.scanner import GalvoScanner, GalvoScannerConfig
from dirigo.hardware.simulated import (
    SimulatedGalvoOutputSession,
    SimulatedGalvoScanner,
)
from dirigo.hardware.waveform import AngularWaveform


def _axis() -> SimulatedGalvoScanner:
    """Construct an axis without accessing hardware."""
    return SimulatedGalvoScanner(
        GalvoScannerConfig(angle_range=units.AngleRange("-10 deg", "15 deg"))
    )


def _waveform(count: int = 2, rate: float = 1000.0) -> AngularWaveform:
    """Build a finite zero-angle waveform at the requested rate."""
    return AngularWaveform([0.0] * count, sample_rate=units.SampleRate(rate))


class RecordingSession(GalvoOutputSession):
    """Record output hooks and allow deterministic completion or failure."""

    def __init__(self, waveforms: Mapping[GalvoScanner, AngularWaveform]) -> None:
        """Validate bindings and initialize hook controls."""
        super().__init__(waveforms)
        self.events: list[str] = []
        self.failures: dict[str, RuntimeError] = {}
        self.complete = False

    def _step(self, name: str) -> None:
        """Record a hook and raise its configured error, if any."""
        self.events.append(name)
        if name in self.failures:
            raise self.failures[name]

    def _prepare_output(self) -> None:
        """Record preparation."""
        self._step("prepare")

    def _start_output(self) -> None:
        """Record starting."""
        self._step("start")

    def _wait_output(self, timeout: float | None) -> bool:
        """Return the test-controlled completion result."""
        self._step("wait")
        return self.complete

    def _stop_output(self) -> None:
        """Record stopping."""
        self._step("stop")

    def _close_output(self) -> None:
        """Record resource cleanup."""
        self._step("close")


@pytest.mark.parametrize(
    ("bindings", "error_type", "message"),
    [
        ([], TypeError, "mapping"),
        ({}, ValueError, "At least one"),
        ({"axis": _waveform()}, TypeError, "GalvoScanner"),
        ({_axis(): object()}, TypeError, "AngularWaveform"),
        (
            {_axis(): _waveform(), _axis(): _waveform(rate=2000.0)},
            ValueError,
            "same sample_rate",
        ),
        (
            {_axis(): _waveform(), _axis(): _waveform(count=3)},
            ValueError,
            "same sample count",
        ),
    ],
)
def test_invalid_groups_are_rejected(
    bindings: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Require a nonempty, typed group with common sample timing and length."""
    with pytest.raises(error_type, match=message):
        RecordingSession(cast(Mapping[GalvoScanner, AngularWaveform], bindings))


def test_group_lifecycle_reserves_axes_until_close() -> None:
    """Own a copied binding map and reserve both axes for the complete run.

    Construction leaves static commands available. Preparation blocks static
    writes and axis closure. A timed-out wait keeps the run active; completion
    still retains reservations until close. Cleanup releases both axes, and
    the same session cannot be prepared again.
    """
    x, y = _axis(), _axis()
    bindings: dict[GalvoScanner, AngularWaveform] = {x: _waveform(), y: _waveform()}
    session = RecordingSession(bindings)
    bindings.clear()
    x.angle = units.Angle("1 deg")
    assert session.state is OutputState.NEW
    assert session.events == []

    try:
        session.prepare()
        assert session.state is OutputState.PREPARED
        with pytest.raises(RuntimeError, match="reserved"):
            x.angle = units.Angle("2 deg")
        with pytest.raises(RuntimeError, match="reserved"):
            y.close()

        session.start()
        assert not session.wait(0.0)
        assert session.state is OutputState.STARTED
        session.complete = True
        assert session.wait(0.0)
        assert session.wait(0.0)
        assert session.state is OutputState.COMPLETED
        with pytest.raises(RuntimeError, match="reserved"):
            y.angle = units.Angle("0 deg")
    finally:
        session.close()

    session.close()
    session.stop()
    assert session.state is OutputState.CLOSED
    assert session.events == ["prepare", "start", "wait", "wait", "close"]
    assert x.angle == units.Angle("1 deg")
    assert y.angle is None
    x.angle = units.Angle("2 deg")
    y.angle = units.Angle("0 deg")
    with pytest.raises(RuntimeError, match="must be new"):
        session.prepare()


def test_partial_reservation_failure_preserves_other_session() -> None:
    """Release earlier reservations without stealing an axis from its owner.

    One session already owns y. A second reserves x before failing to reserve
    y. Its rollback must free x and leave the original reservation on y intact.
    """
    x, y = _axis(), _axis()
    owner = RecordingSession({y: _waveform()})
    attempted = RecordingSession({x: _waveform(), y: _waveform()})
    owner.prepare()
    try:
        with pytest.raises(RuntimeError, match="reserved"):
            attempted.prepare()
        assert attempted.state is OutputState.CLOSED
        x.angle = units.Angle("0 deg")
        with pytest.raises(RuntimeError, match="reserved"):
            y.angle = units.Angle("0 deg")
    finally:
        attempted.close()
        owner.close()
    y.angle = units.Angle("0 deg")


@pytest.mark.parametrize("operation", ["prepare", "start", "wait", "stop"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_operation_failure_closes_and_releases_axes(
    operation: str,
    cleanup_fails: bool,
) -> None:
    """Preserve an output error while closing the session and freeing its axis.

    Inject failure at preparation, start, wait, or stop. Cleanup must run in each
    case. Failures from stopping or cleanup are attached to the original
    exception rather than replacing it.
    """
    axis = _axis()
    session = RecordingSession({axis: _waveform()})
    failure = RuntimeError("Output failed.")
    session.failures[operation] = failure
    if cleanup_fails:
        session.failures["close"] = RuntimeError("Cleanup failed.")
        if operation in ("start", "wait"):
            session.failures["stop"] = RuntimeError("Stop failed.")

    if operation != "prepare":
        session.prepare()
    if operation in ("wait", "stop"):
        session.start()
    actions: dict[str, Callable[[], object]] = {
        "prepare": session.prepare,
        "start": session.start,
        "wait": session.wait,
        "stop": session.stop,
    }

    with pytest.raises(RuntimeError, match="Output failed") as caught:
        actions[operation]()

    assert caught.value is failure
    assert session.state is OutputState.CLOSED
    assert session.events.count("close") == 1
    if cleanup_fails:
        assert any("Cleanup failed" in note for note in failure.__notes__)
        if operation in ("start", "wait"):
            assert any("Stop failed" in note for note in failure.__notes__)
    axis.angle = units.Angle("0 deg")


def test_close_attempts_cleanup_after_stop_failure() -> None:
    """Attempt both stop and cleanup, preserving the first error.

    Both hooks fail during closure. The session must still become permanently
    closed, release its axis, and attach the cleanup error to the stop error.
    """
    axis = _axis()
    session = RecordingSession({axis: _waveform()})
    stopping = RuntimeError("Stop failed.")
    session.failures = {
        "stop": stopping,
        "close": RuntimeError("Cleanup failed."),
    }
    session.prepare()
    session.start()

    with pytest.raises(RuntimeError) as caught:
        session.close()

    assert caught.value is stopping
    assert session.events == ["prepare", "start", "stop", "close"]
    assert session.state is OutputState.CLOSED
    assert any("Cleanup failed" in note for note in stopping.__notes__)
    session.close()
    axis.angle = units.Angle("0 deg")


def test_simulation_supports_timeout_stop_and_finite_completion() -> None:
    """Exercise elapsed-time simulation without timing-sensitive assertions.

    A nominally long run is polled with zero timeout and then stopped.
    A second, very short run completes within a generous timeout. Neither
    waveform run changes the axis's static-command history.
    """
    axis = _axis()
    slow = SimulatedGalvoOutputSession({axis: _waveform(rate=0.002)})
    try:
        slow.prepare()
        slow.start()
        assert not slow.wait(0.0)
        assert slow.state is OutputState.STARTED
        slow.stop()
        slow.stop()
        assert slow.state is OutputState.STOPPED
        assert not slow.wait(0.0)
        with pytest.raises(RuntimeError, match="reserved"):
            axis.angle = units.Angle("0 deg")
    finally:
        slow.close()

    fast = SimulatedGalvoOutputSession({axis: _waveform(count=1, rate=1e9)})
    try:
        fast.prepare()
        fast.start()
        assert fast.wait(1.0)
        assert fast.state is OutputState.COMPLETED
    finally:
        fast.close()
    assert axis.angle is None


def test_unprepared_and_closed_sessions_cannot_start_or_wait() -> None:
    """Require preparation before starting and prohibit reuse after closure."""
    session = RecordingSession({_axis(): _waveform()})
    with pytest.raises(RuntimeError, match="must be prepared"):
        session.start()
    with pytest.raises(RuntimeError, match="must be started"):
        session.wait(0.0)
    session.close()
    with pytest.raises(RuntimeError, match="must be prepared"):
        session.start()
    with pytest.raises(RuntimeError, match="must be started"):
        session.wait(0.0)
