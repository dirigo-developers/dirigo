"""Test how workers distinguish requested cancellation from failures.

OperationCancelled raised during work is treated as normal completion only
when cancellation has been requested. Other work failures remain reportable,
including failures that occur after an abort request.

Cleanup failures must always be reported, even when cleanup follows expected
cancellation. Channel cancellation and graph-wide coordination are tested
separately.
"""

from threading import Condition, Event

import pytest

from dirigo.pipeline import OperationCancelled, Pipeline, Product
from dirigo.pipeline.publisher import Publisher

from .stubs import StubWorker


def _observe_wait(
    condition: Condition,
    monkeypatch: pytest.MonkeyPatch,
) -> Event:
    """Signal when an operation reaches the condition's wait method."""
    waiting = Event()
    original_wait = condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", observed_wait)
    return waiting


@pytest.mark.parametrize("operation", ["receive", "publish"])
def test_worker_abort_interrupts_channel_operation(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Requested cancellation interrupts blocked work without reporting failure.

    Block the worker on either an empty subscription or a full publication
    queue. The test worker connects its abort hook to this additional channel,
    recording the abort request before waking the blocked operation.

    OperationCancelled must then be treated as expected completion. Cleanup
    must run exactly once, and the worker must finish before the channel
    operation's timeout expires.
    """
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=1)

    if operation == "publish":
        publisher.publish(Product(False), timeout=0.0)

    class ChannelWorker(StubWorker):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def _work(self) -> None:
            if operation == "receive":
                with subscription.receive(timeout=5.0):
                    pass
            else:
                publisher.publish(Product(True), timeout=5.0)

        def _request_abort(self) -> None:
            # Record cancellation before waking the external channel.
            super()._request_abort()

            if operation == "receive":
                subscription.abort()
            else:
                publisher.abort()

        def _close(self) -> None:
            self.close_calls += 1

            if operation == "receive":
                subscription.close()
            else:
                publisher.close()

    worker = ChannelWorker()
    pipeline = Pipeline(worker)
    waiting = _observe_wait(publisher._condition, monkeypatch)

    pipeline.start()

    try:
        assert waiting.wait(timeout=2.0)

        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert worker._exception is None
        assert worker.close_calls == 1
    finally:
        pipeline.abort()
        publisher.abort()
        worker._wait(timeout=6.0)


def test_unrequested_cancellation_is_a_worker_failure() -> None:
    """OperationCancelled is a failure when no abort request preceded it.

    The exception type alone must not cause suppression. The original exception
    must be recorded and raised by Pipeline.wait(), while cleanup still runs
    after work fails.
    """
    failure = OperationCancelled("unexpected cancellation")
    calls: list[str] = []

    class UnexpectedCancellationWorker(StubWorker):
        def _work(self) -> None:
            calls.append("work")
            raise failure

        def _close(self) -> None:
            calls.append("close")

    worker = UnexpectedCancellationWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        with pytest.raises(OperationCancelled) as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert worker._exception is failure
        assert pipeline.done
        assert calls == ["work", "close"]
    finally:
        pipeline.abort()
        worker._wait(timeout=2.0)


def test_work_failure_after_abort_is_not_suppressed() -> None:
    """An abort request does not hide an ordinary exception raised during work.

    Wait for the abort request before raising RuntimeError. This ensures the
    failure occurs during cancellation rather than before it.

    Only expected OperationCancelled exceptions may be suppressed; this error
    must remain available through both the worker and the pipeline.
    """
    failure = RuntimeError("work failed during cancellation")

    class FailingWorker(StubWorker):
        def _work(self) -> None:
            self._abort_requested.wait()
            raise failure

    worker = FailingWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        pipeline.abort()

        with pytest.raises(RuntimeError) as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert worker._exception is failure
        assert pipeline.done
    finally:
        pipeline.abort()
        worker._wait(timeout=2.0)


@pytest.mark.parametrize(
    "cleanup_failure",
    [
        OSError("cleanup failed"),
        OperationCancelled("cleanup interrupted"),
    ],
)
def test_cleanup_failure_after_cancellation_is_reported(
    cleanup_failure: BaseException,
) -> None:
    """Cleanup failures remain reportable after expected work cancellation.

    Work waits for an abort request, then raises OperationCancelled normally.
    Its cleanup hook subsequently raises the parameterized exception.

    That cleanup exception must become the reported failure, including when it
    is itself OperationCancelled. Cancellation suppression applies to work,
    not to errors encountered while releasing resources.
    """
    class CleanupFailureWorker(StubWorker):
        def _work(self) -> None:
            self._abort_requested.wait()
            raise OperationCancelled("worker cancelled")

        def _close(self) -> None:
            raise cleanup_failure

    worker = CleanupFailureWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        pipeline.abort()

        with pytest.raises(type(cleanup_failure)) as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is cleanup_failure
        assert pipeline.exception is cleanup_failure
        assert worker._exception is cleanup_failure
        assert worker._close_exception is cleanup_failure
        assert pipeline.done
    finally:
        pipeline.abort()
        worker._wait(timeout=2.0)
