"""Test pipeline startup, waiting, completion, and explicit cancellation.

Verify lifecycle-hook order and execution threads, single-use execution,
restrictions on adding workers, and timeout behavior.

Explicit abort must finish workers without reporting failure, remain harmless
when repeated, and return without waiting for cleanup to complete.

Graph validation and failure handling are tested separately.
"""

from threading import Event, get_ident

import pytest

from dirigo.pipeline import Pipeline

from .stubs import AbortableWorker, BlockingWorker, StubWorker


class SlowClosingWorker(AbortableWorker):
    """Pause cleanup so the test can inspect a pipeline while closing."""

    def __init__(self) -> None:
        super().__init__()
        self.closing = Event()
        self.release_close = Event()

    def _close(self) -> None:
        self.closing.set()

        # Bound the wait so an incorrectly blocking abort cannot hang the test.
        self.release_close.wait(timeout=5.0)

        super()._close()


def test_pipeline_runs_worker_to_completion() -> None:
    """Preparation precedes threaded work and cleanup on normal execution.

    Preparation must run on the thread calling Pipeline.start(). Work must run
    on another thread, and cleanup must run on that same worker thread.
    """
    calls: list[tuple[str, int]] = []

    class RecordingWorker(StubWorker):
        def _prepare(self) -> None:
            calls.append(("prepare", get_ident()))

        def _work(self) -> None:
            calls.append(("work", get_ident()))

        def _close(self) -> None:
            calls.append(("close", get_ident()))

    pipeline = Pipeline(RecordingWorker())

    assert not pipeline.done

    pipeline.start()
    pipeline.wait(timeout=2.0)

    assert [name for name, _ in calls] == ["prepare", "work", "close"]
    assert calls[0][1] == get_ident()
    assert calls[1][1] != get_ident()
    assert calls[2][1] == calls[1][1]
    assert pipeline.done


@pytest.mark.parametrize("finish_first", [False, True])
def test_pipeline_cannot_start_twice(finish_first: bool) -> None:
    """A pipeline cannot restart while running or after normal completion.

    Keep the worker blocked to test restart during execution. In the other
    scenario, release it and wait for completion before attempting restart.
    """
    worker = BlockingWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        if finish_first:
            worker.release.set()
            pipeline.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="already started"):
            pipeline.start()
    finally:
        worker.release.set()
        pipeline.wait(timeout=2.0)


@pytest.mark.parametrize("finish_first", [False, True])
def test_cannot_add_workers_after_start(finish_first: bool) -> None:
    """Starting execution fixes membership, including after execution finishes.

    Attempt to add an independent worker while the pipeline is running or after
    completion. Rejection must preserve the original worker collection.
    """
    worker = BlockingWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        if finish_first:
            worker.release.set()
            pipeline.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="after execution has started"):
            pipeline.add(StubWorker())

        assert pipeline.workers == (worker,)
    finally:
        worker.release.set()
        pipeline.wait(timeout=2.0)


def test_pipeline_cannot_wait_before_start() -> None:
    """Waiting before execution starts raises RuntimeError."""
    pipeline = Pipeline(StubWorker())

    with pytest.raises(RuntimeError, match="has not started"):
        pipeline.wait()


def test_pipeline_wait_timeout_does_not_stop_execution() -> None:
    """A wait timeout leaves execution active without requesting cancellation.

    The worker remains blocked until its release event is set. A zero-timeout
    wait must fail while leaving the pipeline unfinished and the worker's abort
    flag unset. Releasing the worker must then allow normal completion.
    """
    worker = BlockingWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        with pytest.raises(TimeoutError):
            pipeline.wait(timeout=0.0)

        pipeline_done = pipeline.done
        assert not pipeline_done
        assert not worker._abort_requested.is_set()
    finally:
        worker.release.set()
        pipeline.wait(timeout=2.0)

    assert pipeline.done
    assert pipeline.exception is None


def test_successful_pipeline_has_no_exception() -> None:
    """Normal completion leaves the pipeline finished with no recorded failure."""
    pipeline = Pipeline(StubWorker())

    pipeline.start()
    pipeline.wait(timeout=2.0)

    assert pipeline.done
    assert pipeline.exception is None


@pytest.mark.parametrize("worker_count", [1, 4])
def test_abort_finishes_all_workers(worker_count: int) -> None:
    """Repeated abort requests finish every worker and run cleanup exactly once.

    Wait until all workers enter their work hooks before requesting cancellation.
    Repeated requests during execution and after completion must not report a
    failure or repeat cleanup.
    """
    workers = [AbortableWorker() for _ in range(worker_count)]
    pipeline = Pipeline(*workers)

    pipeline.start()

    try:
        for worker in workers:
            assert worker.entered.wait(timeout=2.0)

        pipeline.abort()
        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert all(worker.close_calls == 1 for worker in workers)

        pipeline.abort()

        assert all(worker.close_calls == 1 for worker in workers)
    finally:
        # Release workers directly even if pipeline cancellation is broken.
        for worker in workers:
            worker._abort_requested.set()

        pipeline.wait(timeout=2.0)


def test_abort_returns_without_waiting_for_cleanup() -> None:
    """Abort requests cancellation without waiting for worker cleanup to finish.

    The worker pauses inside its cleanup hook until explicitly released.
    After abort returns, cleanup must be observable as active while the pipeline
    remains unfinished. A zero-timeout wait must still raise TimeoutError.

    Releasing cleanup must then permit normal completion with one cleanup call.
    """
    worker = SlowClosingWorker()
    pipeline = Pipeline(worker)

    pipeline.start()

    try:
        assert worker.entered.wait(timeout=2.0)

        pipeline.abort()

        assert worker.closing.wait(timeout=2.0)
        pipeline_done = pipeline.done
        assert not pipeline_done

        with pytest.raises(TimeoutError):
            pipeline.wait(timeout=0.0)
    finally:
        worker._abort_requested.set()
        worker.release_close.set()
        pipeline.wait(timeout=2.0)

    assert pipeline.done
    assert worker.close_calls == 1
    assert pipeline.exception is None


def test_abort_before_start_raises() -> None:
    """Cancellation before execution starts raises RuntimeError."""
    pipeline = Pipeline(StubWorker())

    with pytest.raises(RuntimeError, match="has not started"):
        pipeline.abort()


def test_abort_after_normal_completion_is_harmless() -> None:
    """Aborting an already completed pipeline preserves its successful outcome."""
    pipeline = Pipeline(StubWorker())

    pipeline.start()
    pipeline.wait(timeout=2.0)

    pipeline.abort()

    assert pipeline.done
    assert pipeline.exception is None
