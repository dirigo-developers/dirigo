"""Test pipeline failure reporting, cancellation, and resource cleanup.

Verify that work, preparation, cleanup, and thread-launch failures preserve the
appropriate original exception. Worker failure must cancel peers, and later
failures must not replace the first reported failure.

Preparation rollback must close attempted workers in reverse order. Failed
thread launch must clean up both running and unlaunched workers exactly once.
"""

from threading import Event, Thread, get_ident

import pytest

from dirigo.pipeline import Pipeline, Worker

from .stubs import AbortableWorker, StubWorker


class PreparationWorker(StubWorker):
    """Record lifecycle calls and optionally fail preparation or cleanup."""

    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        upstream: Worker | None = None,
        prepare_failure: BaseException | None = None,
        close_failure: BaseException | None = None,
    ) -> None:
        super().__init__(name=name, upstream=upstream)
        self.calls = calls
        self.prepare_failure = prepare_failure
        self.close_failure = close_failure

    def _prepare(self) -> None:
        self.calls.append(f"{self.name}.prepare")

        if self.prepare_failure is not None:
            raise self.prepare_failure

    def _work(self) -> None:
        self.calls.append(f"{self.name}.work")

    def _close(self) -> None:
        self.calls.append(f"{self.name}.close")

        if self.close_failure is not None:
            raise self.close_failure


class TriggeredFailureWorker(StubWorker):
    """Raise a chosen failure during work or cleanup when the test permits."""

    def __init__(
        self,
        failure: BaseException,
        *,
        fail_on_close: bool = False,
    ) -> None:
        super().__init__()
        self.failure = failure
        self.fail_on_close = fail_on_close
        self.entered = Event()
        self.fail_now = Event()
        self.close_calls = 0

    def _work(self) -> None:
        self.entered.set()
        self.fail_now.wait()

        if not self.fail_on_close:
            raise self.failure

    def _close(self) -> None:
        self.close_calls += 1

        if self.fail_on_close:
            raise self.failure


def _release_and_join(workers: tuple[Worker, ...]) -> None:
    """Release cancellation waits and join only threads that actually started.

    Set cancellation events directly so cleanup still works when pipeline
    cancellation is the behavior under test. A failed Thread.start() can leave
    a thread object without a started thread, which must not be joined.
    """
    for worker in workers:
        worker._abort_requested.set()

    for worker in workers:
        thread = worker._thread

        if thread is not None and thread.ident is not None:
            worker._wait(timeout=2.0)


@pytest.mark.parametrize(
    ("work_failure", "close_failure"),
    [
        (RuntimeError("work failed"), None),
        (None, OSError("cleanup failed")),
        (RuntimeError("work failed"), OSError("cleanup failed")),
    ],
)
def test_pipeline_reports_worker_failure(
    work_failure: BaseException | None,
    close_failure: BaseException | None,
) -> None:
    """Report a worker's work failure, cleanup failure, or both appropriately.

    Cleanup must run after work even when work raises. When both hooks fail,
    the work exception remains primary, while the cleanup exception is retained
    on the worker and described in a note on the primary exception.
    """
    calls: list[str] = []

    class FailingWorker(StubWorker):
        def _work(self) -> None:
            calls.append("work")

            if work_failure is not None:
                raise work_failure

        def _close(self) -> None:
            calls.append("close")

            if close_failure is not None:
                raise close_failure

    worker = FailingWorker()
    pipeline = Pipeline(worker)
    expected = work_failure if work_failure is not None else close_failure

    assert expected is not None
    pipeline_exception = pipeline.exception
    assert pipeline_exception is None

    try:
        pipeline.start()

        with pytest.raises(type(expected)) as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is expected
        assert pipeline.exception is expected
        assert calls == ["work", "close"]
        assert pipeline.done

        if work_failure is not None and close_failure is not None:
            assert worker._close_exception is close_failure
            assert any("cleanup failed" in note for note in work_failure.__notes__)
    finally:
        _release_and_join(pipeline.workers)


def test_preparation_failure_cleans_up_attempted_workers() -> None:
    """Failed preparation closes attempted workers in reverse preparation order.

    The second worker fails while preparing. Both it and the first worker must
    be closed, but the third worker must remain untouched. No execution threads
    should have started.

    Waiting must report the same failure, and the failed execution must consume
    the pipeline's single permitted start.
    """
    calls: list[str] = []
    failure = RuntimeError("prep failed")

    first = PreparationWorker(name="first", calls=calls)
    second = PreparationWorker(
        name="second",
        calls=calls,
        upstream=first,
        prepare_failure=failure,
    )
    third = PreparationWorker(
        name="third",
        calls=calls,
        upstream=second,
    )
    pipeline = Pipeline(third)

    try:
        with pytest.raises(RuntimeError, match="prep failed") as caught:
            pipeline.start()

        assert caught.value is failure
        assert pipeline.exception is failure
        assert pipeline.done
        assert calls == [
            "first.prepare",
            "second.prepare",
            "second.close",
            "first.close",
        ]
        assert all(worker._thread is None for worker in pipeline.workers)

        with pytest.raises(RuntimeError) as caught_again:
            pipeline.wait(timeout=0.0)

        assert caught_again.value is failure

        with pytest.raises(RuntimeError, match="already started"):
            pipeline.start()

        with pytest.raises(RuntimeError, match="execution has started"):
            pipeline.add(StubWorker())
    finally:
        _release_and_join(pipeline.workers)


@pytest.mark.parametrize("close_failure_at", ["first", "second"])
def test_cleanup_failure_preserves_preparation_failure(
    close_failure_at: str,
) -> None:
    """A rollback cleanup failure must not replace the preparation exception.

    Fail preparation in the second worker, then fail cleanup in either worker.
    Both cleanup hooks must still be attempted in reverse order. Record the
    cleanup error separately and attach a note to the preparation exception.
    """
    calls: list[str] = []
    prepare_failure = RuntimeError("preparation failed")
    close_failure = OSError("cleanup failed")

    first = PreparationWorker(name="first", calls=calls)
    second = PreparationWorker(
        name="second",
        calls=calls,
        upstream=first,
        prepare_failure=prepare_failure,
    )

    failing_closer = first if close_failure_at == "first" else second
    failing_closer.close_failure = close_failure
    pipeline = Pipeline(second)

    try:
        with pytest.raises(RuntimeError) as caught:
            pipeline.start()

        assert caught.value is prepare_failure
        assert pipeline.exception is prepare_failure
        assert pipeline.done
        assert calls == [
            "first.prepare",
            "second.prepare",
            "second.close",
            "first.close",
        ]
        assert failing_closer._close_exception is close_failure
        assert any("cleanup failed" in note for note in prepare_failure.__notes__)
    finally:
        _release_and_join(pipeline.workers)


@pytest.mark.parametrize("fail_on_close", [False, True])
def test_worker_failure_cancels_other_workers(fail_on_close: bool) -> None:
    """A work or cleanup failure automatically cancels all running peers.

    Wait until the failing worker and both peers have entered their work hooks.
    Trigger the failure without explicitly aborting the pipeline.

    Waiting must raise the original failure, and every worker must finish with
    exactly one cleanup call.
    """
    failure = RuntimeError("worker failed")
    failing = TriggeredFailureWorker(
        failure,
        fail_on_close=fail_on_close,
    )
    peers = [AbortableWorker(), AbortableWorker()]
    workers = (failing, *peers)
    pipeline = Pipeline(*workers)

    try:
        pipeline.start()

        assert failing.entered.wait(timeout=2.0)

        for peer in peers:
            assert peer.entered.wait(timeout=2.0)

        failing.fail_now.set()

        with pytest.raises(RuntimeError, match="worker failed") as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert pipeline.done
        assert failing.close_calls == 1
        assert all(peer.close_calls == 1 for peer in peers)
    finally:
        failing.fail_now.set()
        _release_and_join(workers)


def test_failure_during_cancellation_preserves_first_failure() -> None:
    """A peer's later failure during cancellation does not replace the first error.

    Trigger the first worker's failure. The second worker waits for the resulting
    cancellation request before raising a different exception, establishing the
    order of failures without relying on scheduling.

    The pipeline must report the first exception while the second worker retains
    its own failure for inspection.
    """
    first_failure = RuntimeError("first failure")
    second_failure = ValueError("failure during cancellation")

    class FailingOnAbortWorker(AbortableWorker):
        def _work(self) -> None:
            super()._work()
            raise second_failure

    first = TriggeredFailureWorker(first_failure)
    second = FailingOnAbortWorker()
    pipeline = Pipeline(second, first)

    try:
        pipeline.start()

        assert first.entered.wait(timeout=2.0)
        assert second.entered.wait(timeout=2.0)

        first.fail_now.set()

        with pytest.raises(RuntimeError, match="first failure") as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is first_failure
        assert pipeline.exception is first_failure
        assert second._exception is second_failure
        assert pipeline.done
        assert first.close_calls == 1
        assert second.close_calls == 1
    finally:
        first.fail_now.set()
        _release_and_join(pipeline.workers)


@pytest.mark.parametrize("fail_at", ["running", "middle", "source"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_thread_launch_failure_cleans_up_graph(
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
    cleanup_fails: bool,
) -> None:
    """A thread-launch failure cleans up the entire prepared graph exactly once.

    Preparation proceeds upstream first; launch proceeds downstream first.
    Inject a launch failure at each position, after ensuring any earlier
    launches have entered their work hooks.

    Running workers must be cancelled and clean up on their own threads.
    Unlaunched workers must be closed by the caller in reverse preparation
    order. Cleanup failures must not replace the original launch exception.
    """
    prepared: list[str] = []
    closed: list[tuple[str, int]] = []
    launch_attempts: list[str] = []

    launch_failure = RuntimeError("thread launch failed")
    cleanup_errors = {
        name: OSError(f"{name} cleanup failed")
        for name in ("source", "middle", "running")
    }

    class LaunchWorker(AbortableWorker):
        def _prepare(self) -> None:
            prepared.append(self.name)

        def _close(self) -> None:
            closed.append((self.name, get_ident()))
            super()._close()

            if cleanup_fails:
                raise cleanup_errors[self.name]

    source = LaunchWorker(name="source")
    middle = LaunchWorker(name="middle", upstream=source)
    running = LaunchWorker(name="running", upstream=middle)

    workers = (source, middle, running)
    by_name = {worker.name: worker for worker in workers}
    pipeline = Pipeline(running)
    original_start = Thread.start

    def controlled_start(thread: Thread) -> None:
        launch_attempts.append(thread.name)

        if thread.name == fail_at:
            raise launch_failure

        original_start(thread)

        # Establish that this worker is running before the next launch attempt.
        assert by_name[thread.name].entered.wait(timeout=2.0)

    monkeypatch.setattr(Thread, "start", controlled_start)

    try:
        with pytest.raises(RuntimeError, match="thread launch failed") as caught:
            pipeline.start()

        assert caught.value is launch_failure

        with pytest.raises(RuntimeError) as waited:
            pipeline.wait(timeout=2.0)

        assert waited.value is launch_failure
        assert pipeline.exception is launch_failure
        assert pipeline.done

        assert prepared == ["source", "middle", "running"]
        assert all(worker.close_calls == 1 for worker in workers)
        assert all(worker._abort_requested.is_set() for worker in workers)

        launch_order = ["running", "middle", "source"]
        failure_index = launch_order.index(fail_at)

        assert launch_attempts == launch_order[: failure_index + 1]

        manual_closes = [name for name, thread_id in closed if thread_id == get_ident()]
        assert manual_closes == launch_order[failure_index:]

        if cleanup_fails:
            for worker in workers:
                assert worker._close_exception is cleanup_errors[worker.name]

            for name in manual_closes:
                assert any(
                    str(cleanup_errors[name]) in note
                    for note in launch_failure.__notes__
                )
    finally:
        _release_and_join(workers)
