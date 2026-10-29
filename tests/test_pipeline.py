from threading import Thread, get_ident

import pytest

from dirigo.pipeline import Pipeline

from .stubs import (
    AbortableWorker,
    BlockingWorker,
    PreparationWorker,
    SlowClosingWorker,
    StubWorker,
    TriggeredFailureWorker,
)


def test_new_pipeline_has_no_workers() -> None:
    pipeline = Pipeline()

    assert pipeline.workers == ()


def test_add_rejects_non_worker() -> None:
    pipeline = Pipeline()

    with pytest.raises(expected_exception=TypeError, match="worker"):
        pipeline.add(worker=object())  # ty: ignore[invalid-argument-type]

    assert pipeline.workers == ()


def test_add_includes_dependencies_upstream_first() -> None:
    source = StubWorker()
    processor = StubWorker(upstream=source)
    consumer = StubWorker(upstream=processor)
    pipeline = Pipeline()

    pipeline.add(worker=consumer)

    assert pipeline.workers == (source, processor, consumer)


def test_add_includes_shared_dependencies_only_once() -> None:
    source = StubWorker()
    a1 = StubWorker(upstream=source)
    a2 = StubWorker(upstream=a1)
    b1 = StubWorker(upstream=source)
    b2 = StubWorker(upstream=b1)
    pipeline = Pipeline(a2, b2)

    assert pipeline.workers == (source, a1, a2, b1, b2)


def test_add_raises_value_error_for_cycle() -> None:
    first = StubWorker()
    second = StubWorker(upstream=first)

    first._upstream = second # deliberately create an invalid graph

    pipeline = Pipeline()

    with pytest.raises(expected_exception=ValueError, match="cycle"):
        pipeline.add(worker=second)

    assert pipeline.workers == ()


def test_start_rejects_empty_pipeline() -> None:
    pipeline = Pipeline()

    with pytest.raises(expected_exception=ValueError, match="empty"):
        pipeline.start()

    assert not pipeline.done

    # Rejection must not consume the pipeline's single execution.
    worker = StubWorker()
    pipeline.add(worker)
    pipeline.start()
    pipeline.wait(timeout=2.0)

    assert pipeline.done


def test_pipeline_runs_worker_to_completion() -> None:
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

    assert [name for name, _ in calls] == ["prepare", "work", "close"] # hook order
    assert calls[0][1] == get_ident()  # prepare runs in the caller.
    assert calls[1][1] != get_ident()  # work runs in another thread.
    assert calls[2][1] == calls[1][1]  # close runs in that same thread.
    assert pipeline.done # completion


# test 3 scenarios: work fails, cleanup fails, both fail
@pytest.mark.parametrize(
    argnames=("work_failure", "close_failure"),
    argvalues=[
        (RuntimeError("work failed"), None),
        (None, OSError("cleanup failed")),
        (RuntimeError("work failed"), OSError("cleanup failed")),
    ],
)
def test_pipeline_reports_worker_failure(
    work_failure: BaseException | None,
    close_failure: BaseException | None,
) -> None:
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
    expected: BaseException | None = (
        work_failure if work_failure is not None else close_failure
    )

    assert expected is not None
    initial_exception: BaseException | None = pipeline.exception
    assert initial_exception is None

    pipeline.start()

    with pytest.raises(expected_exception=type(expected)) as caught:
        pipeline.wait(timeout=2.0)

    assert caught.value is expected
    assert pipeline.exception is expected
    assert calls == ["work", "close"]
    assert pipeline.done

    if work_failure is not None and close_failure is not None:
        assert worker._close_exception is close_failure
        assert any(
            "cleanup failed" in note
            for note in work_failure.__notes__
        )


@pytest.mark.parametrize("finish_first", [False, True])
def test_pipeline_cannot_start_twice(finish_first: bool) -> None:
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
    worker = BlockingWorker()
    pipeline = Pipeline(worker)
    pipeline.start()

    try:
        if finish_first:
            worker.release.set()
            pipeline.wait(timeout=2.0)

        with pytest.raises(
            expected_exception=RuntimeError,
            match="after execution has started"
        ):
            pipeline.add(worker=StubWorker())

        assert pipeline.workers == (worker,)
    finally:
        worker.release.set()
        pipeline.wait(timeout=2.0)


def test_pipeline_cannot_wait_before_start() -> None:
    pipeline = Pipeline(StubWorker())

    with pytest.raises(expected_exception=RuntimeError, match="has not started"):
        pipeline.wait()


def test_pipeline_wait_timeout_does_not_stop_execution() -> None:
    worker = BlockingWorker()
    pipeline = Pipeline(worker)
    pipeline.start()

    try:
        with pytest.raises(expected_exception=TimeoutError):
            pipeline.wait(timeout=0.0)

        finished_at_timeout: bool = pipeline.done
        assert not finished_at_timeout
    finally:
        worker.release.set()
        pipeline.wait(timeout=2.0)

    assert pipeline.done
    assert pipeline.exception is None


def test_successful_pipeline_has_no_exception() -> None:
    pipeline = Pipeline(StubWorker())

    pipeline.start()
    pipeline.wait(timeout=2.0)

    assert pipeline.done
    assert pipeline.exception is None


def test_preparation_failure_cleans_up_attempted_workers() -> None:
    calls: list[str] = []
    failure = RuntimeError("prep failed")

    first = PreparationWorker(
        name="first",
        calls=calls,
    )
    second = PreparationWorker(
        name="second",
        calls=calls,
        upstream=first,
        prepare_failure=failure, # will result in failure at prepare stage
    )
    third = PreparationWorker(
        name="third",
        calls=calls,
        upstream=second,
    )
    pipeline = Pipeline(third)

    with pytest.raises(expected_exception=RuntimeError, match="prep failed") as caught:
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

    # Waiting reports the failure without joining unstarted threads.
    with pytest.raises(expected_exception=RuntimeError) as caught_again:
        pipeline.wait(timeout=0.0)

    assert caught_again.value is failure

    # A failed execution still consumes this single-use pipeline.
    with pytest.raises(expected_exception=RuntimeError, match="already started"):
        pipeline.start()

    with pytest.raises(expected_exception=RuntimeError, match="execution has started"):
        pipeline.add(worker=StubWorker())


@pytest.mark.parametrize("close_failure_at", ["first", "second"])
def test_cleanup_failure_preserves_preparation_failure(
    close_failure_at: str,
) -> None:
    calls: list[str] = []
    prepare_failure = RuntimeError("preparation failed")
    close_failure = OSError("cleanup failed")

    first = PreparationWorker(
        name="first",
        calls=calls,
    )
    second = PreparationWorker(
        name="second",
        calls=calls,
        upstream=first,
        prepare_failure=prepare_failure,
    )

    failing_closer: PreparationWorker = first if close_failure_at == "first" else second
    failing_closer.close_failure = close_failure
    pipeline = Pipeline(second)

    with pytest.raises(expected_exception=RuntimeError) as caught:
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
    assert any(
        "cleanup failed" in note
        for note in prepare_failure.__notes__
    )


@pytest.mark.parametrize(argnames="worker_count", argvalues=[1, 4])
def test_abort_finishes_all_workers(worker_count: int) -> None:
    workers: list[AbortableWorker] = [AbortableWorker() for _ in range(worker_count)]
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

        # Another request after completion must not repeat cleanup.
        pipeline.abort()
        assert all(worker.close_calls == 1 for worker in workers)
    finally:
        # Release workers even if abort() is broken.
        for worker in workers:
            worker._abort_requested.set()
        pipeline.wait(timeout=2.0)


def test_abort_returns_without_waiting_for_cleanup() -> None:
    worker = SlowClosingWorker()
    pipeline = Pipeline(worker)
    pipeline.start()

    try:
        assert worker.entered.wait(timeout=2.0)

        pipeline.abort()

        assert worker.closing.wait(timeout=2.0)
        finished_while_closing: bool = pipeline.done
        assert not finished_while_closing

        with pytest.raises(expected_exception=TimeoutError):
            pipeline.wait(timeout=0.0)

    finally:
        worker._abort_requested.set()
        worker.release_close.set()
        pipeline.wait(timeout=2.0)

    assert pipeline.done
    assert worker.close_calls == 1
    assert pipeline.exception is None


def test_abort_before_start_raises() -> None:
    pipeline = Pipeline(StubWorker())

    with pytest.raises(expected_exception=RuntimeError, match="has not started"):
        pipeline.abort()


def test_abort_after_normal_completion_is_harmless() -> None:
    pipeline = Pipeline(StubWorker())
    pipeline.start()
    pipeline.wait(timeout=2.0)

    pipeline.abort()

    assert pipeline.done
    assert pipeline.exception is None


@pytest.mark.parametrize(argnames="fail_on_close", argvalues=[False, True])
def test_worker_failure_cancels_other_workers(fail_on_close: bool) -> None:
    failure = RuntimeError("worker failed")
    failing = TriggeredFailureWorker(
        failure,
        fail_on_close=fail_on_close,
    )
    peers: list[AbortableWorker] = [AbortableWorker(), AbortableWorker()]
    workers: tuple[TriggeredFailureWorker, *tuple[AbortableWorker, ...]] = (
        failing,
        *peers
    )
    pipeline = Pipeline(*workers)
    pipeline.start()

    try:
        assert failing.entered.wait(timeout=2.0)
        for peer in peers:
            assert peer.entered.wait(timeout=2.0)

        failing.fail_now.set()

        with pytest.raises(expected_exception=RuntimeError, match="worker failed") as c:
            pipeline.wait(timeout=2.0)

        assert c.value is failure
        assert pipeline.exception is failure
        assert pipeline.done
        assert failing.close_calls == 1
        assert all(peer.close_calls == 1 for peer in peers)
    finally:
        # Release and join workers even if automatic cancellation fails.
        failing.fail_now.set()
        for worker in workers:
            worker._abort_requested.set()
        for worker in workers:
            worker._wait(timeout=2.0)


def test_failure_during_cancellation_preserves_first_failure() -> None:
    first_failure = RuntimeError("first failure")
    second_failure = ValueError("failure during cancellation")

    class FailingOnAbortWorker(AbortableWorker):
        def _work(self) -> None:
            super()._work()
            raise second_failure

    first = TriggeredFailureWorker(first_failure)
    second = FailingOnAbortWorker()

    pipeline = Pipeline(second, first)
    pipeline.start()

    try:
        assert first.entered.wait(timeout=2.0)
        assert second.entered.wait(timeout=2.0)

        first.fail_now.set()

        with pytest.raises(expected_exception=RuntimeError, match="first failure") as c:
            pipeline.wait(timeout=2.0)

        assert c.value is first_failure
        assert pipeline.exception is first_failure
        assert second._exception is second_failure
        assert pipeline.done
        assert first.close_calls == 1
        assert second.close_calls == 1
    finally:
        first.fail_now.set()
        for worker in pipeline.workers:
            worker._abort_requested.set()
        for worker in pipeline.workers:
            worker._wait(timeout=2.0)


@pytest.mark.parametrize(argnames="fail_at", argvalues=["running", "middle", "source"])
@pytest.mark.parametrize(argnames="cleanup_fails", argvalues=[False, True])
def test_thread_launch_failure_cleans_up_graph(
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
    cleanup_fails: bool,
) -> None:
    prepared: list[str] = []
    closed: list[tuple[str, int]] = []
    launch_attempts: list[str] = []

    launch_failure = RuntimeError("thread launch failed")
    cleanup_errors: dict[str, OSError] = {
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

        # Ensure successful launches are running before attempting the next.
        assert by_name[thread.name].entered.wait(timeout=2.0)

    monkeypatch.setattr(target=Thread, name="start", value=controlled_start)

    try:
        with pytest.raises(expected_exception=RuntimeError, match="thread launch failed") as caught:
            pipeline.start()

        assert caught.value is launch_failure

        with pytest.raises(expected_exception=RuntimeError) as waited:
            pipeline.wait(timeout=2.0)

        assert waited.value is launch_failure
        assert pipeline.exception is launch_failure
        assert pipeline.done

        assert prepared == ["source", "middle", "running"]
        assert all(worker.close_calls == 1 for worker in workers)
        assert all(worker._abort_requested.is_set() for worker in workers)

        launch_order: list[str] = ["running", "middle", "source"]
        failure_index: int = launch_order.index(fail_at)

        assert launch_attempts == launch_order[:failure_index + 1]

        # Only unlaunched workers are closed by the caller, in reverse order.
        manual_closes: list[str] = [
            name for name, thread_id in closed
            if thread_id == get_ident()
        ]
        assert manual_closes == launch_order[failure_index:]

        if cleanup_fails:
            for worker in workers:
                assert worker._close_exception is cleanup_errors[worker.name]

            for name in manual_closes:
                assert any(
                    str(object=cleanup_errors[name]) in note
                    for note in launch_failure.__notes__
                )
    finally:
        # Release actual threads even if startup handling is broken.
        for worker in workers:
            worker._abort_requested.set()

        for worker in workers:
            thread: Thread | None = worker._thread
            if thread is not None and thread.ident is not None:
                worker._wait(timeout=2.0)
