"""Test execution and cancellation of connected worker graphs.

Verify ordinary delivery, processing chains, fanout, and early consumer exit.
Finishing one branch must not obstruct the producer or remaining consumers.

Cancellation must wake workers blocked on connected channels. Consumer failure
must cancel its source and sibling branches while preserving the original
exception and running cleanup exactly once.
"""

from threading import Condition, Event, current_thread
from typing import Any, Never

import pytest

from dirigo.pipeline import EndOfStream, Pipeline, Worker

from .stubs import AbortableWorker, StubWorker


class BooleanSource(StubWorker[Never, bool]):
    """Publish a finite sequence and count successful publications."""

    def __init__(self, values: tuple[bool, ...]) -> None:
        super().__init__()
        self.values = values
        self.published = 0

    def _work(self) -> None:
        for value in self.values:
            self._publish(value, timeout=5.0)
            self.published += 1


class CollectingWorker(StubWorker[bool, Never]):
    """Collect incoming values until the upstream ends its stream."""

    def __init__(self, *, upstream: Worker[Any, bool]) -> None:
        super().__init__(upstream=upstream)
        self.values: list[bool] = []

    def _work(self) -> None:
        try:
            while True:
                with self._receive(timeout=5.0) as product:
                    self.values.append(product.payload)
        except EndOfStream:
            return


class FirstOnlyWorker(CollectingWorker):
    """Consume one value and finish without draining subsequent deliveries."""

    def _work(self) -> None:
        with self._receive(timeout=5.0) as product:
            self.values.append(product.payload)


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


def _abort_and_join(pipeline: Pipeline) -> None:
    """Release workers and channels, then join launched execution threads.

    Set cancellation flags directly so an assertion about abort ordering cannot
    leave workers waiting indefinitely during test cleanup.
    """
    if not pipeline._launched:
        return

    for worker in pipeline._launched:
        worker._abort_requested.set()

    try:
        pipeline.abort()
    finally:
        for worker in pipeline._launched:
            worker._wait(timeout=6.0)


def test_workers_deliver_products_and_finish() -> None:
    """A source and sink deliver every value and close their channels normally.

    The collector must finish after the source ends publication. Successful
    completion must also prevent the source from publishing additional values.
    """
    source = BooleanSource((False, True))
    sink = CollectingWorker(upstream=source)
    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert sink.values == [False, True]
        assert source.published == 2
        assert pipeline.done
        assert pipeline.exception is None

        with pytest.raises(RuntimeError, match="closed"):
            source._publish(False, timeout=0.0)
    finally:
        _abort_and_join(pipeline)


def test_worker_fanout_delivers_every_product_to_each_consumer() -> None:
    """Both branches receive the complete sequence from one source.

    Each successful source publication must reach both consumers in order.
    Completion requires the source and both consumers to finish normally.
    """
    values = (False, True, False, True)
    source = BooleanSource(values)
    first = CollectingWorker(upstream=source)
    second = CollectingWorker(upstream=source)
    pipeline = Pipeline(first, second)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert first.values == list(values)
        assert second.values == list(values)
        assert source.published == len(values)
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_consumer_exit_does_not_block_source() -> None:
    """A source can finish after its only consumer exits early.

    The sink consumes one value and returns. Its subscription must detach so
    later publications cannot stall on a queue that nobody will drain.
    """
    source = BooleanSource((False, True, False, True))
    sink = FirstOnlyWorker(upstream=source)
    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert sink.values == [False]
        assert source.published == 4
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_early_branch_exit_preserves_other_branch() -> None:
    """An early consumer exit leaves a sibling's deliveries intact.

    One branch consumes only the first value while the other drains the stream.
    Detaching the early branch must let the source finish and preserve the
    complete sequence for the remaining consumer.
    """
    values = (False, True, False, True)
    source = BooleanSource(values)
    early = FirstOnlyWorker(upstream=source)
    complete = CollectingWorker(upstream=source)
    pipeline = Pipeline(early, complete)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert early.values == [False]
        assert complete.values == list(values)
        assert source.published == len(values)
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_processing_chain_transforms_products_and_finishes() -> None:
    """A processor transforms each input and propagates normal stream completion.

    The processor inverts incoming booleans and publishes new payloads.
    Upstream completion must let it finish, which in turn must let the final
    collector finish after receiving every transformed value.
    """

    class Inverter(StubWorker[bool, bool]):
        def _work(self) -> None:
            try:
                while True:
                    with self._receive(timeout=5.0) as product:
                        self._publish(not product.payload, timeout=5.0)
            except EndOfStream:
                return

    source = BooleanSource((False, True, False))
    processor = Inverter(upstream=source)
    sink = CollectingWorker(upstream=processor)
    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert sink.values == [True, False, True]
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_pipeline_abort_wakes_worker_receiving_from_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pipeline cancellation wakes a consumer blocked on its upstream channel.

    The source waits for cancellation without publishing anything. Observe the
    sink waiting for input before aborting.

    Both workers must finish without failure before the receive timeout expires.
    """

    class IdleSource(StubWorker[Never, bool]):
        def _work(self) -> None:
            self._abort_requested.wait()

    source = IdleSource()
    sink = CollectingWorker(upstream=source)
    pipeline = Pipeline(sink)
    waiting = _observe_wait(source._publisher._condition, monkeypatch)

    try:
        pipeline.start()

        assert waiting.wait(timeout=2.0)

        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert source._exception is None
        assert sink._exception is None
    finally:
        _abort_and_join(pipeline)


def test_pipeline_marks_all_workers_before_waking_channels() -> None:
    """Abort records cancellation on every worker before invoking channel hooks.

    Keep both workers running, then check every cancellation flag whenever a
    worker's abort hook is invoked. This ordering prevents a channel wake-up
    from exposing OperationCancelled to a worker whose request is not yet marked.
    """

    class CheckingWorker(AbortableWorker):
        def _request_abort(self) -> None:
            assert all(worker._abort_requested.is_set() for worker in workers)
            super()._request_abort()

    first = CheckingWorker()
    second = CheckingWorker(upstream=first)
    workers = (first, second)
    pipeline = Pipeline(second)

    try:
        pipeline.start()

        for worker in workers:
            assert worker.entered.wait(timeout=2.0)

        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert all(worker.close_calls == 1 for worker in workers)
    finally:
        _abort_and_join(pipeline)


def test_consumer_failure_cancels_blocked_source_and_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consumer failure cancels peers blocked on publication and reception.

    The failing consumer holds its first delivery while its queue fills with
    the second. The source then blocks on its third publication. Meanwhile,
    the sibling consumes the first two values and waits for another.

    Observe both blocked operations before triggering the consumer failure.
    The original error must be reported, both peers must stop without reporting
    cancellation as failure, and every worker must clean up exactly once.
    """
    failure = RuntimeError("consumer failed")
    holding_product = Event()
    trigger_failure = Event()
    producer_waiting = Event()
    receiver_waiting = Event()
    close_calls = {
        "source": 0,
        "failing": 0,
        "sibling": 0,
    }

    class CountingSource(BooleanSource):
        def _close(self) -> None:
            close_calls["source"] += 1

    class CountingCollector(CollectingWorker):
        def _close(self) -> None:
            close_calls["sibling"] += 1

    class FailingConsumer(StubWorker[bool, Never]):
        def _work(self) -> None:
            with self._receive(timeout=5.0):
                holding_product.set()

                if not trigger_failure.wait(timeout=10.0):
                    raise TimeoutError("Test did not trigger consumer failure.")

                raise failure

        def _close(self) -> None:
            close_calls["failing"] += 1

    source = CountingSource((False, True, False))
    failing = FailingConsumer(upstream=source)
    sibling = CountingCollector(upstream=source)
    pipeline = Pipeline(failing, sibling)
    original_wait = source._publisher._condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        caller = current_thread()

        if caller is source._thread and source.published == 2:
            producer_waiting.set()
        elif caller is sibling._thread and len(sibling.values) == 2:
            receiver_waiting.set()

        return original_wait(timeout)

    monkeypatch.setattr(
        source._publisher._condition,
        "wait",
        observed_wait,
    )

    try:
        pipeline.start()

        assert holding_product.wait(timeout=2.0)
        assert producer_waiting.wait(timeout=2.0)
        assert receiver_waiting.wait(timeout=2.0)

        trigger_failure.set()

        with pytest.raises(RuntimeError, match="consumer failed") as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert pipeline.done
        assert failing._exception is failure
        assert source._exception is None
        assert sibling._exception is None
        assert source.published == 2
        assert sibling.values == [False, True]
        assert close_calls == {
            "source": 1,
            "failing": 1,
            "sibling": 1,
        }
    finally:
        trigger_failure.set()
        _abort_and_join(pipeline)
