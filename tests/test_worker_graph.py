from threading import Event, current_thread
from typing import Never

import pytest

from dirigo import Pipeline
from dirigo.pipeline import EndOfStream

from .stubs import BooleanSource, CollectingWorker, StubWorker


def _abort_and_join(pipeline: Pipeline) -> None:
    """Ensure failed assertions cannot leave worker threads running."""
    pipeline.abort()

    for worker in pipeline.workers:
        worker._wait(timeout=2.0)


def test_worker_fanout_delivers_every_product_to_each_consumer() -> None:
    values = (False, True, False, True)
    source = BooleanSource(values)
    first = CollectingWorker(upstream=source)
    second = CollectingWorker(upstream=source)

    pipeline = Pipeline(first, second)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        assert first.values == list(values)
        assert second.values == list(values)
        assert source.published == len(values)
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_early_branch_exit_preserves_other_branch() -> None:
    class FirstOnlyWorker(CollectingWorker):
        def _work(self) -> None:
            with self._receive(timeout=2.0) as product:
                self.values.append(product.payload)

    values = (False, True, False, True)
    source = BooleanSource(values)
    early = FirstOnlyWorker(upstream=source)
    complete = CollectingWorker(upstream=source)

    pipeline = Pipeline(early, complete)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        assert early.values == [False]
        assert complete.values == list(values)
        assert source.published == len(values)
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_processing_chain_transforms_products_and_finishes() -> None:
    class Inverter(StubWorker[bool, bool]):
        def _work(self) -> None:
            try:
                while True:
                    with self._receive(timeout=2.0) as product:
                        self._publish(
                            not product.payload,
                            timeout=2.0,
                        )
            except EndOfStream:
                return

    source = BooleanSource((False, True, False))
    processor = Inverter(upstream=source)
    sink = CollectingWorker(upstream=processor)

    pipeline = Pipeline(sink)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        assert sink.values == [True, False, True]
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_consumer_failure_cancels_blocked_source_and_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            with self._receive(timeout=2.0):
                holding_product.set()

                if not trigger_failure.wait(timeout=3.0):
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

    pipeline.start()

    try:
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

