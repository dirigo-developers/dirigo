from threading import Event
from typing import Any, Never

import pytest

from dirigo import Pipeline
from dirigo.pipeline import EndOfStream, OperationCancelled, Worker

from .stubs import StubWorker


class PooledSource(StubWorker[Never, bytearray]):
    def __init__(self) -> None:
        super().__init__()
        self.published = 0

    def _prepare(self) -> None:
        self._init_pool(capacity=1, factory=lambda: bytearray(4))

    def _work(self) -> None:
        for value in (b"one!", b"two!"):
            with self._acquire(timeout=2.0) as outgoing:
                outgoing.payload[:] = value
                self._publish(outgoing, timeout=2.0)
                self.published += 1


class BufferCollector(StubWorker[bytearray, Never]):
    def __init__(self, *, upstream: Worker[Any, bytearray]) -> None:
        super().__init__(upstream=upstream)
        self.snapshots: list[bytes] = []
        self.buffer_ids: list[int] = []

    def _work(self) -> None:
        try:
            while True:
                with self._receive(timeout=2.0) as product:
                    self.snapshots.append(bytes(product.payload))
                    self.buffer_ids.append(id(product.payload))
        except EndOfStream:
            return


def _abort_and_join(pipeline: Pipeline) -> None:
    pipeline.abort()
    for worker in pipeline._launched:
        worker._wait(timeout=2.0)


def test_worker_publishes_leases_and_reuses_storage() -> None:
    source = PooledSource()
    sink = BufferCollector(upstream=source)
    pipeline = Pipeline(sink)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)
        assert sink.snapshots == [b"one!", b"two!"]
        assert len(sink.buffer_ids) == 2
        assert sink.buffer_ids[0] == sink.buffer_ids[1]
        assert source.published == 2
        assert pipeline.done
        assert pipeline.exception is None

        # A completed worker no longer offers new output acquisitions.
        with pytest.raises(OperationCancelled), source._acquire(timeout=0.0):
            pass
    finally:
        _abort_and_join(pipeline)


def test_pool_configuration_does_not_change_plain_payload_publication() -> None:
    raw = bytearray(b"raw!")

    class PlainSource(PooledSource):
        def _work(self) -> None:
            # Exhaust the output pool while publishing a separate plain payload.
            with self._acquire(timeout=0.0) as held:
                held.payload[:] = b"held"
                self._publish(raw, timeout=2.0)

    source = PlainSource()
    sink = BufferCollector(upstream=source)
    pipeline = Pipeline(sink)
    pipeline.start()
    try:
        pipeline.wait(timeout=2.0)
        assert sink.snapshots == [b"raw!"]
        assert sink.buffer_ids == [id(raw)]
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_worker_returns_an_unpublished_output_buffer() -> None:
    class UnpublishedSource(PooledSource):
        def __init__(self) -> None:
            super().__init__()
            self.reused = False

        def _work(self) -> None:
            with self._acquire(timeout=0.0) as first:
                original = first.payload
            with self._acquire(timeout=0.0) as second:
                self.reused = second.payload is original

    source = UnpublishedSource()
    pipeline = Pipeline(source)
    pipeline.start()
    try:
        pipeline.wait(timeout=2.0)
        assert source.reused
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_abort_wakes_producer_waiting_for_a_consumer_to_release_its_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_waiting = Event()
    holding_product = Event()

    class ObservedSource(PooledSource):
        def _prepare(self) -> None:
            super()._prepare()
            pool = self._pool
            assert pool is not None
            original_wait = pool._condition.wait

            def observed_wait(timeout: float | None = None) -> bool:
                pool_waiting.set()
                return original_wait(timeout)

            monkeypatch.setattr(pool._condition, "wait", observed_wait)

    class HoldingConsumer(BufferCollector):
        def _work(self) -> None:
            with self._receive(timeout=2.0) as product:
                holding_product.set()
                if not self._abort_requested.wait(timeout=3.0):
                    raise TimeoutError("Test did not request cancellation.")
                # Cancellation must not invalidate this borrowed buffer.
                self.snapshots.append(bytes(product.payload))

    source = ObservedSource()
    sink = HoldingConsumer(upstream=source)
    pipeline = Pipeline(sink)
    pipeline.start()
    try:
        assert holding_product.wait(timeout=2.0)
        assert pool_waiting.wait(timeout=2.0)

        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert source.published == 1
        assert sink.snapshots == [b"one!"]
        assert pipeline.done
        assert pipeline.exception is None
        assert source._exception is None
        assert sink._exception is None
    finally:
        _abort_and_join(pipeline)


def test_worker_acquisition_requires_an_output_pool() -> None:
    worker = StubWorker[Never, bytearray]()
    with pytest.raises(RuntimeError, match="no output pool"):
        worker._acquire(timeout=0.0)


def test_worker_cannot_replace_its_output_pool() -> None:
    worker = StubWorker[Never, bytearray]()
    original = bytearray(4)
    worker._init_pool(capacity=1, factory=lambda: original)

    with pytest.raises(RuntimeError, match="already initialized"):
        worker._init_pool(capacity=2, factory=lambda: bytearray(4))

    with worker._acquire(timeout=0.0) as lease:
        assert lease.payload is original
        with pytest.raises(TimeoutError), worker._acquire(timeout=0.0):
            pass


def test_pool_initialized_after_abort_inherits_cancellation() -> None:
    worker = StubWorker[Never, bytearray]()
    worker._request_abort()
    worker._init_pool(capacity=1, factory=lambda: bytearray(4))

    with pytest.raises(OperationCancelled), worker._acquire(timeout=0.0):
        pass


@pytest.mark.parametrize("stage", ["prepare", "close"])
def test_resource_hook_failure_retires_output_pool(stage: str) -> None:
    failure = OSError("resource hook failed")

    class FailingSource(PooledSource):
        def _prepare(self) -> None:
            super()._prepare()
            if stage == "prepare":
                raise failure

        def _close(self) -> None:
            if stage == "close":
                raise failure

    source = FailingSource()
    pipeline = Pipeline(source)
    try:
        if stage == "prepare":
            with pytest.raises(OSError) as caught:
                pipeline.start()
        else:
            pipeline.start()
            with pytest.raises(OSError) as caught:
                pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert pipeline.done
        with pytest.raises(OperationCancelled), source._acquire(timeout=0.0):
            pass
    finally:
        _abort_and_join(pipeline)


