"""Test worker output pools and producer-lease integration.

Verify that workers can publish pooled storage, reuse it after consumption,
and return buffers that were acquired but never published. Configuring an
output pool must not alter ordinary payload publication.

Cancellation must interrupt acquisition, including when a consumer still holds
the only buffer. Output pools must also remain cancelled after lifecycle-hook
failures.
"""

from threading import Event
from typing import Any, Never

import pytest

from dirigo.pipeline import EndOfStream, OperationCancelled, Pipeline, Worker

from .stubs import StubWorker


class PooledSource(StubWorker[Never, bytearray]):
    """Publish two values through a single reusable output buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.published = 0

    def _prepare(self) -> None:
        self._init_pool(capacity=1, factory=lambda: bytearray(4))

    def _work(self) -> None:
        for value in (b"one!", b"two!"):
            with self._acquire(timeout=5.0) as outgoing:
                outgoing.payload[:] = value
                self._publish(outgoing, timeout=5.0)
                self.published += 1


class BufferCollector(StubWorker[bytearray, Never]):
    """Record payload contents and storage identities while deliveries are valid."""

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
    """Cancel execution and join threads without re-raising worker failures.

    Join only launched workers because preparation may fail before any thread
    starts. Direct worker waits avoid masking an assertion with a failure that
    Pipeline.wait() would report again.
    """
    pipeline.abort()

    for worker in pipeline._launched:
        worker._wait(timeout=6.0)


def test_worker_publishes_leases_and_reuses_storage() -> None:
    """A worker publishes both values using the same pooled buffer.

    The collector copies each payload while its delivery is valid. Its snapshots
    must preserve the two different values, while recorded identities prove
    that the source reused one buffer.

    Normal worker completion must retire the output pool so further acquisition
    reports cancellation.
    """
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

        with pytest.raises(OperationCancelled), source._acquire(timeout=0.0):
            pass
    finally:
        _abort_and_join(pipeline)


def test_pool_configuration_does_not_change_plain_payload_publication() -> None:
    """A worker with an output pool can still publish an independent raw payload.

    Hold the pool's only buffer while publishing a different bytearray. Delivery
    must succeed without requiring another pooled acquisition or substituting
    pooled storage for the supplied payload.
    """
    raw = bytearray(b"raw!")

    class PlainSource(PooledSource):
        def _work(self) -> None:
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
    """Leaving an acquisition context returns a buffer even without publication.

    Acquire and release the sole output buffer, then immediately acquire again.
    Successful reacquisition of the same object proves that unpublished output
    does not leave an outstanding producer claim.
    """

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
    """Cancellation interrupts a producer blocked on its exhausted output pool.

    The consumer holds the first delivery while the source tries to acquire
    storage for its second value. Observe both conditions before aborting.

    The producer must stop without publishing a second value. The consumer
    must still be able to read its borrowed buffer after receiving the abort
    request, and neither worker should report expected cancellation as failure.
    """
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

                if not self._abort_requested.wait(timeout=5.0):
                    raise TimeoutError("Test did not request cancellation.")

                # The receive context still holds a claim after cancellation.
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
    """Acquiring output storage before pool initialization fails immediately."""
    worker = StubWorker[Never, bytearray]()

    with pytest.raises(RuntimeError, match="no output pool"):
        worker._acquire(timeout=0.0)


def test_worker_cannot_replace_its_output_pool() -> None:
    """A rejected second initialization preserves the original pool and capacity.

    Initialize a one-buffer pool, then attempt to replace it with a larger pool.
    The original buffer must remain available, and borrowing it must still
    exhaust the pool.
    """
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
    """Initializing storage after an abort request must not enable acquisition.

    Cancellation may precede pool creation. The newly created pool must inherit
    that state rather than allow the worker to acquire fresh output storage.
    """
    worker = StubWorker[Never, bytearray]()

    worker._request_abort()
    worker._init_pool(capacity=1, factory=lambda: bytearray(4))

    with pytest.raises(OperationCancelled), worker._acquire(timeout=0.0):
        pass


@pytest.mark.parametrize("stage", ["prepare", "close"])
def test_resource_hook_failure_retires_output_pool(stage: str) -> None:
    """Preparation or cleanup failure must leave the output pool cancelled.

    The preparation scenario creates its pool before failing, exercising cleanup
    of partially prepared resources. The cleanup scenario fails after execution.

    Both must preserve the original failure and prevent subsequent acquisition,
    even though a lifecycle hook did not complete successfully.
    """
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
            with pytest.raises(OSError, match="hook failed") as caught:
                pipeline.start()
        else:
            pipeline.start()

            with pytest.raises(OSError, match="hook failed") as caught:
                pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert pipeline.done

        with pytest.raises(OperationCancelled), source._acquire(timeout=0.0):
            pass
    finally:
        _abort_and_join(pipeline)
