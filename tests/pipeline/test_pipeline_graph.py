"""Test pipeline graph construction and validation before execution.

Verify upstream dependency inclusion, deterministic ordering, shared dependencies,
and rejection of invalid additions, cycles, empty graphs, and omitted consumers.

Graph validation must happen before preparation or thread launch. Rejection must
leave the pipeline available for repair and a subsequent start.

Worker ownership and execution lifecycle behavior are tested separately.
"""

import pytest

from dirigo.pipeline import EndOfStream, Pipeline, Worker

from .stubs import StubWorker


class RecordingWorker(StubWorker[int, int]):
    """Produce three values, or record and forward values from an upstream."""

    def __init__(
        self,
        *,
        name: str,
        upstream: Worker[int, int] | None = None,
    ) -> None:
        super().__init__(name=name, upstream=upstream)
        self.prepared = False
        self.closed = False
        self.received: list[int] = []

    def _prepare(self) -> None:
        self.prepared = True

    def _work(self) -> None:
        if self.upstream is None:
            for value in range(3):
                self._publish(value, timeout=1.0)
            return

        while True:
            try:
                with self._receive(timeout=1.0) as product:
                    self.received.append(product.payload)
                    self._publish(product, timeout=1.0)
            except EndOfStream:
                return

    def _close(self) -> None:
        self.closed = True


def _abort_and_join(pipeline: Pipeline) -> None:
    """Cancel and join launched workers, allowing for rejection before launch."""
    if not pipeline._launched:
        return

    pipeline.abort()

    for worker in pipeline._launched:
        worker._wait(timeout=3.0)


def test_new_pipeline_has_no_workers() -> None:
    """An empty constructor creates a graph with no included workers."""
    pipeline = Pipeline()

    assert pipeline.workers == ()


def test_add_rejects_non_worker() -> None:
    """Rejecting a non-worker leaves the graph unchanged."""
    pipeline = Pipeline()

    with pytest.raises(TypeError, match="worker must be a Worker"):
        pipeline.add(worker=object())  # ty: ignore[invalid-argument-type]

    assert pipeline.workers == ()


def test_pipeline_construction_with_non_worker_fails() -> None:
    """Pipeline construction rejects arguments that are not Worker instances."""
    with pytest.raises(TypeError, match="worker must be a Worker"):
        Pipeline(object()) # ty: ignore[invalid-argument-type]


def test_add_includes_dependencies_upstream_first() -> None:
    """Adding a terminal worker includes its entire upstream chain in order.

    The caller supplies only the consumer. The pipeline must discover both its
    processor and source, placing every dependency before its downstream worker.
    """
    source = StubWorker()
    processor = StubWorker(upstream=source)
    consumer = StubWorker(upstream=processor)
    pipeline = Pipeline()

    pipeline.add(consumer)

    assert pipeline.workers == (source, processor, consumer)


def test_add_includes_shared_dependencies_only_once() -> None:
    """Branches share one included source while retaining dependency order.

    Adding both terminal workers must not duplicate their common upstream.
    Newly discovered workers follow the order in which branches are added.
    """
    source = StubWorker()
    a1 = StubWorker(upstream=source)
    a2 = StubWorker(upstream=a1)
    b1 = StubWorker(upstream=source)
    b2 = StubWorker(upstream=b1)

    pipeline = Pipeline(a2, b2)

    assert pipeline.workers == (source, a1, a2, b1, b2)


def test_add_raises_value_error_for_cycle() -> None:
    """A cycle is rejected without partially adding its workers to the graph.

    Deliberately alter the private upstream reference to construct a cycle that
    cannot be established through ordinary immutable upstream relationships.
    """
    first = StubWorker()
    second = StubWorker(upstream=first)
    first._upstream = second

    pipeline = Pipeline()

    with pytest.raises(ValueError, match="cycle"):
        pipeline.add(second)

    assert pipeline.workers == ()


def test_start_rejects_empty_pipeline() -> None:
    """Rejecting an empty graph does not consume the pipeline's single execution.

    After rejection, adding a worker and starting again must succeed. This
    distinguishes graph validation from an execution attempt that has begun.
    """
    pipeline = Pipeline()

    try:
        with pytest.raises(ValueError, match="empty"):
            pipeline.start()

        pipeline_done = pipeline.done
        assert not pipeline_done
        assert pipeline.exception is None

        worker = StubWorker()
        pipeline.add(worker)
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


@pytest.mark.parametrize("late_branch", [False, True])
@pytest.mark.parametrize("branch_from_processor", [False, True])
def test_rejects_incomplete_graph_before_preparation_and_can_be_repaired(
    late_branch: bool,
    branch_from_processor: bool,
) -> None:
    """An omitted connected consumer is rejected before any resources are prepared.

    Attach an extra consumer to either the source or the processor, before or
    after constructing the pipeline. In every case, starting must identify the
    branch point without preparing, closing, or launching any worker.

    Adding the omitted consumer must repair the same pipeline. A subsequent
    start must deliver all values to both terminal consumers and finish normally.
    """
    source = RecordingWorker(name="source")
    processor = RecordingWorker(name="processor", upstream=source)
    sink = RecordingWorker(name="sink", upstream=processor)
    branch_point = processor if branch_from_processor else source

    if late_branch:
        pipeline = Pipeline(sink)
        omitted_sink = RecordingWorker(name="omitted", upstream=branch_point)
    else:
        omitted_sink = RecordingWorker(name="omitted", upstream=branch_point)
        pipeline = Pipeline(sink)

    workers = (source, processor, sink, omitted_sink)

    try:
        with pytest.raises(ValueError, match="outside this pipeline") as caught:
            pipeline.start()

        assert repr(branch_point.name) in str(caught.value)
        assert all(not worker.prepared for worker in workers)
        assert all(not worker.closed for worker in workers)
        assert all(worker._thread is None for worker in workers)
        assert pipeline.exception is None
        pipeline_done = pipeline.done
        assert not pipeline_done

        pipeline.add(omitted_sink)
        pipeline.start()
        pipeline.wait(timeout=3.0)

        assert sink.received == [0, 1, 2]
        assert omitted_sink.received == [0, 1, 2]
        assert all(worker.prepared and worker.closed for worker in workers)
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_detached_subscription_is_allowed() -> None:
    """A detached subscription does not count as an omitted active consumer.

    Create an additional subscription directly on the source, then close it
    before starting. Validation must ignore that detached subscription and
    allow the included source-to-sink graph to execute normally.
    """
    source = RecordingWorker(name="source")
    sink = RecordingWorker(name="sink", upstream=source)
    detached = source._publisher.subscribe()

    detached.close()

    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=3.0)

        assert sink.received == [0, 1, 2]
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_source_without_subscribers_is_allowed() -> None:
    """A source with no consumers is a valid graph and can finish normally.

    Completeness requires including every active consumer that exists; it does
    not require creating a consumer for every publisher.
    """
    source = RecordingWorker(name="source")
    pipeline = Pipeline(source)

    try:
        pipeline.start()
        pipeline.wait(timeout=3.0)

        assert source.prepared
        assert source.closed
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)
