import pytest

from dirigo.pipeline import EndOfStream, Pipeline, Worker


class RecordingWorker(Worker[int, int]):
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


@pytest.mark.parametrize("late_branch", [False, True])
@pytest.mark.parametrize("branch_from_processor", [False, True])
def test_rejects_incomplete_graph_before_preparation_and_can_be_repaired(
    late_branch: bool,
    branch_from_processor: bool,
) -> None:
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
        with pytest.raises(ValueError, match="outside this pipeline") as error:
            pipeline.start()

        assert repr(branch_point.name) in str(error.value)
        assert all(not worker.prepared for worker in workers)
        assert all(not worker.closed for worker in workers)
        assert all(worker._thread is None for worker in workers)
        assert pipeline.exception is None
        assert not pipeline.done

        pipeline.add(omitted_sink)
        pipeline.start()
        pipeline.wait(timeout=3.0)

        assert sink.received == [0, 1, 2]
        assert omitted_sink.received == [0, 1, 2]
        assert all(worker.prepared and worker.closed for worker in workers)
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        if any(worker._thread is not None for worker in workers):
            pipeline.abort()
            pipeline.wait(timeout=3.0)


def test_detached_subscription_is_allowed() -> None:
    source = RecordingWorker(name="source")
    sink = RecordingWorker(name="sink", upstream=source)
    detached = source._publisher.subscribe()
    detached.close()
    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=3.0)
        assert sink.received == [0, 1, 2]
    finally:
        if source._thread is not None:
            pipeline.abort()
            pipeline.wait(timeout=3.0)


def test_source_without_subscribers_is_allowed() -> None:
    source = RecordingWorker(name="source")
    pipeline = Pipeline(source)

    try:
        pipeline.start()
        pipeline.wait(timeout=3.0)
        assert pipeline.done
        assert source.prepared and source.closed
    finally:
        if source._thread is not None:
            pipeline.abort()
            pipeline.wait(timeout=3.0)
