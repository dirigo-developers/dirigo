from typing import Any, Never

import pytest

from dirigo.pipeline import EndOfStream, Pipeline, Sink, Worker

from .stubs import StubWorker


class StubSink(Sink[int]):
    """Implement lifecycle hooks without application behavior."""

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        pass

    def _close(self) -> None:
        pass


class NumberSource(StubWorker[Never, int]):
    """Publish a finite sequence of integers."""

    def __init__(self, values: tuple[int, ...]) -> None:
        super().__init__()
        self.values = values

    def _work(self) -> None:
        for value in self.values:
            self._publish(value, timeout=2.0)


class CollectingSink(StubSink):
    """Collect incoming values until normal end-of-stream."""

    def __init__(self, *, upstream: Worker[Any, int]) -> None:
        super().__init__(upstream=upstream)
        self.values: list[int] = []
        self.closed = False

    def _work(self) -> None:
        while True:
            try:
                with self._receive(timeout=2.0) as product:
                    self.values.append(product.payload)
            except EndOfStream:
                return

    def _close(self) -> None:
        self.closed = True


def test_sink_requires_upstream_argument() -> None:
    """Constructing a concrete sink without an upstream argument raises TypeError."""
    with pytest.raises(TypeError, match="upstream"):
        StubSink()  # ty: ignore[missing-argument]


@pytest.mark.parametrize("upstream", [None, object()])
def test_sink_rejects_invalid_upstream(upstream: object) -> None:
    """A sink rejects None and objects that are not Worker instances."""
    with pytest.raises(TypeError, match="upstream"):
        StubSink(upstream=upstream) # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("name", [None, "recording", ""])
def test_sink_preserves_upstream_and_name(name: str | None) -> None:
    """A sink retains its upstream and follows Worker naming rules.

    None selects the concrete class name; explicit strings are preserved,
    including an empty string.
    """
    upstream = StubWorker[Never, int]()

    sink = StubSink(upstream=upstream, name=name)

    assert sink.upstream is upstream
    assert sink.name == ("StubSink" if name is None else name)


@pytest.mark.parametrize("values", [(), (3, 1, 4)])
def test_sink_consumes_stream_and_closes(
    values: tuple[int, ...],
) -> None:
    source = NumberSource(values)
    sink = CollectingSink(upstream=source)
    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert sink.values == list(values)
        assert sink.closed
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        # Ensure failed assertions or timeouts cannot leave threads running.
        pipeline.abort()
        for worker in pipeline._launched:
            worker._wait(timeout=2.0)
