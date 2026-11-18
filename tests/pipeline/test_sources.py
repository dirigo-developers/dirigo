"""Test source contracts and hardware binding by pipelines.

Sources originate products without upstream dependencies. Hardware workers
may have upstream dependencies and borrow the caller's Hardware collection.

Verify binding before preparation, cleanup after failures, and preservation
of hardware for reuse across successive pipelines.
"""

from typing import Any, Never, cast

import pytest

from dirigo.hardware import Device
from dirigo.pipeline import Pipeline, Source, Worker

from .stubs import StubWorker


class FakeDevice(Device):
    """Record device closure without performing hardware operations."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class NumberSource(Source[int]):
    """Publish one synthetic value and record whether preparation occurred."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.prepared = False

    def _prepare(self) -> None:
        self.prepared = True

    def _work(self) -> None:
        self._publish(36, timeout=2.0)

    def _close(self) -> None:
        pass


class Collector(StubWorker[int, Never]):
    """Collect the single value produced by a source."""

    def __init__(self, upstream: Worker[Any, int]) -> None:
        super().__init__(upstream=upstream)
        self.values: list[int] = []

    def _work(self) -> None:
        with self._receive(timeout=2.0) as product:
            self.values.append(product.payload)


def _abort_and_join(pipeline: Pipeline) -> None:
    """Cancel and join launched workers without re-raising recorded failures."""
    if not pipeline._launched:
        return

    pipeline.abort()

    for worker in pipeline._launched:
        worker._wait(timeout=2.0)


def test_source_runs_without_hardware() -> None:
    """A synthetic source runs without a Hardware collection.

    The source must retain its supplied name, have no upstream dependency,
    and deliver its value to the downstream collector.
    """
    source = NumberSource(name="generated")
    sink = Collector(source)
    pipeline = Pipeline(sink)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert source.upstream is None
        assert source.name == "generated"
        assert sink.values == [36]
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_sources_do_not_accept_upstream() -> None:
    """The Source convenience base does not accept an upstream."""
    untyped_source = cast(Any, NumberSource)

    with pytest.raises(TypeError, match="upstream"):
        untyped_source(upstream=NumberSource())
