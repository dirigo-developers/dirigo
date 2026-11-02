import threading

import pytest

from dirigo.exceptions import EndOfStream
from dirigo.pipeline import Pipeline
from dirigo.source import Source
from dirigo.worker import Worker

PUBLISH_TIMEOUT = 0.5   # turns a stall into an error we can match
WAIT_TIMEOUT = 5.0      # safety net so a broken test can't hang CI


class Counter(Source[int]):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def _prepare(self) -> None: ...
    def _close(self) -> None: ...

    def _work(self) -> None:
        for i in range(self.count):
            self._publish(i, timeout=PUBLISH_TIMEOUT)


class Collector(Worker[int, None]):
    def __init__(self, upstream: Worker[object, int]) -> None:
        super().__init__(upstream=upstream)
        self.received: list[int] = []

    def _prepare(self) -> None: ...
    def _close(self) -> None: ...

    def _work(self) -> None:
        while True:
            try:
                with self._receive() as product:
                    self.received.append(product.payload)
            except EndOfStream:  # handled here so issue 1.4 doesn't get in the way
                return


# --- 1.1: subscribing at construction ---------------------------------------

def test_control_linear_graph_completes() -> None:
    source = Counter(3)
    sink = Collector(source)
    pipeline = Pipeline(sink)

    pipeline.start()
    pipeline.wait(timeout=WAIT_TIMEOUT)

    assert sink.received == [0, 1, 2]


def test_unadded_downstream_worker_stalls_source() -> None:
    source = Counter(3)
    sink = Collector(source)
    Collector(source)  # never added to any pipeline; the reference is even discarded

    pipeline = Pipeline(sink)
    pipeline.start()

    # The orphan's queue (capacity=1) fills with item 0 and is never drained,
    # so publishing item 1 blocks until the source's publish timeout.
    with pytest.raises(TimeoutError, match="Publication exceeded"):
        pipeline.wait(timeout=WAIT_TIMEOUT)

    assert sink.received == [0]


# --- 1.2: a worker can join two pipelines -----------------------------------

class GatedSource(Source[int]):
    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self.gate = gate
        self.prepare_calls = 0
        self.close_calls = 0

    def _prepare(self) -> None:
        self.prepare_calls += 1

    def _close(self) -> None:
        self.close_calls += 1

    def _work(self) -> None:
        self.gate.wait(WAIT_TIMEOUT)


def test_second_pipeline_reprepares_and_closes_a_running_worker() -> None:
    gate = threading.Event()
    source = GatedSource(gate)
    first = Pipeline(source)
    first.start()

    try:
        second = Pipeline(source)  # accepted: workers aren't claimed by a pipeline

        with pytest.raises(RuntimeError, match="Worker execution has already started"):
            second.start()

        assert source.prepare_calls == 2   # _prepare() ran a second time
        assert source.close_calls == 1     # second pipeline closed it mid-run
        assert source._thread is not None and source._thread.is_alive()
        assert first.done                  # first pipeline says done while its thread is running
    finally:
        gate.set()
        first.wait(timeout=WAIT_TIMEOUT)

    assert source.close_calls == 2         # closed again by its own thread