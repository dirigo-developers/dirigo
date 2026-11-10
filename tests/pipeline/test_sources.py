"""Test source contracts and hardware binding by pipelines.

Sources originate products without upstream dependencies. Hardware workers
may have upstream dependencies and borrow the caller's Hardware collection.

Verify binding before preparation, cleanup after failures, and preservation
of hardware for reuse across successive pipelines.
"""

from typing import Any, Never, cast

import pytest

from dirigo.hardware import Device, Hardware
from dirigo.pipeline import HardwareWorker, Pipeline, Source, Worker

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


class DeviceWorker(HardwareWorker[int, int]):
    """Use borrowed hardware and optionally forward an upstream product."""

    def __init__(
        self,
        *,
        upstream: Worker[Any, int] | None = None,
        device_name: str = "digitizer",
        fail_work: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(upstream=upstream, name=name)
        self.device_name = device_name
        self.fail_work = fail_work
        self.prepared = False
        self.worked = False
        self.closed = False
        self.hardware_seen: Hardware | None = None

    def _prepare(self) -> None:
        self.prepared = True
        self.hardware_seen = self._hardware
        self._hardware.require(self.device_name, FakeDevice)

    def _work(self) -> None:
        self.worked = True
        assert self._hardware is self.hardware_seen

        if self.fail_work:
            raise RuntimeError("Acquisition failed.")

        if self.upstream is None:
            self._publish(7, timeout=2.0)
        else:
            with self._receive(timeout=2.0) as product:
                self._publish(product, timeout=2.0)

    def _close(self) -> None:
        self.closed = True


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


def test_hardware_access_before_binding_raises() -> None:
    """A hardware source cannot access a collection before it has been bound."""
    source = DeviceWorker()

    with pytest.raises(RuntimeError, match="bound"):
        _ = source._hardware


def test_binding_is_allowed_once() -> None:
    """A second binding attempt fails without replacing the original collection."""
    first = Hardware({})
    source = DeviceWorker()

    source._bind_hardware(first)

    with pytest.raises(RuntimeError, match="bound"):
        source._bind_hardware(Hardware({}))

    assert source._hardware is first


def test_pipeline_rejects_invalid_hardware() -> None:
    """Pipeline construction rejects hardware that is not a Hardware collection."""
    invalid_hardware = cast(Any, object())

    with pytest.raises(TypeError, match="hardware"):
        Pipeline(NumberSource(), hardware=invalid_hardware)


@pytest.mark.parametrize("with_upstream", [False, True])
def test_pipeline_binds_hardware_worker_before_preparation(
    with_upstream: bool,
) -> None:
    """Bind an automatically included hardware worker in either graph role."""
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    upstream = NumberSource() if with_upstream else None
    worker = DeviceWorker(upstream=upstream, name="device")
    sink = Collector(worker)
    pipeline = Pipeline(sink, hardware=hardware)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert worker.upstream is upstream
        assert worker.name == "device"
        assert worker.prepared
        assert worker.hardware_seen is hardware
        assert worker.worked
        assert worker.closed
        assert sink.values == ([36] if with_upstream else [7])
        assert pipeline.done
        assert pipeline.exception is None

        assert device.close_calls == 0
        assert hardware.require("digitizer", FakeDevice) is device
    finally:
        _abort_and_join(pipeline)
        hardware.close()


def test_all_hardware_workers_are_bound_before_any_preparation() -> None:
    """Bind downstream hardware before even the first preparation hook."""
    hardware = Hardware({"digitizer": FakeDevice()})
    upstream = NumberSource()
    second = DeviceWorker(upstream=upstream, name="second")

    class FirstWorker(DeviceWorker):
        def _prepare(self) -> None:
            assert second._hardware is hardware
            super()._prepare()

    first = FirstWorker(name="first")
    sink = Collector(second)
    pipeline = Pipeline(first, sink, hardware=hardware)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert first.hardware_seen is hardware
        assert second.hardware_seen is hardware
        assert first.worked
        assert second.worked
        assert first.closed
        assert second.closed
        assert sink.values == [36]
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)
        hardware.close()


@pytest.mark.parametrize("with_upstream", [False, True])
def test_missing_hardware_fails_before_any_worker_prepares(
    with_upstream: bool,
) -> None:
    """Reject missing hardware before preparing any part of the graph."""
    synthetic = NumberSource()
    live = DeviceWorker(
        upstream=synthetic if with_upstream else None,
        name="live",
    )
    pipeline = Pipeline(synthetic, live)

    try:
        with pytest.raises(RuntimeError, match="live") as caught:
            pipeline.start()

        assert not synthetic.prepared
        assert not live.prepared
        assert not live.worked
        assert not live.closed
        assert pipeline.done
        assert pipeline.exception is caught.value

        with pytest.raises(RuntimeError, match="live") as waited:
            pipeline.wait(timeout=2.0)

        assert waited.value is caught.value
    finally:
        _abort_and_join(pipeline)


def test_device_lookup_failure_cleans_source_but_preserves_hardware() -> None:
    """A device lookup failure cleans the attempted source without closing hardware.

    The collection exists, but the source requests a missing device name.
    Preparation must fail before work begins, and source cleanup must still run.

    The configured device must remain open and available through the collection.
    """
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceWorker(device_name="missing")
    pipeline = Pipeline(source, hardware=hardware)

    try:
        with pytest.raises(KeyError, match="missing"):
            pipeline.start()

        assert source.closed
        assert not source.worked
        assert pipeline.done
        assert device.close_calls == 0
        assert hardware.require("digitizer", FakeDevice) is device
    finally:
        _abort_and_join(pipeline)
        hardware.close()


def test_downstream_preparation_failure_preserves_hardware() -> None:
    """A downstream preparation failure cleans the source but preserves its hardware.

    The source prepares successfully before its collector fails preparation.
    Rollback must close the source without starting its work or closing the
    borrowed device collection.
    """

    class FailingCollector(Collector):
        def _prepare(self) -> None:
            raise RuntimeError("Downstream preparation failed.")

    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceWorker()
    pipeline = Pipeline(FailingCollector(source), hardware=hardware)

    try:
        with pytest.raises(RuntimeError, match="Downstream preparation failed"):
            pipeline.start()

        assert source.closed
        assert not source.worked
        assert device.close_calls == 0
        assert hardware.require("digitizer", FakeDevice) is device
    finally:
        _abort_and_join(pipeline)
        hardware.close()


def test_execution_failure_preserves_hardware() -> None:
    """A source execution failure triggers cleanup without closing borrowed hardware.

    Waiting must report the acquisition failure after source cleanup completes.
    The device must remain open and available to the caller afterward.
    """
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceWorker(fail_work=True)
    pipeline = Pipeline(source, hardware=hardware)

    try:
        pipeline.start()

        with pytest.raises(RuntimeError, match="Acquisition"):
            pipeline.wait(timeout=2.0)

        assert source.closed
        assert pipeline.done
        assert device.close_calls == 0
        assert hardware.require("digitizer", FakeDevice) is device
    finally:
        _abort_and_join(pipeline)
        hardware.close()


def test_hardware_can_be_reused_across_successive_pipelines() -> None:
    """Fresh pipelines and workers can successively borrow the same hardware.

    Each pipeline must finish without closing the shared device. Only the
    caller's explicit Hardware.close() should close it, exactly once.
    """
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})

    try:
        for _ in range(2):
            source = DeviceWorker()
            sink = Collector(source)
            pipeline = Pipeline(sink, hardware=hardware)

            try:
                pipeline.start()
                pipeline.wait(timeout=2.0)

                assert sink.values == [7]
            finally:
                _abort_and_join(pipeline)

        assert device.close_calls == 0

        hardware.close()

        assert device.close_calls == 1
    finally:
        hardware.close()
