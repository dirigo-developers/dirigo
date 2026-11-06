"""Test source contracts and hardware binding by pipelines.

Sources originate products without upstream dependencies. Hardware sources
borrow the caller's Hardware collection, which must be bound before any worker
begins preparation.

Verify binding validation, cleanup after preparation and execution failures,
and preservation of hardware for reuse across successive pipelines.
"""

from typing import Any, Never, cast

import pytest

from dirigo.hardware import Device, Hardware
from dirigo.pipeline import HardwareSource, Pipeline, Source, Worker

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


class DeviceSource(HardwareSource[int]):
    """Look up a borrowed device and record source lifecycle activity."""

    def __init__(
        self,
        *,
        device_name: str = "digitizer",
        fail_work: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
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

        self._publish(7, timeout=2.0)

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


@pytest.mark.parametrize("source_type", [NumberSource, DeviceSource])
def test_sources_do_not_accept_upstream(
    source_type: type[Source[int]],
) -> None:
    """Both synthetic and hardware sources reject an upstream argument."""
    untyped_source = cast(Any, source_type)

    with pytest.raises(TypeError, match="upstream"):
        untyped_source(upstream=NumberSource())


def test_hardware_access_before_binding_raises() -> None:
    """A hardware source cannot access a collection before it has been bound."""
    source = DeviceSource()

    with pytest.raises(RuntimeError, match="bound"):
        _ = source._hardware


def test_binding_is_allowed_once() -> None:
    """A second binding attempt fails without replacing the original collection."""
    first = Hardware({})
    source = DeviceSource()

    source._bind_hardware(first)

    with pytest.raises(RuntimeError, match="bound"):
        source._bind_hardware(Hardware({}))

    assert source._hardware is first


def test_pipeline_rejects_invalid_hardware() -> None:
    """Pipeline construction rejects hardware that is not a Hardware collection."""
    invalid_hardware = cast(Any, object())

    with pytest.raises(TypeError, match="hardware"):
        Pipeline(NumberSource(), hardware=invalid_hardware)


def test_pipeline_binds_upstream_source_before_preparation() -> None:
    """An automatically included upstream source receives hardware before preparing.

    Construct the pipeline from its sink alone. The source must see the supplied
    collection during preparation and execution, and clean up after completion.

    Source cleanup must leave the device and its Hardware collection open.
    """
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceSource()
    sink = Collector(source)
    pipeline = Pipeline(sink, hardware=hardware)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert source.hardware_seen is hardware
        assert source.worked
        assert source.closed
        assert sink.values == [7]
        assert pipeline.done
        assert pipeline.exception is None
        assert device.close_calls == 0
        assert hardware.require("digitizer", FakeDevice) is device
    finally:
        _abort_and_join(pipeline)
        hardware.close()


def test_all_hardware_sources_are_bound_before_any_preparation() -> None:
    """Hardware binding for the whole graph precedes the first preparation hook.

    The first source checks the second source's hardware during its own
    preparation. This rejects an implementation that binds and prepares each
    source individually before moving to the next.
    """
    hardware = Hardware({"digitizer": FakeDevice()})
    second = DeviceSource(name="second")

    class FirstSource(DeviceSource):
        def _prepare(self) -> None:
            assert second._hardware is hardware
            super()._prepare()

    first = FirstSource(name="first")
    pipeline = Pipeline(first, second, hardware=hardware)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        assert first.hardware_seen is hardware
        assert second.hardware_seen is hardware
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)
        hardware.close()


def test_missing_hardware_fails_before_any_worker_prepares() -> None:
    """Missing required hardware fails startup before any preparation occurs.

    Include a synthetic source before the hardware source to verify that even
    unrelated preparation has not begun when the missing collection is detected.

    The pipeline must record the startup failure and report that same exception
    when waited on, without invoking source cleanup for unprepared workers.
    """
    synthetic = NumberSource()
    live = DeviceSource(name="live")
    pipeline = Pipeline(synthetic, live)

    try:
        # expect RuntimeError(f"Hardware collection required by source(s): {names}.")
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
    source = DeviceSource(device_name="missing")
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
    source = DeviceSource()
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
    source = DeviceSource(fail_work=True)
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
            source = DeviceSource()
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
