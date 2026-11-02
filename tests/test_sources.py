from typing import Any, Never, cast

import pytest

from dirigo.hardware import Device, Hardware
from dirigo.pipeline import Pipeline
from dirigo.source import HardwareSource, Source
from dirigo.worker import Worker


class FakeDevice(Device):
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class NumberSource(Source[int]):
    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.prepared = False

    def _prepare(self) -> None:
        self.prepared = True

    def _work(self) -> None:
        self._publish(36)

    def _close(self) -> None:
        pass


class DeviceSource(HardwareSource[int]):
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

        self._publish(7)

    def _close(self) -> None:
        self.closed = True


class Collector(Worker[int, Never]):
    def __init__(self, upstream: Worker[Any, int]) -> None:
        super().__init__(upstream=upstream)
        self.values: list[int] = []

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        with self._receive(timeout=1) as product:
            self.values.append(product.payload)

    def _close(self) -> None:
        pass


def test_source_runs_without_hardware() -> None:
    source = NumberSource(name="generated")
    sink = Collector(source)
    pipeline = Pipeline(sink)

    pipeline.start()
    pipeline.wait(timeout=2)

    assert source.upstream is None
    assert source.name == "generated"
    assert sink.values == [36]
    assert pipeline.done


def test_sources_do_not_accept_upstream() -> None:
    for source_type in (NumberSource, DeviceSource):
        untyped_source = cast(Any, source_type)

        with pytest.raises(TypeError, match="upstream"):
            untyped_source(upstream=NumberSource())


def test_hardware_access_before_binding_raises() -> None:
    source = DeviceSource()

    with pytest.raises(RuntimeError, match="bound"):
        _ = source._hardware


def test_binding_is_allowed_once() -> None:
    first = Hardware({})
    source = DeviceSource()
    source._bind_hardware(first)

    with pytest.raises(RuntimeError, match="bound"):
        source._bind_hardware(Hardware({}))

    assert source._hardware is first


def test_pipeline_rejects_invalid_hardware() -> None:
    invalid_hardware = cast(Any, object())

    with pytest.raises(TypeError, match="hardware"):
        Pipeline(NumberSource(), hardware=invalid_hardware)


def test_pipeline_binds_upstream_source_before_preparation() -> None:
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceSource()
    sink = Collector(source)
    pipeline = Pipeline(sink, hardware=hardware)

    pipeline.start()
    pipeline.wait(timeout=2)

    assert source.hardware_seen is hardware
    assert source.worked
    assert source.closed
    assert sink.values == [7]
    assert device.close_calls == 0
    assert hardware.require("digitizer", FakeDevice) is device


def test_all_hardware_sources_are_bound_before_any_preparation() -> None:
    hardware = Hardware({"digitizer": FakeDevice()})
    second = DeviceSource(name="second")

    class FirstSource(DeviceSource):
        def _prepare(self) -> None:
            assert second._hardware is hardware
            super()._prepare()

    first = FirstSource(name="first")
    pipeline = Pipeline(first, second, hardware=hardware)

    pipeline.start()
    pipeline.wait(timeout=2)

    assert first.hardware_seen is hardware
    assert second.hardware_seen is hardware


def test_missing_hardware_fails_before_any_worker_prepares() -> None:
    synthetic = NumberSource()
    live = DeviceSource(name="live")
    pipeline = Pipeline(synthetic, live)

    with pytest.raises(RuntimeError, match="live") as caught:
        pipeline.start()

    assert not synthetic.prepared
    assert not live.prepared
    assert not live.worked
    assert not live.closed
    assert pipeline.done
    assert pipeline.exception is caught.value

    with pytest.raises(RuntimeError, match="live"):
        pipeline.wait(timeout=2)


def test_device_lookup_failure_cleans_source_but_preserves_hardware() -> None:
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceSource(device_name="missing")
    pipeline = Pipeline(source, hardware=hardware)

    with pytest.raises(KeyError, match="missing"):
        pipeline.start()

    assert source.closed
    assert not source.worked
    assert pipeline.done
    assert device.close_calls == 0
    assert hardware.require("digitizer", FakeDevice) is device


def test_downstream_preparation_failure_preserves_hardware() -> None:
    class FailingCollector(Collector):
        def _prepare(self) -> None:
            raise RuntimeError("Downstream preparation failed.")

    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceSource()
    pipeline = Pipeline(FailingCollector(source), hardware=hardware)

    with pytest.raises(RuntimeError, match="Downstream"):
        pipeline.start()

    assert source.closed
    assert not source.worked
    assert device.close_calls == 0
    assert hardware.require("digitizer", FakeDevice) is device


def test_execution_failure_preserves_hardware() -> None:
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})
    source = DeviceSource(fail_work=True)
    pipeline = Pipeline(source, hardware=hardware)
    pipeline.start()

    with pytest.raises(RuntimeError, match="Acquisition"):
        pipeline.wait(timeout=2)

    assert source.closed
    assert pipeline.done
    assert device.close_calls == 0
    assert hardware.require("digitizer", FakeDevice) is device


def test_hardware_can_be_reused_across_successive_pipelines() -> None:
    device = FakeDevice()
    hardware = Hardware({"digitizer": device})

    for _ in range(2):
        source = DeviceSource()
        sink = Collector(source)
        pipeline = Pipeline(sink, hardware=hardware)

        pipeline.start()
        pipeline.wait(timeout=2)

        assert sink.values == [7]

    assert device.close_calls == 0

    hardware.close()

    assert device.close_calls == 1
