"""Test core sequence selection and worker preparation without NI imports."""

from collections.abc import Iterator
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from threading import get_ident
from typing import cast
from unittest.mock import Mock

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware import acquisition as acquisition_module
from dirigo.hardware.acquisition import (
    FrameSequenceFactory,
    discover_frame_sequences,
    load_frame_sequence_factory,
)
from dirigo.hardware.capture import CaptureInfo, CaptureLayout, CaptureSettings
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.hardware import Hardware
from dirigo.pipeline.pipeline import Pipeline
from dirigo.sources.frame_acquisition import FrameAcquisitionWorker


@dataclass(frozen=True)
class DemoConfig:
    """Declare one inspectable factory parameter."""

    label: str = "demo"


@pytest.fixture
def hardware() -> Iterator[Hardware]:
    """Supply an empty collection; mocked sequences need no devices."""
    result = Hardware({})
    try:
        yield result
    finally:
        result.close()


def _worker(factory: Mock) -> FrameAcquisitionWorker:
    """Explicitly select a mocked factory for one complete record."""
    return FrameAcquisitionWorker(
        RecordFramePlan(
            capture_settings=CaptureSettings(
                sample_rate=units.SampleRate("1 MS/s"),
                record_length=1,
                records_per_batch=1,
                batch_count=1,
            ),
            records_per_frame=1,
        ),
        sequence="demo",
        waveforms={},
        sequence_config={"label": "selected"},
        factories={"demo": cast(FrameSequenceFactory[DemoConfig], factory)},
    )


def test_prepare_precedes_execution_and_runs_on_pipeline_caller(
    hardware: Hardware,
) -> None:
    """Prepare before execution and parse the selected configuration.

    The mocked sequence emits one batch and coordinated EOF. Preparation
    occurs on the Pipeline.start() caller; start and normal close occur on
    the acquisition thread. NI integration tests cover actual batch data.
    """
    caller = get_ident()
    events: list[tuple[str, int]] = []
    sequence = Mock()
    sequence.layout = CaptureLayout(
        shape=(1, 1, 1),
        channel_indices=(0,),
        dtype=np.dtype(np.int16),
    )
    sequence.prepare.side_effect = lambda: events.append(("prepare", get_ident()))
    sequence.start.side_effect = lambda: events.append(("start", get_ident()))
    sequence.close.side_effect = lambda: events.append(("close", get_ident()))
    sequence.read_into.side_effect = [CaptureInfo(), None]

    factory = Mock(config_type=DemoConfig, return_value=sequence)
    pipeline = Pipeline(_worker(factory), hardware=hardware)
    factory.assert_not_called()

    try:
        pipeline.start()
        pipeline.wait(timeout=2)
    finally:
        if not pipeline.done:
            pipeline.abort()
            pipeline.wait(timeout=2)

    assert factory.call_args.args[0] == DemoConfig(label="selected")
    assert [name for name, _ in events] == ["prepare", "start", "close"]
    assert events[0][1] == caller
    assert events[1][1] == events[2][1] != caller


def test_failed_prepare_is_closed_before_pipeline_start_raises(
    hardware: Hardware,
) -> None:
    """Rollback a returned sequence whose preparation fails, without starting.

    The sequence deliberately does not clean itself up. Pipeline rollback
    must invoke the worker's close hook and preserve the original exception.
    """
    failure = RuntimeError("Preparation failed.")
    sequence = Mock()
    sequence.prepare.side_effect = failure
    factory = Mock(config_type=DemoConfig, return_value=sequence)
    pipeline = Pipeline(_worker(factory), hardware=hardware)

    with pytest.raises(RuntimeError) as caught:
        pipeline.start()

    assert caught.value is failure
    sequence.start.assert_not_called()
    sequence.close.assert_called_once_with()
    assert pipeline.done


def test_discovery_is_lazy_and_loads_only_selected_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """List registrations without importing their implementations."""
    entries = importlib_metadata.EntryPoints(
        [
            importlib_metadata.EntryPoint(
                name="demo",
                value="demo:factory",
                group="dirigo.acquisition_sequences",
            ),
            importlib_metadata.EntryPoint(
                name="unused",
                value="missing_sdk:factory",
                group="dirigo.acquisition_sequences",
            ),
        ]
    )
    loaded: list[str] = []
    factory = Mock(config_type=DemoConfig)

    def load(entry: importlib_metadata.EntryPoint) -> object:
        """Resolve only the factory explicitly selected by the test."""
        loaded.append(entry.name)
        assert entry.name == "demo"
        return factory

    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: entries)
    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", load)

    assert set(discover_frame_sequences()) == {"demo", "unused"}
    assert loaded == []
    assert load_frame_sequence_factory("demo") is factory
    assert loaded == ["demo"]


def test_explicit_registry_including_empty_bypasses_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exactly the supplied registry, including an empty registry."""

    def forbidden() -> dict[str, importlib_metadata.EntryPoint]:
        """Fail if registry selection accidentally consults installed plugins."""
        raise AssertionError("Must not discover plugins.")

    monkeypatch.setattr(
        acquisition_module,
        "discover_frame_sequences",
        forbidden,
    )
    factory = Mock(config_type=DemoConfig)

    assert load_frame_sequence_factory("demo", factories={"demo": factory}) is factory
    with pytest.raises(ValueError, match="Unknown acquisition sequence"):
        load_frame_sequence_factory("demo", factories={})


def test_duplicate_registrations_identify_both_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ambiguous selection before either implementation is imported."""
    entries = importlib_metadata.EntryPoints(
        [
            importlib_metadata.EntryPoint(
                name="demo",
                value="first:factory",
                group="dirigo.acquisition_sequences",
            ),
            importlib_metadata.EntryPoint(
                name="demo",
                value="second:factory",
                group="dirigo.acquisition_sequences",
            ),
        ]
    )
    monkeypatch.setattr(importlib_metadata, "entry_points", lambda: entries)

    with pytest.raises(ValueError, match="demo") as caught:
        discover_frame_sequences()

    assert "first:factory" in str(caught.value)
    assert "second:factory" in str(caught.value)
