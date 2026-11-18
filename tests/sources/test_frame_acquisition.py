"""Test sequence selection and explicit acquisition-system dependencies."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
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
from dirigo.pipeline.pipeline import Pipeline
from dirigo.sources.frame_acquisition import (
    FrameAcquisitionWorker,
    RasterFrameAcquisitionSpec,
)
from dirigo.system import System


@dataclass(frozen=True)
class DemoConfig:
    """Declare one inspectable factory parameter."""

    label: str = "demo"


@pytest.fixture
def system() -> Iterator[System]:
    """Supply an empty system; mocked sequences need no devices."""
    result = System({})

    try:
        yield result
    finally:
        result.close()


def _plan() -> RecordFramePlan:
    """Request one complete frame containing one record."""
    return RecordFramePlan(
        capture_settings=CaptureSettings(
            sample_rate=units.SampleRate("1 MS/s"),
            record_length=1,
            records_per_batch=1,
            batch_count=1,
        ),
        records_per_frame=1,
    )


def _worker(
    factory: Mock,
    *,
    system: System,
) -> FrameAcquisitionWorker:
    """Construct a worker with an explicit system and factory registry."""
    return FrameAcquisitionWorker(
        _plan(),
        system=system,
        sequence="demo",
        waveforms={},
        sequence_config={"label": "selected"},
        factories={
            "demo": cast(FrameSequenceFactory[DemoConfig], factory),
        },
    )


def test_constructor_rejects_invalid_system() -> None:
    """Reject an invalid dependency before invoking a sequence factory."""
    factory = Mock(config_type=DemoConfig)

    with pytest.raises(TypeError, match="system must be a System"):
        _worker(factory, system=cast(System, object()))

    factory.assert_not_called()


def test_prepare_precedes_execution_and_runs_on_pipeline_caller(
    system: System,
) -> None:
    """Forward the explicit system and prepare before worker execution.

    Preparation runs on the Pipeline.start() caller. Sequence start and
    normal closure run on the acquisition thread. The sequence supplies
    one batch followed by EOF.
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
    worker = _worker(factory, system=system)
    pipeline = Pipeline(worker)

    factory.assert_not_called()
    sequence.prepare.assert_not_called()

    try:
        pipeline.start()
        pipeline.wait(timeout=2)
    finally:
        if not pipeline.done:
            pipeline.abort()
            pipeline.wait(timeout=2)

    factory.assert_called_once_with(
        DemoConfig(label="selected"),
        system=system,
        plan=_plan(),
        waveforms={},
    )
    assert factory.call_args.kwargs["system"] is system

    assert [name for name, _ in events] == [
        "prepare",
        "start",
        "close",
    ]
    assert events[0][1] == caller
    assert events[1][1] == events[2][1] != caller

    assert sequence.read_into.call_count == 2
    sequence.close.assert_called_once_with()


def test_failed_prepare_is_closed_before_pipeline_start_raises(
    system: System,
) -> None:
    """Close a returned sequence whose preparation fails.

    The sequence deliberately does not clean itself up. Pipeline rollback
    must invoke the worker's close hook and preserve the original exception.
    """
    failure = RuntimeError("Preparation failed.")
    sequence = Mock()
    sequence.prepare.side_effect = failure

    factory = Mock(config_type=DemoConfig, return_value=sequence)
    worker = _worker(factory, system=system)
    pipeline = Pipeline(worker)

    with pytest.raises(RuntimeError) as caught:
        pipeline.start()

    assert caught.value is failure
    sequence.start.assert_not_called()
    sequence.read_into.assert_not_called()
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
        """Resolve only the explicitly selected factory."""
        loaded.append(entry.name)
        assert entry.name == "demo"
        return factory

    monkeypatch.setattr(
        importlib_metadata,
        "entry_points",
        lambda: entries,
    )
    monkeypatch.setattr(
        importlib_metadata.EntryPoint,
        "load",
        load,
    )

    assert set(discover_frame_sequences()) == {"demo", "unused"}
    assert loaded == []

    assert load_frame_sequence_factory("demo") is factory
    assert loaded == ["demo"]


def test_explicit_registry_including_empty_bypasses_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exactly the supplied registry, including an empty registry."""

    def forbidden() -> dict[str, importlib_metadata.EntryPoint]:
        """Fail if selection consults installed plugins."""
        raise AssertionError("Must not discover plugins.")

    monkeypatch.setattr(
        acquisition_module,
        "discover_frame_sequences",
        forbidden,
    )
    factory = Mock(config_type=DemoConfig)

    assert (
        load_frame_sequence_factory(
            "demo",
            factories={"demo": factory},
        )
        is factory
    )

    with pytest.raises(ValueError, match="Unknown acquisition sequence"):
        load_frame_sequence_factory("demo", factories={})


def test_duplicate_registrations_identify_both_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject ambiguous selection before loading either implementation."""
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
    monkeypatch.setattr(
        importlib_metadata,
        "entry_points",
        lambda: entries,
    )

    with pytest.raises(ValueError, match="demo") as caught:
        discover_frame_sequences()

    assert "first:factory" in str(caught.value)
    assert "second:factory" in str(caught.value)


@pytest.fixture
def valid_spec() -> RasterFrameAcquisitionSpec:
    return RasterFrameAcquisitionSpec(
        pixel_width=units.Length("1 um"),
        pixel_height=units.Length("1 um"),
        frame_width=units.Length("1 mm"),
        frame_height=units.Length("1 mm"),
        bidirectional=False,
        frames_per_acquisition=10,
    )


def test_valid_spec_is_accepted(valid_spec: RasterFrameAcquisitionSpec) -> None:
    assert valid_spec.bidirectional is False


def test_frames_per_acquisition_can_be_none(
    valid_spec: RasterFrameAcquisitionSpec,
) -> None:
    new_spec = replace(valid_spec, frames_per_acquisition=None)
    assert new_spec.frames_per_acquisition is None


@pytest.mark.parametrize(
    "field",
    ["pixel_width", "pixel_height", "frame_width", "frame_height"],
)
def test_length_fields_require_length(
    valid_spec: RasterFrameAcquisitionSpec, field: str
) -> None:
    with pytest.raises(TypeError, match=f"{field} must be a Length"):
        replace(valid_spec, **{field: object()})


def test_bidirectional_requires_bool(valid_spec: RasterFrameAcquisitionSpec) -> None:
    with pytest.raises(TypeError, match="bidirectional must be a bool"):
        replace(valid_spec, bidirectional=1)


def test_frame_per_acquisition_requires_int(
    valid_spec: RasterFrameAcquisitionSpec,
) -> None:
    with pytest.raises(TypeError, match="frames_per_acquisition must be an int"):
        replace(valid_spec, frames_per_acquisition=3.14)


@pytest.mark.parametrize(
    "field",
    ["pixel_width", "pixel_height", "frame_width", "frame_height"],
)
@pytest.mark.parametrize("value", [units.Length(0), units.Length("-1 um")])
def test_dimensions_must_be_positive(
    valid_spec: RasterFrameAcquisitionSpec, field: str, value: units.Length
) -> None:
    with pytest.raises(ValueError, match=f"{field} must be positive"):
        replace(valid_spec, **{field: value})


def test_frames_per_acquisition_must_be_positive(
    valid_spec: RasterFrameAcquisitionSpec,
) -> None:
    with pytest.raises(ValueError, match="frames_per_acquisition must be positive"):
        replace(valid_spec, frames_per_acquisition=-1)
