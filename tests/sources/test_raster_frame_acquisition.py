"""Test specification-driven acquisition with mocked hardware participants."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import cast
from unittest.mock import Mock, PropertyMock

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.raster_frame_sync import (
    RasterFrameSync,
    RasterFrameSyncSession,
)
from dirigo.hardware.scanner import (
    GalvoScanner,
    ResonantScanner,
    ResonantScannerConfig,
)
from dirigo.hardware.timing import DigitalRecordTrigger
from dirigo.optics import LaserScanningOptics
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.pipeline import Pipeline
from dirigo.pipeline.sink import Sink
from dirigo.raster import RasterFrameAcquisitionSpec
from dirigo.records import RecordBatch
from dirigo.resonant_raster import ResonantRasterSettings
from dirigo.sources import raster_frame_acquisition as acquisition_module
from dirigo.sources.raster_frame_acquisition import RasterFrameAcquisition
from dirigo.system import System


@dataclass
class Participants:
    """Expose the configured system and mocked acquisition participants."""

    system: System
    capture: Mock
    sync: Mock
    backend: Mock
    fast: Mock
    slow: Mock
    amplitudes: list[units.Angle]
    events: list[str]


def _spec() -> RasterFrameAcquisitionSpec:
    """Request three exactly representable bidirectional frames."""
    return RasterFrameAcquisitionSpec(
        pixel_width=units.Length("1 um"),
        pixel_height=units.Length("1 um"),
        frame_width=units.Length("128 um"),
        frame_height=units.Length("64 um"),
        bidirectional=True,
        frames_per_acquisition=3,
    )


@pytest.fixture
def participants(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Participants]:
    """Supply passive mocks with preparation-dependent capture layout."""
    monkeypatch.setattr(acquisition_module, "monotonic", lambda: 0.0)

    events: list[str] = []
    amplitudes: list[units.Angle] = []

    fast = Mock(spec=ResonantScanner)
    fast.config = ResonantScannerConfig(
        nominal_frequency=units.Frequency("1 kHz"),
        max_amplitude=units.Angle("20 deg"),
        settling_time=units.Time("0 s"),
        frequency_tolerance=0.0,
    )
    monkeypatch.setattr(
        type(fast),
        "amplitude",
        PropertyMock(side_effect=lambda value: amplitudes.append(value)),
        raising=False,
    )
    slow = Mock(spec=GalvoScanner)

    capture = Mock(spec=DigitizerCaptureSession)
    capture.state = CaptureState.NEW
    capture.requested_sample_rate = units.SampleRate("1 MS/s")
    capture.resolve_record_length_constraints.return_value = (64, 16)

    def configure(
        *,
        record_length: int,
        records_per_batch: int,
        batch_count: int | None,
    ) -> None:
        capture.settings = CaptureSettings(
            sample_rate=capture.requested_sample_rate,
            record_length=record_length,
            records_per_batch=records_per_batch,
            batch_count=batch_count,
            record_trigger=DigitalRecordTrigger(source="external"),
        )

    def prepare_capture() -> None:
        settings = capture.settings
        capture.layout = CaptureLayout(
            shape=(
                settings.records_per_batch,
                settings.record_length,
                1,
            ),
            channel_indices=(0,),
            dtype=np.dtype(np.int16),
        )
        capture.state = CaptureState.PREPARED

    def start_capture() -> None:
        capture.state = CaptureState.STARTED

    def stop_capture() -> None:
        if capture.state is not CaptureState.CLOSED:
            capture.state = CaptureState.STOPPED

    def close_capture() -> None:
        events.append("capture.close")
        capture.state = CaptureState.CLOSED

    delivered = 0

    def read(
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        nonlocal delivered
        delivered += 1
        assert delivered <= 3, "Worker read beyond finite delivery."
        destination.fill(delivered)

        if delivered == 3:
            capture.state = CaptureState.COMPLETED

        return CaptureInfo()

    capture.configure.side_effect = configure
    capture.prepare.side_effect = prepare_capture
    capture.start.side_effect = start_capture
    capture.stop.side_effect = stop_capture
    capture.close.side_effect = close_capture
    capture.read_into.side_effect = read

    sync = Mock(spec=RasterFrameSyncSession)
    sync.max_record_rate = units.SampleRate("1 kS/s")
    sync.prepare.side_effect = capture.prepare
    sync.start.side_effect = capture.start

    def close_sync() -> None:
        events.append("sync.close")
        capture.stop()

    sync.close.side_effect = close_sync

    backend = Mock(spec=RasterFrameSync)
    backend.create_session.return_value = sync

    digitizer = Mock(spec=Digitizer)
    digitizer.create_capture.return_value = capture

    settings = ResonantRasterSettings(
        line_fill_fraction=0.8,
        trigger_phase=units.Angle("0 deg"),
        capture_rearm_time=units.Time("10 us"),
        slow_sample_rate=units.SampleRate("100 kS/s"),
        slow_flyback_time=units.Time("2 ms"),
        output_rearm_time=units.Time("100 us"),
        slow_settling_time=units.Time("0 s"),
    )
    system = System(
        {
            "fast_scanner": cast(ResonantScanner, fast),
            "slow_scanner": cast(GalvoScanner, slow),
            "digitizer": cast(Digitizer, digitizer),
        },
        syncs={
            "raster_frame_sync": cast(RasterFrameSync, backend),
        },
        optics=LaserScanningOptics(
            objective_focal_length=units.Length("10 mm"),
            relay_magnification=2.0,
        ),
        resonant_raster=settings,
    )

    try:
        yield Participants(
            system,
            capture,
            sync,
            backend,
            fast,
            slow,
            amplitudes,
            events,
        )
    finally:
        system.close()


class Collector(Sink[RecordBatch]):
    """Retain independently owned batches after receive contexts exit."""

    def __init__(self, upstream: RasterFrameAcquisition) -> None:
        super().__init__(upstream=upstream)
        self.batches: list[RecordBatch] = []
        self.indices: list[int | None] = []

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        while True:
            try:
                with self._receive() as product:
                    self.batches.append(product.payload)
                    self.indices.append(product.metadata.sequence_index)
            except EndOfStream:
                return

    def _close(self) -> None:
        pass


def _run(pipeline: Pipeline) -> None:
    """Join successful or failed execution and cancel an unfinished graph."""
    pipeline.start()

    try:
        pipeline.wait(timeout=2)
    finally:
        if not pipeline.done:
            pipeline.abort()
            pipeline.wait(timeout=2)


def test_construction_is_passive(
    participants: Participants,
) -> None:
    """Retain dependencies without resolving or preparing hardware."""
    p = participants
    source = RasterFrameAcquisition(_spec(), system=p.system)

    p.backend.create_session.assert_not_called()
    p.capture.resolve_record_length_constraints.assert_not_called()
    assert p.amplitudes == []

    with pytest.raises(RuntimeError, match="successful preparation"):
        _ = source.plan


def test_pipeline_reads_capture_and_retains_independent_batches(
    participants: Participants,
) -> None:
    """Retry capture timeouts and publish exactly the requested frame count."""
    p = participants
    deliver = p.capture.read_into.side_effect
    assert callable(deliver)
    first = True

    def read(
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        nonlocal first

        if first:
            first = False
            raise TimeoutError("Retry this read.")

        return deliver(destination, timeout=timeout)

    p.capture.read_into.side_effect = read

    source = RasterFrameAcquisition(_spec(), system=p.system)
    collector = Collector(source)
    _run(Pipeline(source, collector))

    plan = source.plan
    assert len(collector.batches) == 3
    assert collector.indices == [0, 1, 2]
    assert p.capture.read_into.call_count == 4

    assert [batch.first_record for batch in collector.batches] == [
        0,
        plan.records_per_frame,
        2 * plan.records_per_frame,
    ]

    assert len({id(batch.samples) for batch in collector.batches}) == 3

    for index, batch in enumerate(collector.batches, start=1):
        assert not batch.samples.flags.writeable
        np.testing.assert_array_equal(
            batch.samples,
            np.full(batch.layout.shape, index),
        )

    p.backend.create_session.assert_called_once_with(
        fast_scanner=p.fast,
        slow_scanner=p.slow,
        capture=p.capture,
    )
    p.sync.start.assert_called_once_with()
    p.sync.close.assert_called_once_with()
    p.capture.close.assert_called_once_with()

    assert p.events.index("sync.close") < p.events.index("capture.close")
    assert p.amplitudes == [plan.fast_amplitude, units.Angle(0)]

    # Pipeline cleanup leaves the borrowed system open.
    assert p.system.require_optics().relay_magnification == 2.0


def test_failed_preparation_closes_both_sessions(
    participants: Participants,
) -> None:
    """Release owned sessions and reset amplitude after partial preparation."""
    p = participants
    failure = RuntimeError("Synchronization preparation failed.")
    p.sync.prepare.side_effect = failure

    source = RasterFrameAcquisition(_spec(), system=p.system)
    pipeline = Pipeline(source)

    with pytest.raises(RuntimeError) as caught:
        pipeline.start()

    assert caught.value is failure
    assert pipeline.done
    p.sync.start.assert_not_called()
    p.sync.close.assert_called_once_with()
    p.capture.close.assert_called_once_with()
    assert p.amplitudes[-1] == units.Angle(0)


def test_settling_can_be_cancelled_before_start(
    participants: Participants,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wait on the worker's cancellation event before starting acquisition."""
    p = participants
    settings = replace(
        p.system.require_resonant_raster(),
        slow_settling_time=units.Time("200 ms"),
    )
    monkeypatch.setattr(
        p.system,
        "require_resonant_raster",
        lambda: settings,
    )

    source = RasterFrameAcquisition(_spec(), system=p.system)
    waits: list[float | None] = []

    def cancel_wait(timeout: float | None = None) -> bool:
        waits.append(timeout)
        source._abort_requested.set()
        return True

    monkeypatch.setattr(
        source._abort_requested,
        "wait",
        cancel_wait,
    )

    collector = Collector(source)
    _run(Pipeline(source, collector))

    assert waits == [pytest.approx(0.2)]
    assert collector.batches == []
    p.sync.start.assert_not_called()
    p.capture.read_into.assert_not_called()
    p.capture.close.assert_called_once_with()


def test_closed_capture_timeout_is_fatal(
    participants: Participants,
) -> None:
    """Do not retry a timeout after the capture session has closed."""
    p = participants
    failure = TimeoutError("Fatal capture timeout.")

    def fail(
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        p.capture.state = CaptureState.CLOSED
        raise failure

    p.capture.read_into.side_effect = fail

    source = RasterFrameAcquisition(_spec(), system=p.system)
    pipeline = Pipeline(source)

    with pytest.raises(TimeoutError) as caught:
        _run(pipeline)

    assert caught.value is failure
    p.capture.read_into.assert_called_once()
    p.sync.close.assert_called_once_with()
    p.capture.close.assert_called_once_with()


def test_premature_capture_eof_is_rejected(
    participants: Participants,
) -> None:
    """Require the requested number of successfully acquired frames."""
    p = participants
    p.capture.read_into.side_effect = None
    p.capture.read_into.return_value = None

    source = RasterFrameAcquisition(_spec(), system=p.system)

    with pytest.raises(RuntimeError, match="before the requested frame count"):
        _run(Pipeline(source))

    p.sync.close.assert_called_once_with()
    p.capture.close.assert_called_once_with()


def test_sync_cleanup_failure_does_not_skip_capture_or_scanner_cleanup(
    participants: Participants,
) -> None:
    """Attempt every owned cleanup operation after a synchronization error."""
    p = participants
    p.sync.close.side_effect = RuntimeError("Synchronization cleanup failed.")

    source = RasterFrameAcquisition(_spec(), system=p.system)

    with pytest.raises(BaseExceptionGroup):
        _run(Pipeline(source))

    p.capture.close.assert_called_once_with()
    assert p.amplitudes[-1] == units.Angle(0)
