"""Exercise finite line acquisition through a pipeline with fake hardware."""

from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast
from unittest.mock import Mock

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
from dirigo.hardware.capture_sync import CaptureOnlySyncSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.scanner import ResonantScanner, ResonantScannerConfig
from dirigo.hardware.timing import DigitalRecordTrigger
from dirigo.optics import LaserScanningOptics
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.pipeline import Pipeline
from dirigo.pipeline.sink import Sink
from dirigo.pipeline.worker import Worker
from dirigo.processors._encoding import _encode_int16
from dirigo.processors.line import LineProcessor
from dirigo.raster import RasterLineAcquisitionSpec
from dirigo.resonant_raster import ResonantScanSettings
from dirigo.sources.raster_line_acquisition import RasterLineAcquisition
from dirigo.system import System


class StubResonantScanner(ResonantScanner):
    """Record amplitude commands without importing a vendor driver."""

    def __init__(self) -> None:
        """Describe a scanner that requires no settling delay."""
        super().__init__(
            ResonantScannerConfig(
                nominal_frequency=units.Frequency("1 kHz"),
                max_amplitude=units.Angle("20 deg"),
                settling_time=units.Time(0),
                frequency_tolerance=0.0,
            )
        )
        self.commands: list[units.Angle] = []

    def _write_amplitude(self, value: units.Angle) -> None:
        """Retain each commanded optical amplitude."""
        self.commands.append(value)

    def _close(self) -> None:
        """Release no external resources."""
        return


class Collector[BatchT](Sink[BatchT]):
    """Retain independent batches after their receive contexts exit."""

    def __init__(self, upstream: Worker[Any, BatchT]) -> None:
        """Subscribe to the batches published by upstream."""
        super().__init__(upstream=upstream)
        self.batches: list[BatchT] = []
        self.indices: list[int | None] = []

    def _prepare(self) -> None:
        """Allocate no external resources."""
        return

    def _work(self) -> None:
        """Drain all published line batches."""
        while True:
            try:
                with self._receive() as product:
                    self.batches.append(product.payload)
                    self.indices.append(product.metadata.sequence_index)
            except EndOfStream:
                return

    def _close(self) -> None:
        """Retain observations after pipeline completion."""
        return


type LineRig = tuple[System, Mock, Mock, StubResonantScanner]


@pytest.fixture
def rig() -> Iterator[LineRig]:
    """Provide only a digitizer and fast scanner, with no synchronization backend."""
    capture = Mock(spec=DigitizerCaptureSession)
    capture.state = CaptureState.NEW
    capture.requested_sample_rate = units.SampleRate("1 MS/s")
    capture.resolve_record_length_constraints.return_value = (64, 16)

    def configure(
        *,
        record_length: int,
        records_per_batch: int,
        batch_count: int,
    ) -> None:
        """Resolve complete settings from the worker's deferred geometry."""
        capture.settings = CaptureSettings(
            sample_rate=units.SampleRate("1 MS/s"),
            record_length=record_length,
            records_per_batch=records_per_batch,
            batch_count=batch_count,
            record_trigger=DigitalRecordTrigger(source="external"),
        )

    def prepare() -> None:
        """Establish layout without starting acquisition."""
        settings = capture.settings
        capture.layout = CaptureLayout(
            shape=(settings.records_per_batch, settings.record_length, 1),
            channel_indices=(0,),
            dtype=np.dtype(np.int16),
        )
        capture.state = CaptureState.PREPARED

    delivered = 0

    def read_into(
        destination: SampleArray,
        *,
        timeout: float | None = None,
    ) -> CaptureInfo:
        """Fill exactly the requested finite batch count."""
        nonlocal delivered
        assert delivered < capture.settings.batch_count
        delivered += 1
        destination.fill(delivered)
        if delivered == capture.settings.batch_count:
            capture.state = CaptureState.COMPLETED
        return CaptureInfo()

    capture.configure.side_effect = configure
    capture.prepare.side_effect = prepare
    capture.read_into.side_effect = read_into
    digitizer = Mock(spec=Digitizer)
    digitizer.data_range = units.IntRange(-32768, 32767)
    digitizer.create_capture.return_value = capture
    fast = StubResonantScanner()
    system = System(
        {
            "fast_scanner": fast,
            "digitizer": cast(Digitizer, digitizer),
        },
        optics=LaserScanningOptics(
            objective_focal_length=units.Length("10 mm"),
            relay_magnification=2.0,
        ),
        resonant_scan=ResonantScanSettings(
            line_fill_fraction=0.8,
            trigger_phase=units.Angle(0),
            capture_rearm_time=units.Time("10 us"),
        ),
    )
    try:
        yield system, capture, digitizer, fast
    finally:
        system.close()


def _spec() -> RasterLineAcquisitionSpec:
    """Request three batches of four useful lines."""
    return RasterLineAcquisitionSpec(
        pixel_width=units.Length("1 um"),
        line_width=units.Length("128 um"),
        bidirectional=True,
        lines_per_acquisition=12,
        lines_per_batch=4,
    )


@pytest.mark.parametrize("bidirectional", [False, True])
def test_line_pipeline_needs_no_slow_scanner_or_sync_backend(
    rig: LineRig,
    bidirectional: bool,
) -> None:
    """Reuse shared execution and retain exact record and publication positions."""
    system, capture, digitizer, fast = rig
    source = RasterLineAcquisition(
        replace(_spec(), bidirectional=bidirectional),
        system=system,
    )
    digitizer.create_capture.assert_not_called()
    assert fast.commands == []

    collector = Collector(source)
    pipeline = Pipeline(source, collector)
    pipeline.start()
    try:
        pipeline.wait(timeout=2)
    finally:
        if not pipeline.done:
            pipeline.abort()
            pipeline.wait(timeout=2)

    plan = source.plan
    records_per_batch = 2 if bidirectional else 4
    assert plan.records_per_batch == records_per_batch
    assert collector.indices == [0, 1, 2]
    assert [b.first_record for b in collector.batches] == [
        0,
        records_per_batch,
        2 * records_per_batch,
    ]
    assert all(b.records_per_frame is None for b in collector.batches)
    assert len({id(b.samples) for b in collector.batches}) == 3
    for index, batch in enumerate(collector.batches, start=1):
        assert not batch.samples.flags.writeable
        assert np.all(batch.samples == index)
    assert capture.read_into.call_count == 3
    capture.stop.assert_called_once_with()
    capture.close.assert_called_once_with()
    assert fast.commands == [plan.fast_axis.fast_amplitude, units.Angle(0)]
    assert system.require_device("fast_scanner", ResonantScanner) is fast


def test_failed_line_preparation_releases_capture_and_turns_off_scanner(
    rig: LineRig,
) -> None:
    """Preserve preparation failure while cleaning partially prepared capture."""
    system, capture, _, fast = rig
    failure = RuntimeError("prepare failed")
    capture.prepare.side_effect = failure
    source = RasterLineAcquisition(_spec(), system=system)
    pipeline = Pipeline(source)

    with pytest.raises(RuntimeError) as caught:
        pipeline.start()

    assert caught.value is failure
    capture.start.assert_not_called()
    capture.stop.assert_called_once_with()
    capture.close.assert_called_once_with()
    assert len(fast.commands) == 2
    assert fast.commands[-1] == units.Angle(0)


@pytest.mark.parametrize("prepare", [False, True])
def test_capture_only_sync_leaves_capture_closure_to_its_owner(
    prepare: bool,
) -> None:
    """Stop only after attempted preparation and never close borrowed capture."""
    capture = Mock(spec=DigitizerCaptureSession)
    capture.state = CaptureState.NEW
    sync = CaptureOnlySyncSession(cast(DigitizerCaptureSession, capture))
    if prepare:
        sync.prepare()
        sync.start()

    sync.close()
    sync.close()

    assert capture.stop.call_count == int(prepare)
    capture.close.assert_not_called()
    capture.read_into.assert_not_called()


def test_abort_during_a_read_discards_the_unpublished_batch(rig: LineRig) -> None:
    """Close capture and scanner after cancellation inside a pending read."""
    system, capture, _, fast = rig
    source = RasterLineAcquisition(_spec(), system=system)
    collector = Collector(source)
    pipeline = Pipeline(source, collector)

    def cancel_read(
        destination: SampleArray,
        *,
        timeout: float | None = None,
    ) -> CaptureInfo:
        """Deliver cancellation before the worker can publish this batch."""
        pipeline.abort()
        return CaptureInfo()

    capture.read_into.side_effect = cancel_read
    pipeline.start()
    pipeline.wait(timeout=2)

    assert collector.batches == []
    capture.stop.assert_called_once_with()
    capture.close.assert_called_once_with()
    assert fast.commands[-1] == units.Angle(0)


@pytest.mark.parametrize("offset_binary", [False, True])
def test_line_processor_publishes_two_batches(
    rig: LineRig, offset_binary: bool
) -> None:
    """Encode two-channel records while preserving sweep order and metadata."""
    system, capture, digitizer, _ = rig
    if offset_binary:
        digitizer.data_range = units.IntRange(0, 4095)
        dtype = np.dtype(np.uint16)
        zero_code, gain = 2048, 16
    else:
        dtype = np.dtype(np.int16)
        zero_code, gain = 0, 1

    def prepare() -> None:
        settings = replace(
            capture.settings, channel_indices=(2, 0), timestamps_enabled=True
        )
        capture.settings = settings
        capture.layout = CaptureLayout(
            shape=(settings.records_per_batch, settings.record_length, 2),
            channel_indices=settings.channel_indices,
            dtype=dtype,
        )
        capture.state = CaptureState.PREPARED

    base = np.array([[-2, 10], [4, -20]])
    capture_infos = [CaptureInfo([0.0, 0.001]), CaptureInfo([0.002, 0.003])]
    delivered = 0

    def read_into(
        destination: SampleArray, *, timeout: float | None = None
    ) -> CaptureInfo:
        nonlocal delivered
        destination[...] = zero_code + base[:, None, :] + 100 * delivered
        # The first forward pixel averages an extra half code. The reverse
        # bins are in time order here and flip during reconstruction.
        destination[:, 50:150, :] += 1
        destination[:, 550:750, :] += 2
        destination[:, 750:950, :] += 3
        info = capture_infos[delivered]
        delivered += 1
        if delivered == len(capture_infos):
            capture.state = CaptureState.COMPLETED
        return info

    capture.prepare.side_effect = prepare
    capture.read_into.side_effect = read_into
    spec = replace(_spec(), line_width=units.Length("2 um"), lines_per_acquisition=8)
    source = RasterLineAcquisition(spec, system=system)
    processor = LineProcessor(upstream=source)
    collector = Collector(processor)
    pipeline = Pipeline(collector)
    pipeline.start()
    try:
        pipeline.wait(timeout=2)
    finally:
        if not pipeline.done:
            pipeline.abort()
            pipeline.wait(timeout=2)

    assert collector.indices == [0, 1]
    assert [batch.first_record for batch in collector.batches] == [0, 2]
    assert source.sample_code_range == digitizer.data_range
    assert collector.batches[0].pixels is not collector.batches[1].pixels

    pixel_additions = np.array([[gain // 2, 0], [3 * gain, 2 * gain]] * 2)
    for index, batch in enumerate(collector.batches):
        assert batch.pixel_width == units.Length("1 um")
        assert batch.lines_per_record == 2
        assert batch.channel_indices == (2, 0)
        assert batch.info is capture_infos[index]
        assert batch.pixels.dtype == np.dtype(np.int16)
        assert not batch.pixels.flags.writeable
        record_values = np.repeat((base + 100 * index) * gain, 2, axis=0)
        np.testing.assert_array_equal(
            batch.pixels, record_values[:, None, :] + pixel_additions[:, :, None]
        )


def test_pixel_encoding_rounds_and_saturates_before_cast() -> None:
    averages = np.array([-40_000.0, -0.5, 0.5, 40_000.0], dtype=np.float64)
    pixels = _encode_int16(averages, units.IntRange(-32768, 32767))
    np.testing.assert_array_equal(pixels, [-32768, 0, 0, 32767])
