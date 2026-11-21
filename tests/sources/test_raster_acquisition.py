"""Exercise raster delivery independently of frame planning and scanner hardware."""

from collections.abc import Iterator
from typing import cast
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
from dirigo.pipeline.product import Product
from dirigo.records import RecordBatch
from dirigo.sources.raster_acquisition import RasterAcquisition
from dirigo.system import System


class StubSync:
    """Supply the lifecycle operations used by shared raster execution."""

    def __init__(self, events: list[str]) -> None:
        """Keep operation order and an optional cleanup failure."""
        self.events = events
        self.close_failure: BaseException | None = None

    def start(self) -> None:
        """Record startup without managing capture data."""
        self.events.append("sync.start")

    def close(self) -> None:
        """Record closure and surface the selected failure."""
        self.events.append("sync.close")
        if self.close_failure is not None:
            raise self.close_failure


class PreparedRaster(RasterAcquisition):
    """Attach prepared test sessions without a frame plan or scanner devices."""

    def __init__(
        self,
        capture: Mock,
        sync: StubSync,
        events: list[str],
        *,
        system: System,
    ) -> None:
        """Keep test participants passive until preparation."""
        super().__init__(system=system)
        self.test_capture = capture
        self.test_sync = sync
        self.events = events
        self.scanner_close_failure: BaseException | None = None

    def _prepare(self) -> None:
        """Attach the fixture's already described capture."""
        self._capture = cast(DigitizerCaptureSession, self.test_capture)
        self._sync = self.test_sync
        self._sample_code_range = units.IntRange(-32768, 32767)
        self._capture_layout = self.test_capture.layout
        self._prepared = True

    def _close_scanners(self) -> None:
        """Exercise the cleanup hook without requiring real scanner commands."""
        self.events.append("scanners.close")
        if self.scanner_close_failure is not None:
            raise self.scanner_close_failure


@pytest.fixture
def rig() -> Iterator[tuple[PreparedRaster, Mock, StubSync, list[str]]]:
    """Describe three batches containing two records each."""
    events: list[str] = []
    capture = Mock(spec=DigitizerCaptureSession)
    capture.settings = CaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=4,
        records_per_batch=2,
        batch_count=3,
    )
    capture.layout = CaptureLayout(
        shape=(2, 4, 1),
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
        """Fill one batch and reject an unnecessary read after the final batch."""
        nonlocal delivered
        assert delivered < 3
        delivered += 1
        destination.fill(delivered)
        if delivered == 3:
            capture.state = CaptureState.COMPLETED
        return CaptureInfo()

    capture.read_into.side_effect = read_into
    capture.close.side_effect = lambda: events.append("capture.close")
    sync = StubSync(events)
    system = System({})
    source = PreparedRaster(capture, sync, events, system=system)
    try:
        yield source, capture, sync, events
    finally:
        source._close()
        system.close()


@pytest.mark.parametrize("records_per_frame", [None, 6])
def test_shared_delivery_is_independent_of_frame_batching(
    rig: tuple[PreparedRaster, Mock, StubSync, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    records_per_frame: int | None,
) -> None:
    """Deliver ungrouped records or one frame spread across three batches."""
    source, capture, _, _ = rig
    source._records_per_frame = records_per_frame
    published: list[Product[RecordBatch]] = []
    monkeypatch.setattr(source, "_publish", published.append)

    source._prepare()
    source._work()

    assert capture.read_into.call_count == 3
    assert [p.metadata.sequence_index for p in published] == [0, 1, 2]
    assert [p.payload.first_record for p in published] == [0, 2, 4]
    assert [p.payload.records_per_frame for p in published] == [records_per_frame] * 3
    assert len({id(p.payload.samples) for p in published}) == 3
    for index, product in enumerate(published, start=1):
        assert not product.payload.samples.flags.writeable
        assert np.all(product.payload.samples == index)


def test_shared_cleanup_attempts_every_participant_once(
    rig: tuple[PreparedRaster, Mock, StubSync, list[str]],
) -> None:
    """Preserve all cleanup failures and avoid repeating failed operations."""
    source, capture, sync, events = rig
    source._prepare()
    sync_failure = RuntimeError("sync close failed")
    capture_failure = RuntimeError("capture close failed")
    scanner_failure = RuntimeError("scanner close failed")
    sync.close_failure = sync_failure
    source.scanner_close_failure = scanner_failure

    def close_capture() -> None:
        """Fail after recording the capture cleanup attempt."""
        events.append("capture.close")
        raise capture_failure

    capture.close.side_effect = close_capture

    with pytest.raises(BaseExceptionGroup) as caught:
        source._close()

    assert caught.value.exceptions == (
        sync_failure,
        capture_failure,
        scanner_failure,
    )
    assert events == ["sync.close", "capture.close", "scanners.close"]

    source._close()

    assert events == ["sync.close", "capture.close", "scanners.close"]
    capture.close.assert_called_once_with()
