from collections.abc import Generator
from dataclasses import replace
from threading import Event
from typing import Any

import numpy as np
import pytest

from dirigo.display import (
    ChannelSettings,
    ColorVector,
    Display,
    DisplayImage,
    DisplayPixelFormat,
    DisplaySettings,
    Linear,
)
from dirigo.frames import FrameArray, FrameMetadata
from dirigo.pipeline import EndOfStream, Pipeline, Product, Sink, Source, Worker
from dirigo.pipeline.product import Metadata
from dirigo.processors.display import DisplayProcessor
from dirigo.units import IntRange

_TIMEOUT = 10.0


class FrameSource(Source[FrameArray]):
    """Publish supplied frames with identifiable immutable metadata."""

    def __init__(self, *frames: FrameArray) -> None:
        super().__init__()
        self.frames = frames
        self.records = tuple(
            FrameMetadata(sequence_index=i, time_index=i, z_index=2)
            for i in range(len(frames))
        )

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        for frame, record in zip(self.frames, self.records, strict=True):
            self._publish(
                Product(frame, metadata=record),
                timeout=_TIMEOUT,
            )

    def _close(self) -> None:
        pass


class PooledSource(FrameSource):
    """Reuse one scientific buffer and overwrite it after the final delivery."""

    def __init__(self, *frames: FrameArray) -> None:
        super().__init__(*frames)
        self.recycled = Event()

    def _prepare(self) -> None:
        self._init_pool(
            capacity=1,
            factory=lambda: np.empty_like(self.frames[0]),
        )

    def _work(self) -> None:
        for frame, record in zip(self.frames, self.records, strict=True):
            with self._acquire(timeout=_TIMEOUT) as lease:
                np.copyto(lease.payload, frame)
                lease.metadata = record
                self._publish(lease, timeout=_TIMEOUT)

        with self._acquire(timeout=_TIMEOUT) as lease:
            lease.payload.fill(0)
            self.recycled.set()


class IdleSource(FrameSource):
    """Keep a pipeline active until cancellation without publishing frames."""

    def _work(self) -> None:
        self._abort_requested.wait()


class Collector(Sink[FrameArray]):
    """Retain display publications whose arrays have independently owned storage."""

    def __init__(
        self,
        *,
        upstream: Worker[Any, FrameArray],
        hold_first: bool = False,
    ) -> None:
        super().__init__(upstream=upstream)
        self.received: list[Product[FrameArray]] = []
        self.hold_first = hold_first
        self.entered = Event()
        self.release = Event()
        self.closed = False

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        while True:
            try:
                with self._receive(timeout=_TIMEOUT) as product:
                    self.received.append(product)

                    if self.hold_first and len(self.received) == 1:
                        self.entered.set()
                        assert self.release.wait(_TIMEOUT)
            except EndOfStream:
                return

    def _close(self) -> None:
        self.closed = True


@pytest.fixture
def settings() -> DisplaySettings:
    return DisplaySettings(
        channels=(
            ChannelSettings(
                limits=IntRange(0, 255),
                color=ColorVector((1.0, 1.0, 1.0)),
            ),
        ),
        transfer_function=Linear(),
    )


@pytest.fixture
def display(settings: DisplaySettings) -> Display:
    return Display(settings=settings)


@pytest.fixture
def pipelines() -> Generator[list[Pipeline]]:
    """Ensure assertions and expected failures cannot leave worker threads running."""
    created: list[Pipeline] = []
    try:
        yield created
    finally:
        for pipeline in reversed(created):
            if not pipeline._launched:
                continue

            pipeline.abort()
            for worker in pipeline._launched:
                worker._wait(timeout=_TIMEOUT)


def start(
    pipelines: list[Pipeline],
    *workers: Worker[Any, Any],
) -> Pipeline:
    """Register cleanup before starting a pipeline, including failed startups."""
    pipeline = Pipeline(*workers)
    pipelines.append(pipeline)
    pipeline.start()
    return pipeline


def frame(value: int) -> FrameArray:
    return np.full((1, 1, 1), value, dtype=np.uint16)


@pytest.mark.parametrize("name", [None, "preview", ""])
def test_factory_preserves_display_upstream_and_name(
    display: Display,
    name: str | None,
) -> None:
    """Create a processor connected to the selected display and upstream worker."""
    source = FrameSource()

    processor = display.create_processor(upstream=source, name=name)

    assert isinstance(processor, DisplayProcessor)
    assert processor.display is display
    assert processor.upstream is source
    assert processor.name == ("DisplayProcessor" if name is None else name)


@pytest.mark.parametrize("upstream", [None, object()])
def test_processor_requires_a_valid_upstream(
    display: Display,
    upstream,  # ruff: ignore[missing-type-function-argument]
) -> None:
    """Reject missing or unrelated upstream workers before execution."""
    with pytest.raises(TypeError, match="upstream"):
        display.create_processor(upstream=upstream)


@pytest.mark.parametrize("display", [None, object()])
def test_processor_requires_a_display(display) -> None:  # ruff: ignore[missing-type-function-argument]
    """Reject unrelated display objects before creating an upstream subscription."""
    with pytest.raises(TypeError, match="display must be a Display"):
        DisplayProcessor(upstream=FrameSource(), display=display)


def test_submit_returns_its_committed_snapshot(display: Display) -> None:
    """Return the exact image committed by a direct frame submission."""
    image = display.submit(frame(64))

    assert image is display.latest_image


@pytest.mark.parametrize("pixel_format", list(DisplayPixelFormat))
def test_frames_metadata_and_paused_redraw(
    settings: DisplaySettings,
    pixel_format: DisplayPixelFormat,
    pipelines: list[Pipeline],
) -> None:
    """Preserve output pixels and metadata after input reuse and pipeline completion."""
    display = Display(settings=settings, pixel_format=pixel_format)
    source = PooledSource(frame(64), frame(128))
    sink = Collector(upstream=display.create_processor(upstream=source))

    pipeline = start(pipelines, sink)
    pipeline.wait(timeout=_TIMEOUT)

    assert source.recycled.is_set()
    assert sink.closed
    assert len(sink.received) == 2

    for index, value in enumerate((64, 128)):
        expected = [value, value, value]
        if pixel_format is DisplayPixelFormat.BGRX32:
            expected.append(255)

        product = sink.received[index]
        np.testing.assert_array_equal(product.payload, [[expected]])
        assert product.metadata is source.records[index]
        assert not product.payload.flags.writeable

    previous = display.latest_image
    assert previous is not None
    assert previous.pixels is sink.received[-1].payload

    red = replace(
        settings,
        channels=(replace(settings.channels[0], color=ColorVector((1.0, 0.0, 0.0))),),
    )
    display.set_settings(red)

    image = display.latest_image
    assert image is not None
    assert image.metadata is source.records[-1]

    expected = (
        [128, 0, 0] if pixel_format is DisplayPixelFormat.RGB24 else [0, 0, 128, 255]
    )
    np.testing.assert_array_equal(image.pixels, [[expected]])
    np.testing.assert_array_equal(previous.pixels, sink.received[-1].payload)
    assert len(sink.received) == 2


def test_empty_stream_closes_the_consumer(
    display: Display,
    pipelines: list[Pipeline],
) -> None:
    """Propagate normal completion without creating an image for an empty stream."""
    sink = Collector(
        upstream=display.create_processor(upstream=FrameSource()),
    )

    pipeline = start(pipelines, sink)
    pipeline.wait(timeout=_TIMEOUT)

    assert sink.received == []
    assert sink.closed
    assert display.latest_image is None
    assert pipeline.done


@pytest.mark.parametrize(
    "changed",
    [
        np.zeros((2, 1, 1), dtype=np.uint16),
        np.zeros((1, 1, 2), dtype=np.uint16),
        np.zeros((1, 1, 1), dtype=np.uint8),
    ],
)
def test_layout_changes_fail_and_release_the_display(
    display: Display,
    changed: FrameArray,
    pipelines: list[Pipeline],
) -> None:
    """Reject shape or dtype changes while retaining the last valid frame."""
    source = FrameSource(frame(64), changed)
    processor = display.create_processor(upstream=source)
    pipeline = start(pipelines, processor)

    with pytest.raises(ValueError, match="must remain constant"):
        pipeline.wait(timeout=_TIMEOUT)

    image = display.latest_image
    assert image is not None
    assert image.metadata is source.records[0]
    np.testing.assert_array_equal(image.pixels, [[[64, 64, 64]]])

    replacement = display.create_processor(upstream=FrameSource())
    start(pipelines, replacement).wait(timeout=_TIMEOUT)


@pytest.mark.parametrize("invalid", [None, object()])
def test_nonarray_input_fails_and_releases_the_display(
    display: Display,
    invalid,  # ruff: ignore[missing-type-function-argument]
    pipelines: list[Pipeline],
) -> None:
    """Reject nonarray payloads without leaving the display attached."""
    processor = display.create_processor(upstream=FrameSource(invalid))
    pipeline = start(pipelines, processor)

    with pytest.raises(TypeError, match="NumPy arrays"):
        pipeline.wait(timeout=_TIMEOUT)

    replacement = display.create_processor(upstream=FrameSource())
    start(pipelines, replacement).wait(timeout=_TIMEOUT)


def test_display_can_serve_successive_acquisitions(
    display: Display,
    pipelines: list[Pipeline],
) -> None:
    """Allow fresh processors to use different layouts across completed acquisitions."""
    first = display.create_processor(upstream=FrameSource(frame(64)))
    start(pipelines, first).wait(timeout=_TIMEOUT)
    previous = display.latest_image
    assert previous is not None

    changed = np.full((2, 3, 1), 128, dtype=np.uint8)
    second = display.create_processor(upstream=FrameSource(changed))
    start(pipelines, second).wait(timeout=_TIMEOUT)

    image = display.latest_image
    assert image is not None
    np.testing.assert_array_equal(
        image.pixels,
        np.full((2, 3, 3), 128, dtype=np.uint8),
    )
    np.testing.assert_array_equal(previous.pixels, [[[64, 64, 64]]])


def test_failed_attachment_cannot_release_another_processor(
    display: Display,
    settings: DisplaySettings,
    pipelines: list[Pipeline],
) -> None:
    """Reject competing pipelines and direct submissions without losing the owner."""
    first = display.create_processor(upstream=IdleSource())
    active = start(pipelines, first)

    with pytest.raises(RuntimeError, match="active processor"):
        display.submit(frame(64))

    # Settings remain adjustable while the processor waits for scientific data.
    display.set_settings(settings)

    for _ in range(2):
        competitor = display.create_processor(upstream=FrameSource())
        with pytest.raises(RuntimeError, match="active processor"):
            start(pipelines, competitor)

    active.abort()
    active.wait(timeout=_TIMEOUT)

    replacement = display.create_processor(upstream=FrameSource())
    start(pipelines, replacement).wait(timeout=_TIMEOUT)


def test_preparation_rollback_releases_the_display(
    display: Display,
    pipelines: list[Pipeline],
) -> None:
    """Release a prepared processor's attachment when downstream preparation fails."""

    class FailingCollector(Collector):
        def _prepare(self) -> None:
            raise RuntimeError("Downstream preparation failed.")

    processor = display.create_processor(upstream=FrameSource())
    sink = FailingCollector(upstream=processor)

    with pytest.raises(RuntimeError, match="Downstream preparation"):
        start(pipelines, sink)

    replacement = display.create_processor(upstream=FrameSource())
    start(pipelines, replacement).wait(timeout=_TIMEOUT)


def test_upstream_buffer_is_released_before_publication_waits(
    display: Display,
    pipelines: list[Pipeline],
) -> None:
    """Return scientific storage even when a slow downstream consumer blocks output."""
    source = PooledSource(frame(64), frame(128), frame(192))
    sink = Collector(
        upstream=display.create_processor(upstream=source),
        hold_first=True,
    )
    pipeline = start(pipelines, sink)

    try:
        assert sink.entered.wait(_TIMEOUT)
        assert source.recycled.wait(_TIMEOUT)
    finally:
        sink.release.set()

    pipeline.wait(timeout=_TIMEOUT)
    assert len(sink.received) == 3


def test_publication_uses_the_submission_snapshot(
    display: Display,
    settings: DisplaySettings,
    pipelines: list[Pipeline],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish the submitted image even if a settings redraw replaces latest_image."""
    original = display._submit
    captured: list[DisplayImage] = []
    red = replace(
        settings,
        channels=(replace(settings.channels[0], color=ColorVector((1.0, 0.0, 0.0))),),
    )

    def submit_then_redraw(
        incoming: FrameArray,
        *,
        metadata: Metadata | None,
        processor: object | None,
    ) -> DisplayImage:
        image = original(incoming, metadata=metadata, processor=processor)
        captured.append(image)
        display.set_settings(red)
        return image

    monkeypatch.setattr(display, "_submit", submit_then_redraw)
    sink = Collector(
        upstream=display.create_processor(upstream=FrameSource(frame(128))),
    )

    start(pipelines, sink).wait(timeout=_TIMEOUT)

    assert sink.received[0].payload is captured[0].pixels
    np.testing.assert_array_equal(sink.received[0].payload, [[[128, 128, 128]]])

    latest = display.latest_image
    assert latest is not None
    np.testing.assert_array_equal(latest.pixels, [[[128, 0, 0]]])


def test_abort_releases_a_borrowed_buffer_and_display_attachment(
    display: Display,
    pipelines: list[Pipeline],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release an in-flight borrow and allow another processor after abort."""
    entered = Event()
    release = Event()
    original = display._submit

    def blocked_submit(
        incoming: FrameArray,
        *,
        metadata: Metadata | None,
        processor: object | None,
    ) -> DisplayImage:
        entered.set()
        assert release.wait(_TIMEOUT)
        return original(incoming, metadata=metadata, processor=processor)

    monkeypatch.setattr(display, "_submit", blocked_submit)
    source = PooledSource(frame(64))
    processor = display.create_processor(upstream=source)
    pipeline = start(pipelines, processor)

    try:
        assert entered.wait(_TIMEOUT)
        pipeline.abort()
    finally:
        release.set()

    pipeline.wait(timeout=_TIMEOUT)

    # Cleanup aborts the pool, so inspect returned storage rather than reacquiring.
    pool = source._pool
    assert pool is not None
    with pool._condition:
        assert len(pool._available) == 1

    replacement = display.create_processor(upstream=FrameSource())
    start(pipelines, replacement).wait(timeout=_TIMEOUT)
