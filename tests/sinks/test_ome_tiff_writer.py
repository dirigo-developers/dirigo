"""Test finite OME Z-stacks, pooled input, and incomplete-recording cleanup."""

import json
from pathlib import Path
from threading import Event
from typing import Any, BinaryIO
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import tifffile

from dirigo import units
from dirigo.frames import FrameArray, FrameInterpretation, FrameMetadata
from dirigo.pipeline import Metadata, Pipeline, Product, Source
from dirigo.sinks import OmeTiffWriter


class StackSource(Source[FrameArray]):
    """Publish supplied planes, optionally reusing one buffer or staying open."""

    def __init__(
        self,
        products: tuple[Product[FrameArray], ...],
        *,
        pooled: bool = False,
        hold_open: bool = False,
    ) -> None:
        super().__init__()
        self.products = products
        self.pooled = pooled
        self.hold_open = hold_open
        self.buffer_ids: list[int] = []

    def _prepare(self) -> None:
        if self.pooled:
            template = self.products[0].payload
            self._init_pool(capacity=1, factory=lambda: np.empty_like(template))

    def _work(self) -> None:
        for product in self.products:
            if self.pooled:
                with self._acquire(timeout=5.0) as lease:
                    np.copyto(lease.payload, product.payload)
                    lease.metadata = product.metadata
                    self.buffer_ids.append(id(lease.payload))
                    self._publish(lease, timeout=5.0)
            else:
                self._publish(product, timeout=5.0)

        if self.hold_open:
            self._abort_requested.wait()

    def _close(self) -> None:
        pass


def _abort_and_join(pipeline: Pipeline) -> None:
    """Stop launched workers without re-raising a recorded execution failure."""
    if pipeline._launched:
        pipeline.abort()
        for worker in pipeline._launched:
            worker._wait(timeout=5.0)


def _run(writer: OmeTiffWriter) -> None:
    """Run to completion and ensure failures cannot leave threads running."""
    pipeline = Pipeline(writer)
    try:
        pipeline.start()
        pipeline.wait(timeout=5.0)
    finally:
        _abort_and_join(pipeline)


def _track_output(
    writer: OmeTiffWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> list[BinaryIO]:
    """Track real file handles so failure tests can verify closure."""
    handles: list[BinaryIO] = []
    open_file = writer._open_file

    def tracked_open() -> BinaryIO:
        handle = open_file()
        handles.append(handle)
        return handle

    monkeypatch.setattr(writer, "_open_file", tracked_open)
    return handles


@pytest.mark.parametrize(
    ("interpretation", "channels"),
    [("monochrome", 1), ("multichannel", 1), ("multichannel", 3)],
)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("z_planes", [1, 3])
@pytest.mark.parametrize("calibrated", [False, True])
def test_stack_round_trip_with_reused_input_buffer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interpretation: FrameInterpretation,
    channels: int,
    dtype: type[np.uint8] | type[np.uint16],
    z_planes: int,
    calibrated: bool,
) -> None:
    """Write one OME image with exact pixels, dimensions, calibration, and records.

    A one-buffer source overwrites its storage for every plane. Include singleton
    Z/channel dimensions and three scientific channels. Compare pixels after
    reshaping to ZCYX because readers may omit singleton axes.
    """
    shape = (4, 5) if interpretation == "monochrome" else (4, 5, channels)
    frames = tuple(
        np.arange(np.prod(shape), dtype=dtype).reshape(shape) + z * 10
        for z in range(z_planes)
    )
    for z, frame in enumerate(frames):
        frame[0, 0] = np.iinfo(dtype).max - z

    records = tuple(
        FrameMetadata(sequence_index=10 + z * 3, time_index=0, z_index=z)
        for z in range(z_planes)
    )
    products = tuple(
        Product(frame, metadata=record)
        for frame, record in zip(frames, records, strict=True)
    )
    source = StackSource(products, pooled=True)
    path = tmp_path / "recording" / "stack.ome.tif"
    writer = OmeTiffWriter(
        upstream=source,
        path=path,
        z_planes=z_planes,
        interpretation=interpretation,
        pixel_size_x=units.Position("0.5 um") if calibrated else None,
        pixel_size_y=units.Position("1.25 um") if calibrated else None,
        z_spacing=units.Position("2 um") if calibrated else None,
    )
    handles = _track_output(writer, monkeypatch)

    _run(writer)

    assert len(handles) == 1
    assert handles[0].closed
    assert len(source.buffer_ids) == z_planes
    assert len(set(source.buffer_ids)) == 1

    with tifffile.TiffFile(path) as tif:
        assert tif.is_bigtiff
        assert tif.is_ome
        assert len(tif.series) == 1
        assert len(tif.pages) == z_planes * channels
        assert tif.pages.first.photometric == tifffile.PHOTOMETRIC.MINISBLACK
        assert tif.pages.first.samplesperpixel == 1

        actual = tif.series[0].asarray().reshape(z_planes, channels, 4, 5)
        expected = np.stack(frames)
        expected = (
            expected[:, np.newaxis, ...]
            if interpretation == "monochrome"
            else np.moveaxis(expected, -1, 1)
        )
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)

        xml = tif.ome_metadata
        assert xml is not None
        root = ET.fromstring(xml)

    pixels = root.find(".//{*}Pixels")
    assert pixels is not None
    assert pixels.attrib["SizeX"] == "5"
    assert pixels.attrib["SizeY"] == "4"
    assert pixels.attrib["SizeZ"] == str(z_planes)
    assert pixels.attrib["SizeC"] == str(channels)
    assert pixels.attrib["SizeT"] == "1"
    assert pixels.attrib["Type"] == np.dtype(dtype).name

    for axis, size in (("X", 0.5e-6), ("Y", 1.25e-6), ("Z", 2e-6)):
        key = f"PhysicalSize{axis}"
        if calibrated:
            assert float(pixels.attrib[key]) == pytest.approx(size)
            assert pixels.attrib[f"{key}Unit"] == "m"
        else:
            assert key not in pixels.attrib

    entry = root.find(
        ".//{*}MapAnnotation/{*}Value/{*}M[@K='frame_metadata']"
    )
    assert entry is not None and entry.text is not None
    assert json.loads(entry.text) == [
        {
            "sequence_index": record.sequence_index,
            "time_index": record.time_index,
            "z_index": record.z_index,
        }
        for record in records
    ]


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"z_planes": 0}, ValueError),
        ({"z_planes": -1}, ValueError),
        ({"z_planes": True}, TypeError),
        ({"z_planes": 1.5}, TypeError),
        ({"interpretation": "rgb"}, ValueError),
        ({"z_spacing": units.Position(0)}, ValueError),
        ({"z_spacing": units.Position(-1e-6)}, ValueError),
        ({"z_spacing": units.Position(float("nan"))}, ValueError),
        ({"z_spacing": units.Position(float("inf"))}, ValueError),
        ({"z_spacing": 1e-6}, TypeError),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path,
    options: dict[str, Any],
    error: type[Exception],
) -> None:
    """Reject invalid depth counts, interpretations, and spacing before execution."""
    arguments: dict[str, Any] = {"z_planes": 1}
    arguments.update(options)

    with pytest.raises(error, match=next(iter(options))):
        OmeTiffWriter(
            upstream=StackSource(()),
            path=tmp_path / "stack.ome.tif",
            **arguments,
        )


@pytest.mark.parametrize(
    ("records", "z_planes", "message", "saved_pages"),
    [
        ((), 2, "Incomplete", 0),
        ((FrameMetadata(z_index=0),), 2, "Incomplete", 1),
        ((Metadata(),), 1, "FrameMetadata", 0),
        ((FrameMetadata(),), 1, "z_index", 0),
        ((FrameMetadata(z_index=1),), 1, "Expected z_index", 0),
        (
            (FrameMetadata(z_index=0), FrameMetadata(z_index=0)),
            2,
            "Expected z_index",
            1,
        ),
        (
            (FrameMetadata(z_index=0), FrameMetadata(z_index=2)),
            2,
            "Expected z_index",
            1,
        ),
        (
            (FrameMetadata(z_index=0), FrameMetadata(z_index=1)),
            1,
            "more than",
            1,
        ),
        ((FrameMetadata(time_index=1, z_index=0),), 1, "time_index", 0),
    ],
)
def test_invalid_stream_is_not_finalized_as_ome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: tuple[Metadata, ...],
    z_planes: int,
    message: str,
    saved_pages: int,
) -> None:
    """Reject incomplete or invalid streams and close any partial output.

    Exercise missing coordinates, incorrect Z order, excess planes, and an
    unsupported time point. Only preceding valid planes may have been written.
    Failure before the first valid plane must create no file.
    """
    products = tuple(
        Product(np.full((4, 5), i, dtype=np.uint8), metadata=record)
        for i, record in enumerate(records)
    )
    path = tmp_path / "stack.ome.tif"
    writer = OmeTiffWriter(
        upstream=StackSource(products), path=path, z_planes=z_planes
    )
    handles = _track_output(writer, monkeypatch)

    with pytest.raises(ValueError, match=message):
        _run(writer)

    assert all(handle.closed for handle in handles)
    if saved_pages == 0:
        assert not path.exists()
    else:
        assert len(handles) == 1
        with tifffile.TiffFile(path) as tif:
            assert not tif.is_ome
            assert len(tif.pages) == saved_pages


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_layout_change_rejects_plane_without_losing_first(
    tmp_path: Path,
    change: str,
) -> None:
    """Reject a changed shape or dtype while retaining the preceding valid plane."""
    first = np.arange(20, dtype=np.uint8).reshape(4, 5)
    second = (
        np.zeros((5, 4), dtype=np.uint8)
        if change == "shape"
        else np.zeros((4, 5), dtype=np.uint16)
    )
    products = (
        Product(first, metadata=FrameMetadata(z_index=0)),
        Product(second, metadata=FrameMetadata(z_index=1)),
    )
    path = tmp_path / "stack.ome.tif"
    writer = OmeTiffWriter(upstream=StackSource(products), path=path, z_planes=2)

    with pytest.raises(ValueError, match="constant"):
        _run(writer)

    with tifffile.TiffFile(path) as tif:
        assert not tif.is_ome
        assert len(tif.pages) == 1
        np.testing.assert_array_equal(tif.pages.first.asarray(), first)


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    """Exclusive creation preserves an existing destination byte for byte."""
    path = tmp_path / "stack.ome.tif"
    path.write_bytes(b"existing recording")
    product = Product(
        np.zeros((4, 5), dtype=np.uint8), metadata=FrameMetadata(z_index=0)
    )
    writer = OmeTiffWriter(
        upstream=StackSource((product,)), path=path, z_planes=1
    )

    with pytest.raises(FileExistsError):
        _run(writer)

    assert path.read_bytes() == b"existing recording"


def test_abort_closes_incrementally_written_output_without_ome_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write before end-of-stream and close partial output on cancellation.

    The source stays open after publishing one multichannel plane. Wait for a
    real TIFF write before aborting, proving that output starts before the full
    stack arrives. All channels of the borrowed plane may finish before abort.
    """
    written = Event()
    write = tifffile.TiffWriter.write

    def tracked_write(
        tif: tifffile.TiffWriter, *args: Any, **kwargs: Any
    ) -> Any:
        result = write(tif, *args, **kwargs)
        written.set()
        return result

    monkeypatch.setattr(tifffile.TiffWriter, "write", tracked_write)
    product = Product(
        np.zeros((4, 5, 2), dtype=np.uint8),
        metadata=FrameMetadata(z_index=0),
    )
    path = tmp_path / "stack.ome.tif"
    writer = OmeTiffWriter(
        upstream=StackSource((product,), hold_open=True),
        path=path,
        z_planes=2,
        interpretation="multichannel",
    )
    handles = _track_output(writer, monkeypatch)
    pipeline = Pipeline(writer)

    try:
        pipeline.start()
        assert written.wait(timeout=5.0)
        pipeline.abort()
        pipeline.wait(timeout=5.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert len(handles) == 1
        assert handles[0].closed
        with tifffile.TiffFile(path) as tif:
            assert not tif.is_ome
            assert len(tif.pages) == 2
    finally:
        _abort_and_join(pipeline)
