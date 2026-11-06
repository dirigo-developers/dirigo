"""Test monochrome TIFF recording through real pipeline execution."""

from pathlib import Path
from threading import Event
from typing import BinaryIO

import numpy as np
import pytest
import tifffile

from dirigo import units
from dirigo.frames import FrameArray, FrameInterpretation, FrameMetadata
from dirigo.pipeline import Metadata, Pipeline, Product, Source
from dirigo.sinks import TiffWriter


class FrameSource(Source[FrameArray]):
    """Publish a finite sequence of synthetic frames."""

    def __init__(self, frames: tuple[FrameArray, ...]) -> None:
        super().__init__()
        self.frames = frames

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        for frame in self.frames:
            self._publish(frame, timeout=2.0)

    def _close(self) -> None:
        pass


class ProductSource(Source[FrameArray]):
    """Publish supplied products with their original metadata."""

    def __init__(self, products: tuple[Product[FrameArray], ...]) -> None:
        super().__init__()
        self.products = products

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        for product in self.products:
            self._publish(product, timeout=2.0)

    def _close(self) -> None:
        pass


class PooledFrameSource(FrameSource):
    """Publish frames and identifiers while reusing one output buffer."""

    def _prepare(self) -> None:
        template = self.frames[0]
        self._init_pool(capacity=1, factory=lambda: np.empty_like(template))

    def _work(self) -> None:
        for index, frame in enumerate(self.frames):
            with self._acquire(timeout=5.0) as lease:
                np.copyto(lease.payload, frame)
                lease.metadata = Metadata(sequence_index=index)
                self._publish(lease, timeout=5.0)


class HoldingProductSource(ProductSource):
    """Publish products, then keep the stream open until the pipeline aborts."""

    def _work(self) -> None:
        super()._work()
        self._abort_requested.wait()


def _run_writer(writer: TiffWriter) -> None:
    """Run the writer and always join launched threads, including after failure."""
    pipeline = Pipeline(writer)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)
    finally:
        if pipeline._launched:
            pipeline.abort()

            for worker in pipeline._launched:
                worker._wait(timeout=2.0)


def _track_tiff_output(
    writer: TiffWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> list[BinaryIO]:
    """Track the actual output handle to verify closure on every exit path."""
    handles: list[BinaryIO] = []
    open_file = writer._open_file

    def tracked_open() -> BinaryIO:
        handle = open_file()
        handles.append(handle)
        return handle

    monkeypatch.setattr(writer, "_open_file", tracked_open)
    return handles


def _fail_tiff_finalization(*args: object, **kwargs: object) -> None:
    """Inject a metadata failure without changing ordinary file cleanup."""
    raise OSError("metadata failure")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_frames_round_trip_in_delivery_order(
    tmp_path: Path,
    dtype: type[np.uint8] | type[np.uint16],
) -> None:
    """Preserve frame count, order, shape, dtype, and pixel values.

    Use spatially varying frames with distinct values and a pixel near the
    dtype's maximum to exercise both ordering and the full unsigned bit depth.
    """
    frames = tuple(np.arange(20, dtype=dtype).reshape(4, 5) + i for i in range(3))
    for i, frame in enumerate(frames):
        frame[0, 0] = np.iinfo(dtype).max - i

    path = tmp_path / "recording.tif"
    writer = TiffWriter(upstream=FrameSource(frames), path=path)

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        assert tif.is_bigtiff
        assert len(tif.pages) == len(frames)

        for page, expected in zip(tif.pages, frames, strict=True):
            actual = page.asarray()

            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


def test_empty_stream_creates_no_output(tmp_path: Path) -> None:
    """An empty stream completes without creating the file or its parent."""
    path = tmp_path / "recording" / "frames.tif"
    writer = TiffWriter(upstream=FrameSource(()), path=path)

    _run_writer(writer)

    assert not path.parent.exists()


def test_existing_output_is_preserved(tmp_path: Path) -> None:
    """An existing destination raises FileExistsError without changing its bytes."""
    path = tmp_path / "recording.tif"
    path.write_bytes(b"existing recording")
    frame = np.zeros((4, 5), dtype=np.uint8)
    writer = TiffWriter(upstream=FrameSource((frame,)), path=path)

    with pytest.raises(FileExistsError):
        _run_writer(writer)

    assert path.read_bytes() == b"existing recording"


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        pytest.param(
            np.zeros((4, 5, 1), dtype=np.uint8),
            "two-dimensional",
            id="extra-axis",
        ),
        pytest.param(
            np.zeros((0, 5), dtype=np.uint8),
            "nonempty",
            id="empty-frame",
        ),
        pytest.param(
            np.zeros((4, 5), dtype=np.float32),
            "uint8 or uint16",
            id="unsupported-dtype",
        ),
    ],
)
def test_invalid_first_frame_creates_no_output(
    tmp_path: Path,
    frame: FrameArray,
    message: str,
) -> None:
    """Reject an unsupported first frame before creating any output."""
    path = tmp_path / "recording" / "frames.tif"
    writer = TiffWriter(upstream=FrameSource((frame,)), path=path)

    with pytest.raises(ValueError, match=message):
        _run_writer(writer)

    assert not path.parent.exists()


@pytest.mark.parametrize(
    "payload",
    [None, [[1, 2], [3, 4]], object()],
    ids=["none", "nested-list", "arbitrary-object"],
)
def test_non_array_payload_is_rejected(
    tmp_path: Path,
    payload: object,
) -> None:
    """Reject non-array payloads before creating the output or its parent.

    Deliberately violate the source's annotated payload type to exercise
    runtime validation at the writer boundary.
    """
    source = FrameSource((payload,))  # ty: ignore[invalid-argument-type]
    path = tmp_path / "recording" / "frames.tif"
    writer = TiffWriter(upstream=source, path=path)

    with pytest.raises(TypeError, match="TIFF frames must be NumPy arrays"):
        _run_writer(writer)

    assert not path.parent.exists()


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_layout_change_closes_output_and_preserves_earlier_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """Reject a later layout change, close the handle, and retain earlier data.

    Write one valid frame before sending a different shape or dtype. Track the
    real output handle to verify closure, then reopen the TIFF to verify that
    the first frame was finalized successfully.
    """
    first = np.arange(20, dtype=np.uint8).reshape(4, 5)
    second = (
        np.zeros((5, 4), dtype=np.uint8)
        if change == "shape"
        else np.zeros((4, 5), dtype=np.uint16)
    )
    path = tmp_path / "recording.tif"
    writer = TiffWriter(
        upstream=FrameSource((first, second)),
        path=path,
    )

    handles: list[BinaryIO] = []
    open_file = writer._open_file

    def tracked_open() -> BinaryIO:
        handle = open_file()
        handles.append(handle)
        return handle

    monkeypatch.setattr(writer, "_open_file", tracked_open)

    with pytest.raises(ValueError, match="constant"):
        _run_writer(writer)

    assert len(handles) == 1
    assert handles[0].closed

    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 1
        np.testing.assert_array_equal(tif.pages[0].asarray(), first)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_multichannel_frames_preserve_channels_and_order(
    tmp_path: Path,
    dtype: type[np.uint8] | type[np.uint16],
    channels: int,
) -> None:
    """Preserve scientific channels as grayscale pages grouped by input frame.

    Use distinct spatial, channel, and frame values. Include three-channel
    input to verify that its shape does not cause RGB interpretation, and
    one-channel input to verify preservation of the declared channel axis.
    """
    frames = tuple(
        np.arange(5 * 7 * channels, dtype=dtype).reshape(5, 7, channels) + i * 10
        for i in range(2)
    )
    for i, frame in enumerate(frames):
        frame[0, 0, 0] = np.iinfo(dtype).max - i

    path = tmp_path / "channels.tif"
    writer = TiffWriter(
        upstream=FrameSource(frames),
        path=path,
        interpretation="multichannel",
    )

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == len(frames) * channels

        for frame_index, expected in enumerate(frames):
            for channel in range(channels):
                page = tif.pages[frame_index * channels + channel]

                if isinstance(page, tifffile.TiffFrame):
                    page = page.aspage()
                assert page.photometric == tifffile.PHOTOMETRIC.MINISBLACK
                assert page.samplesperpixel == 1
                np.testing.assert_array_equal(
                    page.asarray(),
                    expected[:, :, channel],
                )

        assert len(tif.series) == len(frames)

        for series, expected in zip(tif.series, frames, strict=True):
            actual = series.asarray()

            assert series.axes == "CYX"
            assert actual.dtype == expected.dtype
            assert actual.shape == (channels, 5, 7)
            np.testing.assert_array_equal(
                actual,
                np.moveaxis(expected, -1, 0),
            )


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_rgb_frames_preserve_pixels_and_rgb_interpretation(
    tmp_path: Path,
    dtype: type[np.uint8] | type[np.uint16],
) -> None:
    """Preserve RGB pixels and declare three interleaved color components."""
    frames = tuple(
        np.arange(60, dtype=dtype).reshape(4, 5, 3) + i * 10 for i in range(2)
    )
    for i, frame in enumerate(frames):
        frame[0, 0, 0] = np.iinfo(dtype).max - i

    path = tmp_path / "rgb.tif"
    writer = TiffWriter(
        upstream=FrameSource(frames),
        path=path,
        interpretation="rgb",
    )

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == len(frames)

        for page, expected in zip(tif.pages, frames, strict=True):
            actual = page.asarray()

            if isinstance(page, tifffile.TiffFrame):
                page = page.aspage()
            assert page.photometric == tifffile.PHOTOMETRIC.RGB
            assert page.planarconfig == tifffile.PLANARCONFIG.CONTIG
            assert page.samplesperpixel == 3
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("interpretation", ["", "RGB", "gray"])
def test_unknown_interpretation_is_rejected(
    tmp_path: Path,
    interpretation: str,
) -> None:
    """Reject an unsupported interpretation during construction."""
    with pytest.raises(ValueError, match="interpretation"):
        TiffWriter(
            upstream=FrameSource(()),
            path=tmp_path / "frames.tif",
            interpretation=interpretation,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize(
    ("interpretation", "frame", "message"),
    [
        pytest.param(
            "multichannel",
            np.zeros((4, 5), dtype=np.uint8),
            "three-dimensional",
            id="multichannel-missing-channel-axis",
        ),
        pytest.param(
            "multichannel",
            np.zeros((4, 5, 0), dtype=np.uint8),
            "nonempty",
            id="multichannel-no-channels",
        ),
        pytest.param(
            "rgb",
            np.zeros((4, 5), dtype=np.uint8),
            "three-dimensional",
            id="rgb-missing-component-axis",
        ),
        pytest.param(
            "rgb",
            np.zeros((4, 5, 1), dtype=np.uint8),
            "exactly three",
            id="rgb-too-few-components",
        ),
        pytest.param(
            "rgb",
            np.zeros((4, 5, 4), dtype=np.uint8),
            "exactly three",
            id="rgb-too-many-components",
        ),
    ],
)
def test_shape_must_match_interpretation(
    tmp_path: Path,
    interpretation: FrameInterpretation,
    frame: FrameArray,
    message: str,
) -> None:
    """Reject a mismatched first frame before creating any output."""
    path = tmp_path / "recording" / "frames.tif"
    writer = TiffWriter(
        upstream=FrameSource((frame,)),
        path=path,
        interpretation=interpretation,
    )

    with pytest.raises(ValueError, match=message):
        _run_writer(writer)

    assert not path.parent.exists()


def test_scientific_channel_count_cannot_change(tmp_path: Path) -> None:
    """Reject a changed channel count while retaining the first complete frame."""
    first = np.arange(40, dtype=np.uint8).reshape(4, 5, 2)
    second = np.zeros((4, 5, 3), dtype=np.uint8)
    path = tmp_path / "channels.tif"
    writer = TiffWriter(
        upstream=FrameSource((first, second)),
        path=path,
        interpretation="multichannel",
    )

    with pytest.raises(ValueError, match="constant"):
        _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        assert len(tif.pages) == 2
        assert len(tif.series) == 1
        np.testing.assert_array_equal(
            tif.series[0].asarray(),
            np.moveaxis(first, -1, 0),
        )


@pytest.mark.parametrize(
    ("interpretation", "shape"),
    [
        ("monochrome", (4, 5)),
        ("multichannel", (4, 5, 2)),
        ("rgb", (4, 5, 3)),
    ],
)
@pytest.mark.parametrize("calibrated", [False, True])
def test_spatial_calibration_matches_configuration(
    tmp_path: Path,
    interpretation: FrameInterpretation,
    shape: tuple[int, ...],
    calibrated: bool,
) -> None:
    """Write supplied X/Y spacing on every page, or declare no absolute unit.

    Use unequal X/Y spacing to detect swapped axes. Publish two frames so
    calibration is checked across successive frames and scientific channels.
    """
    frames = (
        np.zeros(shape, dtype=np.uint16),
        np.ones(shape, dtype=np.uint16),
    )
    path = tmp_path / "calibrated.tif"
    writer = TiffWriter(
        upstream=FrameSource(frames),
        path=path,
        interpretation=interpretation,
        pixel_size_x=units.Position("0.5 um") if calibrated else None,
        pixel_size_y=units.Position("1.25 um") if calibrated else None,
    )

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        for page in tif.pages:
            if isinstance(page, tifffile.TiffFrame):
                page = page.aspage()

            resolution_unit = page.tags["ResolutionUnit"].value

            if not calibrated:
                assert resolution_unit == tifffile.RESUNIT.NONE
                continue

            assert resolution_unit == tifffile.RESUNIT.CENTIMETER

            x_numerator, x_denominator = page.tags["XResolution"].value
            y_numerator, y_denominator = page.tags["YResolution"].value

            # Convert pixels per centimeter back to meters per pixel.
            actual_spacing = (
                0.01 * x_denominator / x_numerator,
                0.01 * y_denominator / y_numerator,
            )
            assert actual_spacing == pytest.approx(
                (0.5e-6, 1.25e-6),
                rel=1e-6,
                abs=0.0,
            )


@pytest.mark.parametrize(
    ("pixel_size_x", "pixel_size_y"),
    [
        (units.Position("1 um"), None),
        (None, units.Position("1 um")),
    ],
)
def test_spatial_calibration_requires_both_axes(
    tmp_path: Path,
    pixel_size_x: units.Position | None,
    pixel_size_y: units.Position | None,
) -> None:
    """Reject calibration with only one axis specified during construction."""
    with pytest.raises(ValueError, match="both"):
        TiffWriter(
            upstream=FrameSource(()),
            path=tmp_path / "frames.tif",
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
        )


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "size",
    [
        units.Position(0),
        units.Position(-1e-6),
        units.Position(float("nan")),
        units.Position(float("inf")),
    ],
    ids=["zero", "negative", "nan", "infinite"],
)
def test_pixel_sizes_must_be_finite_and_positive(
    tmp_path: Path,
    axis: str,
    size: units.Position,
) -> None:
    """Reject zero, negative, and nonfinite spacing on either axis."""
    valid = units.Position("1 um")

    with pytest.raises(ValueError, match=f"pixel_size_{axis}"):
        TiffWriter(
            upstream=FrameSource(()),
            path=tmp_path / "frames.tif",
            pixel_size_x=size if axis == "x" else valid,
            pixel_size_y=size if axis == "y" else valid,
        )


@pytest.mark.parametrize(
    "size",
    [
        1e-6,
        "1 um",
        units.Time("1 s"),
    ],
    ids=["bare-number", "bare-string", "wrong-dimension"],
)
def test_pixel_sizes_require_position_quantities(
    tmp_path: Path,
    size: object,
) -> None:
    """Require explicit Position quantities rather than ambiguous input values."""
    with pytest.raises(TypeError, match="Position"):
        TiffWriter(
            upstream=FrameSource(()),
            path=tmp_path / "frames.tif",
            pixel_size_x=size,  # ty: ignore[invalid-argument-type]
            pixel_size_y=units.Position("1 um"),
        )


@pytest.mark.parametrize(
    ("interpretation", "shape", "axes"),
    [
        ("monochrome", (4, 5), "YX"),
        ("multichannel", (4, 5, 3), "CYX"),
        ("rgb", (4, 5, 3), "YXS"),
    ],
)
def test_frame_coordinates_remain_paired_with_saved_pixels(
    tmp_path: Path,
    interpretation: FrameInterpretation,
    shape: tuple[int, ...],
    axes: str,
) -> None:
    """Preserve each frame's identifiers, coordinates, and pixels in delivery order.

    Use sequence gaps, reordered coordinates, and a repeated coordinate pair.
    The writer must preserve these labels without sorting, renumbering, or
    merging frames. Each input product must produce one TIFF series.
    """
    records = (
        FrameMetadata(sequence_index=9, time_index=2, z_index=3),
        FrameMetadata(sequence_index=15, time_index=0, z_index=1),
        FrameMetadata(sequence_index=11, time_index=1, z_index=0),
        FrameMetadata(sequence_index=17, time_index=2, z_index=3),
    )
    frames = tuple(
        np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + i * 100
        for i in range(len(records))
    )
    products = tuple(
        Product(frame, metadata=record)
        for frame, record in zip(frames, records, strict=True)
    )
    path = tmp_path / "coordinates.tif"
    writer = TiffWriter(
        upstream=ProductSource(products),
        path=path,
        interpretation=interpretation,
    )

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        saved_records = tif.shaped_metadata

        assert saved_records is not None
        assert len(saved_records) == len(products)
        assert len(tif.series) == len(products)

        for index, record in enumerate(records):
            saved = saved_records[index]
            series = tif.series[index]
            expected = frames[index]

            assert saved["axes"] == axes
            assert saved["dirigo"] == {
                "sequence_index": record.sequence_index,
                "time_index": record.time_index,
                "z_index": record.z_index,
            }
            assert series.axes == axes

            if interpretation == "multichannel":
                expected = np.moveaxis(expected, -1, 0)

            np.testing.assert_array_equal(series.asarray(), expected)


def test_missing_coordinates_are_not_inferred_or_carried_forward(
    tmp_path: Path,
) -> None:
    """Preserve missing coordinates and accept ordinary pipeline metadata.

    Follow a fully specified frame with partially specified and empty records.
    Later records must not inherit earlier coordinates, and explicit zero
    indices must remain distinguishable from unspecified indices.
    """
    records: tuple[Metadata, ...] = (
        FrameMetadata(sequence_index=4, time_index=0, z_index=0),
        Metadata(sequence_index=8),
        FrameMetadata(time_index=0),
        FrameMetadata(z_index=0),
        Metadata(),
        FrameMetadata(),
    )
    products = tuple(
        Product(np.full((4, 5), index, dtype=np.uint8), metadata=record)
        for index, record in enumerate(records)
    )
    path = tmp_path / "unspecified.tif"
    writer = TiffWriter(upstream=ProductSource(products), path=path)

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        saved_records = tif.shaped_metadata

        assert saved_records is not None
        assert len(saved_records) == len(records)

        coordinates = [
            (
                saved["dirigo"]["sequence_index"],
                saved["dirigo"]["time_index"],
                saved["dirigo"]["z_index"],
            )
            for saved in saved_records
        ]

    assert coordinates == [
        (4, 0, 0),
        (8, None, None),
        (None, 0, None),
        (None, None, 0),
        (None, None, None),
        (None, None, None),
    ]


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("frame_count", [1, 3])
@pytest.mark.parametrize(
    ("interpretation", "shape", "axes"),
    [
        ("monochrome", (4, 5), "YX"),
        ("multichannel", (4, 5, 1), "CYX"),
        ("multichannel", (4, 5, 3), "CYX"),
        ("rgb", (4, 5, 3), "YXS"),
    ],
)
def test_tiff_layout_preserves_pixels_metadata_and_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contiguous: bool,
    dtype: type[np.uint8] | type[np.uint16],
    frame_count: int,
    interpretation: FrameInterpretation,
    shape: tuple[int, ...],
    axes: str,
) -> None:
    """Preserve pixels and records in either per-frame or contiguous layout.

    Include singleton dimensions, both bit depths, and three scientific
    channels. Metadata includes gaps, zero coordinates, and unspecified axes.
    For contiguous output, verify adjacency of every pixel strip on disk.
    """
    frames = tuple(
        np.arange(np.prod(shape), dtype=dtype).reshape(shape) + i * 10
        for i in range(frame_count)
    )
    for index, frame in enumerate(frames):
        frame[0, 0] = np.iinfo(dtype).max - index

    records: tuple[Metadata, ...] = (
        FrameMetadata(sequence_index=7, time_index=2, z_index=0),
        Metadata(sequence_index=12),
        FrameMetadata(sequence_index=9, time_index=0, z_index=3),
    )
    expected_records = [
        {"sequence_index": 7, "time_index": 2, "z_index": 0},
        {"sequence_index": 12, "time_index": None, "z_index": None},
        {"sequence_index": 9, "time_index": 0, "z_index": 3},
    ][:frame_count]
    products = tuple(
        Product(frame, metadata=record)
        for frame, record in zip(frames, records[:frame_count], strict=True)
    )
    path = tmp_path / "recording.tif"
    writer = TiffWriter(
        upstream=ProductSource(products),
        path=path,
        interpretation=interpretation,
        contiguous=contiguous,
        pixel_size_x=units.Position("0.5 um"),
        pixel_size_y=units.Position("1.25 um"),
    )
    handles = _track_tiff_output(writer, monkeypatch)

    _run_writer(writer)

    assert len(handles) == 1
    assert handles[0].closed
    stored_frames = tuple(
        np.moveaxis(frame, -1, 0) if interpretation == "multichannel" else frame
        for frame in frames
    )
    with tifffile.TiffFile(path) as tif:
        saved = tif.shaped_metadata
        assert saved is not None

        if contiguous:
            assert len(tif.series) == len(saved) == 1
            assert tif.series[0].axes == "Q" + axes
            actual = tif.series[0].asarray()
            expected = np.stack(stored_frames)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
            assert saved[0]["dirigo"] == {
                "complete": True,
                "frames": expected_records,
            }
        else:
            assert len(tif.series) == len(saved) == frame_count
            for index, expected in enumerate(stored_frames):
                assert tif.series[index].axes == axes
                assert saved[index]["dirigo"] == expected_records[index]
                np.testing.assert_array_equal(tif.series[index].asarray(), expected)

        previous_end: int | None = None
        for page in tif.pages:
            if isinstance(page, tifffile.TiffFrame):
                page = page.aspage()
            assert page.resolutionunit == tifffile.RESUNIT.CENTIMETER
            x_num, x_den = page.tags["XResolution"].value
            y_num, y_den = page.tags["YResolution"].value
            assert (0.01 * x_den / x_num, 0.01 * y_den / y_num) == pytest.approx(
                (0.5e-6, 1.25e-6), rel=1e-6, abs=0.0
            )
            for offset, count in zip(
                page.dataoffsets, page.databytecounts, strict=True
            ):
                if contiguous and previous_end is not None:
                    assert offset == previous_end
                previous_end = offset + count


def test_contiguous_writer_releases_reused_input_storage(tmp_path: Path) -> None:
    """Save distinct frames correctly while a one-buffer source reuses storage."""
    frames = tuple(
        np.full((4, 5, 3), 1000 + index, dtype=np.uint16) for index in range(3)
    )
    path = tmp_path / "pooled.tif"
    writer = TiffWriter(
        upstream=PooledFrameSource(frames),
        path=path,
        interpretation="multichannel",
        contiguous=True,
    )

    _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        np.testing.assert_array_equal(
            tif.series[0].asarray(), np.moveaxis(np.stack(frames), -1, 1)
        )


def test_contiguous_validation_failure_finalizes_completed_frames(
    tmp_path: Path,
) -> None:
    """Retain earlier pixels and metadata with complete=False after rejection.

    The first product is valid. A later product changes shape and must not
    appear in either the recording's final shape or its frame records.
    """
    frame = np.arange(20, dtype=np.uint8).reshape(4, 5)
    products = (
        Product(frame, metadata=FrameMetadata(sequence_index=4, z_index=0)),
        Product(np.zeros((5, 4), dtype=np.uint8)),
    )
    path = tmp_path / "partial.tif"
    writer = TiffWriter(upstream=ProductSource(products), path=path, contiguous=True)

    with pytest.raises(ValueError, match="constant"):
        _run_writer(writer)

    with tifffile.TiffFile(path) as tif:
        saved = tif.shaped_metadata
        assert saved is not None
        assert len(saved) == 1
        assert saved[0]["dirigo"] == {
            "complete": False,
            "frames": [{"sequence_index": 4, "time_index": None, "z_index": 0}],
        }
        np.testing.assert_array_equal(tif.series[0].asarray(), frame[np.newaxis])


@pytest.mark.parametrize("fail_finalization", [False, True])
def test_contiguous_abort_finalizes_frames_or_reports_metadata_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_finalization: bool,
) -> None:
    """Finalize completed frames after abort and report any metadata failure.

    Keep the source open after one product and wait for its real frame write
    before aborting. Inject a finalization error in the second case to ensure
    expected cancellation cannot silently suppress an output failure.
    """
    frame = np.arange(20, dtype=np.uint8).reshape(4, 5)
    product = Product(frame, metadata=Metadata(sequence_index=6))
    path = tmp_path / "aborted.tif"
    writer = TiffWriter(
        upstream=HoldingProductSource((product,)), path=path, contiguous=True
    )
    handles = _track_tiff_output(writer, monkeypatch)
    written = Event()
    write_frame = writer._write_frame

    def tracked_write(
        tif: tifffile.TiffWriter,
        incoming: Product[FrameArray],
        *,
        first: bool,
    ) -> None:
        write_frame(tif, incoming, first=first)
        written.set()

    monkeypatch.setattr(writer, "_write_frame", tracked_write)
    if fail_finalization:
        monkeypatch.setattr(writer, "_finalize_contiguous", _fail_tiff_finalization)
    pipeline = Pipeline(writer)
    try:
        pipeline.start()
        assert written.wait(timeout=5.0)
        pipeline.abort()
        if fail_finalization:
            with pytest.raises(OSError, match="metadata failure"):
                pipeline.wait(timeout=5.0)
        else:
            pipeline.wait(timeout=5.0)
            assert pipeline.exception is None

        assert len(handles) == 1
        assert handles[0].closed
        if not fail_finalization:
            with tifffile.TiffFile(path) as tif:
                saved = tif.shaped_metadata
                assert saved is not None
                assert saved[0]["dirigo"] == {
                    "complete": False,
                    "frames": [
                        {"sequence_index": 6, "time_index": None, "z_index": None}
                    ],
                }
                np.testing.assert_array_equal(
                    tif.series[0].asarray(), frame[np.newaxis]
                )
    finally:
        pipeline.abort()
        for worker in pipeline._launched:
            worker._wait(timeout=5.0)


@pytest.mark.parametrize("invalid_next_frame", [False, True])
def test_contiguous_finalization_preserves_existing_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_next_frame: bool,
) -> None:
    """Raise finalization errors on success and annotate an existing failure."""
    frame = np.zeros((4, 5), dtype=np.uint8)
    frames = (
        (frame, np.zeros((5, 4), dtype=np.uint8)) if invalid_next_frame else (frame,)
    )
    writer = TiffWriter(
        upstream=FrameSource(frames),
        path=tmp_path / "failure.tif",
        contiguous=True,
    )
    handles = _track_tiff_output(writer, monkeypatch)
    monkeypatch.setattr(writer, "_finalize_contiguous", _fail_tiff_finalization)
    error = ValueError if invalid_next_frame else OSError
    message = "constant" if invalid_next_frame else "metadata failure"

    with pytest.raises(error, match=message) as caught:
        _run_writer(writer)

    assert len(handles) == 1
    assert handles[0].closed
    if invalid_next_frame:
        assert any(
            "metadata failure" in note
            for note in getattr(caught.value, "__notes__", [])
        )


def test_contiguous_write_failure_does_not_commit_a_guessed_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave metadata unfinalized when an encoder error makes frame count uncertain.

    Raise after the second real frame write to model a backend error after
    output may have changed. Reporting only the first completed record as the
    file's final shape would misdescribe the additional pixel data.
    """
    frame = np.zeros((4, 5), dtype=np.uint8)
    path = tmp_path / "write_failure.tif"
    writer = TiffWriter(
        upstream=FrameSource((frame, frame)), path=path, contiguous=True
    )
    handles = _track_tiff_output(writer, monkeypatch)
    write_frame = writer._write_frame

    def failing_write(
        tif: tifffile.TiffWriter,
        incoming: Product[FrameArray],
        *,
        first: bool,
    ) -> None:
        write_frame(tif, incoming, first=first)
        if not first:
            raise OSError("frame write failed")

    monkeypatch.setattr(writer, "_write_frame", failing_write)
    with pytest.raises(OSError, match="frame write failed"):
        _run_writer(writer)

    assert len(handles) == 1
    assert handles[0].closed
    with tifffile.TiffFile(path) as tif:
        assert tif.shaped_metadata is None


@pytest.mark.parametrize("contiguous", [None, 0, 1, "yes"])
def test_contiguous_option_requires_boolean(tmp_path: Path, contiguous: object) -> None:
    """Reject non-boolean options instead of interpreting their truthiness."""
    with pytest.raises(TypeError, match="contiguous"):
        TiffWriter(
            upstream=FrameSource(()),
            path=tmp_path / "frames.tif",
            contiguous=contiguous,  # ty: ignore[invalid-argument-type]
        )


def test_empty_contiguous_recording_creates_no_output(tmp_path: Path) -> None:
    """An empty stream creates neither output nor its parent directory."""
    path = tmp_path / "recording" / "empty.tif"
    writer = TiffWriter(upstream=FrameSource(()), path=path, contiguous=True)

    _run_writer(writer)

    assert not path.parent.exists()


def test_contiguous_recording_preserves_existing_file(tmp_path: Path) -> None:
    """Contiguous recording retains exclusive creation and preserves existing data."""
    path = tmp_path / "existing.tif"
    path.write_bytes(b"existing recording")
    writer = TiffWriter(
        upstream=FrameSource((np.zeros((4, 5), dtype=np.uint8),)),
        path=path,
        contiguous=True,
    )

    with pytest.raises(FileExistsError):
        _run_writer(writer)

    assert path.read_bytes() == b"existing recording"
