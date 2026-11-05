import json
import math
from contextlib import ExitStack
from os import PathLike
from typing import Any, Literal

import numpy as np
import tifffile

from dirigo import units
from dirigo.frames import FrameArray, FrameInterpretation, FrameMetadata
from dirigo.pipeline.exceptions import EndOfStream, OperationCancelled
from dirigo.pipeline.product import Metadata, Product
from dirigo.pipeline.worker import Worker
from dirigo.sinks.writer import Writer


class TiffWriter(Writer[FrameArray]):
    """Save uint8 or uint16 frames to a BigTIFF file.

    Monochrome input has shape (Y, X), multichannel input (Y, X, C),
    and RGB input (Y, X, 3). Shape and dtype must remain constant.

    With contiguous=False, each frame has its own series and metadata record.
    With contiguous=True, frames share one series. Its leading Q axis records
    delivery order; identifiers and coordinates are saved as a list at close.
    The recording's complete flag indicates normal end-of-stream.

    Both pixel sizes must be supplied together as finite, positive Positions.
    Output is created on the first valid frame and never overwrites a file.
    """

    def __init__(
        self,
        *,
        upstream: Worker[Any, FrameArray],
        path: str | PathLike[str],
        interpretation: FrameInterpretation = "monochrome",
        contiguous: bool = False,
        pixel_size_x: units.Position | None = None,
        pixel_size_y: units.Position | None = None,
        name: str | None = None,
    ) -> None:
        if interpretation not in ("monochrome", "multichannel", "rgb"):
            raise ValueError(
                "interpretation must be 'monochrome', 'multichannel', or 'rgb'."
            )
        if not isinstance(contiguous, bool):
            raise TypeError("contiguous must be a boolean.")
        if (pixel_size_x is None) != (pixel_size_y is None):
            raise ValueError(
                "Specify both pixel_size_x and pixel_size_y, or neither."
            )

        self._resolution: tuple[float, float] | None = None
        if pixel_size_x is not None and pixel_size_y is not None:
            for label, size in (
                ("pixel_size_x", pixel_size_x),
                ("pixel_size_y", pixel_size_y),
            ):
                if not isinstance(size, units.Position):
                    raise TypeError(f"{label} must be a Position.")
                meters = float(size)
                if not math.isfinite(meters) or meters <= 0:
                    raise ValueError(f"{label} must be finite and positive.")
            self._resolution = (
                0.01 / float(pixel_size_x),
                0.01 / float(pixel_size_y),
            )

        self._pixel_size_x = pixel_size_x
        self._pixel_size_y = pixel_size_y
        self._interpretation = interpretation
        self._contiguous = contiguous
        super().__init__(upstream=upstream, path=path, name=name)

    def _prepare(self) -> None:
        """Defer output creation until the first valid frame arrives."""
        return None

    def _work(self) -> None:
        """Consume frames and finalize contiguous metadata before closing output."""
        shape: tuple[int, ...] | None = None
        dtype: np.dtype[Any] | None = None
        tif: tifffile.TiffWriter | None = None
        records: list[dict[str, int | None]] = []
        normal_end = False
        write_in_progress = False
        failure: BaseException | None = None

        with ExitStack() as resources:
            try:
                while True:
                    try:
                        with self._receive() as product:
                            frame = product.payload
                            self._validate_frame(frame)
                            first = tif is None

                            if tif is None:
                                output = resources.enter_context(self._open_file())
                                tif = resources.enter_context(
                                    tifffile.TiffWriter(
                                        output, bigtiff=True, ome=False
                                    )
                                )
                                shape, dtype = frame.shape, frame.dtype
                            elif frame.shape != shape or frame.dtype != dtype:
                                raise ValueError(
                                    "TIFF frame shape and dtype must remain constant."
                                )

                            # A failed write may leave part of an extra frame.
                            # Finalize only after pixels and bookkeeping succeed.
                            write_in_progress = True
                            self._write_frame(tif, product, first=first)
                            if self._contiguous:
                                records.append(self._frame_record(product.metadata))
                            write_in_progress = False
                    except EndOfStream:
                        normal_end = True
                        break
            except BaseException as exc:
                failure = exc
                raise
            finally:
                if (
                    self._contiguous
                    and tif is not None
                    and records
                    and not write_in_progress
                ):
                    assert shape is not None
                    try:
                        self._finalize_contiguous(
                            tif,
                            shape,
                            records,
                            complete=normal_end and not self._abort_requested.is_set(),
                        )
                    except BaseException as exc:
                        expected_cancellation = (
                            isinstance(failure, OperationCancelled)
                            and self._abort_requested.is_set()
                        )
                        if failure is None or expected_cancellation:
                            raise
                        failure.add_note(
                            f"TIFF metadata finalization also failed: {exc!r}"
                        )

    def _close(self) -> None:
        """Output resources are closed by the context managers in _work()."""
        return None

    def _validate_frame(self, frame: FrameArray) -> None:
        """Require a nonempty unsigned image matching the interpretation."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("TIFF frames must be NumPy arrays.")
        if frame.size == 0:
            raise ValueError("TIFF frames must be nonempty arrays.")
        if self._interpretation == "monochrome":
            if frame.ndim != 2:
                raise ValueError(
                    "Monochrome TIFF frames must be two-dimensional."
                )
        else:
            if frame.ndim != 3:
                raise ValueError(
                    "Multichannel and RGB TIFF frames must be three-dimensional."
                )
            if self._interpretation == "rgb" and frame.shape[2] != 3:
                raise ValueError(
                    "RGB TIFF frames must have exactly three components."
                )
        if frame.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
            raise ValueError("TIFF frames must have dtype uint8 or uint16.")

    @staticmethod
    def _frame_record(record: Metadata) -> dict[str, int | None]:
        """Extract supported identifiers without inferring missing coordinates."""
        coordinates: dict[str, int | None] = {
            "sequence_index": record.sequence_index,
            "time_index": None,
            "z_index": None,
        }
        if isinstance(record, FrameMetadata):
            coordinates["time_index"] = record.time_index
            coordinates["z_index"] = record.z_index
        return coordinates

    def _write_frame(
        self,
        tif: tifffile.TiffWriter,
        product: Product[FrameArray],
        *,
        first: bool,
    ) -> None:
        """Encode one borrowed frame with the selected series organization."""
        data = product.payload
        axes = "YX"
        photometric = "minisblack"
        planarconfig: Literal["contig"] | None = None
        if self._interpretation == "multichannel":
            data = np.moveaxis(data, -1, 0)
            axes = "CYX"
        elif self._interpretation == "rgb":
            axes = "YXS"
            photometric = "rgb"
            planarconfig = "contig"

        metadata = (
            None
            if self._contiguous
            else {"axes": axes, "dirigo": self._frame_record(product.metadata)}
        )
        tif.write(
            data,
            photometric=photometric,
            planarconfig=planarconfig,
            contiguous=self._contiguous,
            metadata=metadata,
            description=(
                "Dirigo: unfinalized contiguous recording"
                if self._contiguous and first
                else None
            ),
            resolution=self._resolution,
            resolutionunit=(
                tifffile.RESUNIT.NONE
                if self._resolution is None
                else tifffile.RESUNIT.CENTIMETER
            ),
        )

    def _finalize_contiguous(
        self,
        tif: tifffile.TiffWriter,
        shape: tuple[int, ...],
        records: list[dict[str, int | None]],
        *,
        complete: bool,
    ) -> None:
        """Describe completed frames and their metadata as one shaped series."""
        if self._interpretation == "multichannel":
            stored_shape = (len(records), shape[2], shape[0], shape[1])
            axes = "QCYX"
        else:
            stored_shape = (len(records), *shape)
            axes = "QYXS" if self._interpretation == "rgb" else "QYX"

        tif.overwrite_description(
            json.dumps(
                {
                    "shape": stored_shape,
                    "axes": axes,
                    "dirigo": {"complete": complete, "frames": records},
                }
            )
        )
