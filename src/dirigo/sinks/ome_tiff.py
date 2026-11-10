import json
import math
from contextlib import ExitStack
from os import PathLike
from typing import Any

import numpy as np
import tifffile

from dirigo import units
from dirigo.frames import FrameArray, FrameInterpretation, FrameMetadata
from dirigo.pipeline.exceptions import EndOfStream, OperationCancelled
from dirigo.pipeline.worker import Worker
from dirigo.sinks.tiff import TiffWriter


class OmeTiffWriter(TiffWriter):
    """Write one finite, ordered monochrome or multichannel OME Z-stack.

    Each product contains one XY plane with all its channels and must carry
    FrameMetadata with the next z_index. time_index may be None or zero.

    Pages are written while their input product is borrowed. OME metadata is
    added only after exactly z_planes products and normal end-of-stream.
    Interrupted or invalid streams retain partial TIFF output without an
    OME completion description. Existing files are never overwritten.
    """

    def __init__(
        self,
        *,
        upstream: Worker[Any, FrameArray],
        path: str | PathLike[str],
        z_planes: int,
        interpretation: FrameInterpretation = "monochrome",
        pixel_size_x: units.Length | None = None,
        pixel_size_y: units.Length | None = None,
        z_spacing: units.Length | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(z_planes, int) or isinstance(z_planes, bool):
            raise TypeError("z_planes must be an integer.")
        if z_planes <= 0:
            raise ValueError("z_planes must be positive.")
        if interpretation not in ("monochrome", "multichannel"):
            raise ValueError("interpretation must be 'monochrome' or 'multichannel'.")
        if z_spacing is not None:
            if not isinstance(z_spacing, units.Length):
                raise TypeError("z_spacing must be a Position.")
            if not math.isfinite(float(z_spacing)) or float(z_spacing) <= 0:
                raise ValueError("z_spacing must be finite and positive.")

        self._z_planes = z_planes
        self._z_spacing = z_spacing
        super().__init__(
            upstream=upstream,
            path=path,
            interpretation=interpretation,
            pixel_size_x=pixel_size_x,
            pixel_size_y=pixel_size_y,
            name=name,
        )

    def _work(self) -> None:
        """Write pages incrementally, then describe only a complete stack."""
        shape: tuple[int, ...] | None = None
        dtype: np.dtype[Any] | None = None
        tif: tifffile.TiffWriter | None = None
        records: list[dict[str, int | None]] = []

        with ExitStack() as resources:
            while True:
                try:
                    with self._receive() as product:
                        index = len(records)
                        if index >= self._z_planes:
                            raise ValueError(
                                f"Received more than {self._z_planes} Z planes."
                            )

                        frame = product.payload
                        self._validate_frame(frame)
                        record = product.metadata
                        if not isinstance(record, FrameMetadata):
                            raise ValueError("OME Z-stacks require FrameMetadata.")
                        if record.z_index is None:
                            raise ValueError("OME Z-stacks require z_index.")
                        if record.time_index not in (None, 0):
                            raise ValueError("time_index must be None or zero.")
                        if record.z_index != index:
                            raise ValueError(
                                f"Expected z_index {index}, received {record.z_index}."
                            )

                        if tif is None:
                            output = resources.enter_context(self._open_file())
                            tif = resources.enter_context(
                                tifffile.TiffWriter(output, bigtiff=True, ome=False)
                            )
                            shape, dtype = frame.shape, frame.dtype
                        elif frame.shape != shape or frame.dtype != dtype:
                            raise ValueError(
                                "OME frame shape and dtype must remain constant."
                            )

                        self._write_z_plane(tif, frame)
                        records.append(
                            {
                                "sequence_index": record.sequence_index,
                                "time_index": record.time_index,
                                "z_index": record.z_index,
                            }
                        )
                except EndOfStream:
                    break

            if self._abort_requested.is_set():
                raise OperationCancelled("OME-TIFF recording was aborted.")
            if len(records) != self._z_planes:
                raise ValueError(
                    f"Incomplete Z-stack: expected {self._z_planes} planes, "
                    f"received {len(records)}."
                )

            assert tif is not None
            assert shape is not None
            assert dtype is not None
            self._finalize_ome(tif, shape, dtype, records)

    def _write_z_plane(
        self,
        tif: tifffile.TiffWriter,
        frame: FrameArray,
    ) -> None:
        """Write each scientific channel before releasing the borrowed plane."""
        channels = (
            frame[np.newaxis, ...]
            if self._interpretation == "monochrome"
            else np.moveaxis(frame, -1, 0)
        )
        resolution_unit = (
            tifffile.RESUNIT.NONE
            if self._resolution is None
            else tifffile.RESUNIT.CENTIMETER
        )
        for channel in channels:
            tif.write(
                channel,
                photometric="minisblack",
                contiguous=True,
                metadata=None,
                description="Dirigo: incomplete Z-stack",
                resolution=self._resolution,
                resolutionunit=resolution_unit,
            )

    def _finalize_ome(
        self,
        tif: tifffile.TiffWriter,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        records: list[dict[str, int | None]],
    ) -> None:
        """Map completed grayscale pages to one ZCYX image and retain identifiers."""
        height, width = shape[:2]
        channels = 1 if self._interpretation == "monochrome" else shape[2]
        calibration: dict[str, Any] = {}
        for axis, spacing in (
            ("X", self._pixel_size_x),
            ("Y", self._pixel_size_y),
            ("Z", self._z_spacing),
        ):
            if spacing is not None:
                calibration[f"PhysicalSize{axis}"] = float(spacing)
                calibration[f"PhysicalSize{axis}Unit"] = "m"

        ome = tifffile.OmeXml()
        ome.addimage(
            dtype=dtype,
            shape=(self._z_planes, channels, height, width),
            storedshape=(self._z_planes * channels, 1, 1, height, width, 1),
            axes="ZCYX",
            MapAnnotation={
                "Namespace": "dirigo",
                "frame_metadata": json.dumps(records),
            },
            **calibration,
        )
        tif.overwrite_description(ome.tostring())
