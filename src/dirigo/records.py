"""Raw record batches published by acquisition workers."""

from dataclasses import dataclass

from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray


@dataclass(frozen=True, slots=True, eq=False)
class RecordBatch:
    """Own one raw batch and its acquisition-position information.

    samples uses layout.shape and layout.dtype and is read-only after
    publication. Each batch owns an independent array, so unpooled products
    may be retained after receive() exits. info retains optional record times.

    first_record is the zero-based logical record index within this acquisition.
    records_per_frame describes logical grouping, independent of DMA batches.
    Neither field corrects counter startup phase or galvo response delay;
    downstream frame processing applies that alignment.
    """

    samples: SampleArray
    layout: CaptureLayout
    info: CaptureInfo
    first_record: int
    records_per_frame: int
