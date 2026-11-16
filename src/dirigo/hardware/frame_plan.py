"""Describe complete frames within a finite stream of acquired records."""

from dataclasses import dataclass

from dirigo.hardware.capture import CaptureSettings


@dataclass(frozen=True, slots=True, kw_only=True)
class RecordFramePlan:
    """Group a finite capture's records into complete, equal-sized frames.

    capture_settings supplies the total record count through records_per_batch
    and batch_count. records_per_frame determines frame membership independently
    of transfer batching. Every captured record belongs to exactly one frame;
    partial frames are rejected, and no records are padded or discarded.

    Indices are zero-based acquired-record indices. The first frame starts at
    record zero. A record can contain multiple image lines; this plan does not
    specify image geometry or infer physical timing from sample_rate or
    record_length. Both independently triggered and consecutive records can
    be grouped. Hardware backends must validate their own timing support.

    Construction performs no hardware operations. Returned ranges describe
    logical record positions without allocating sample arrays or index lists.
    """

    capture_settings: CaptureSettings
    records_per_frame: int

    def __post_init__(self) -> None:
        """Require finite delivery and a positive frame size dividing the capture."""
        if not isinstance(self.capture_settings, CaptureSettings):
            raise TypeError("capture_settings must be CaptureSettings.")
        if self.capture_settings.batch_count is None:
            raise ValueError("Frame planning requires a finite capture.")
        if not isinstance(self.records_per_frame, int) or isinstance(
            self.records_per_frame, bool
        ):
            raise TypeError("records_per_frame must be an integer.")
        if self.records_per_frame <= 0:
            raise ValueError("records_per_frame must be positive.")

        if self.record_count % self.records_per_frame:
            raise ValueError("Capture must contain a whole number of frames.")

    @property
    def record_count(self) -> int:
        """Total acquired records across all requested transfer batches."""
        batch_count = self.capture_settings.batch_count
        assert batch_count is not None
        return self.capture_settings.records_per_batch * batch_count

    @property
    def frame_count(self) -> int:
        """Number of complete frames contained in the capture."""
        return self.record_count // self.records_per_frame

    @property
    def frame_start_indices(self) -> range:
        """Start record of each frame, excluding the terminal capture boundary."""
        return range(0, self.record_count, self.records_per_frame)

    def records_for_frame(self, frame_index: int) -> range:
        """Return acquired-record indices for a zero-based frame.

        Require an integer index in [0, frame_count). Negative indexing is not
        supported. Raise TypeError for a noninteger index or IndexError for an
        index outside the plan.
        """
        if not isinstance(frame_index, int) or isinstance(frame_index, bool):
            raise TypeError("frame_index must be an integer.")
        if not 0 <= frame_index < self.frame_count:
            raise IndexError("frame_index is outside the plan.")

        first = frame_index * self.records_per_frame
        return range(first, first + self.records_per_frame)
