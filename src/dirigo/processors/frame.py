"""Reconstruct the imaging records of a nominal resonant raster frame."""

from dirigo import units
from dirigo.frames import FrameArray, FrameMetadata
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.product import Product
from dirigo.pipeline.worker import Worker
from dirigo.processors._encoding import _encode_int16
from dirigo.processors._reconstruction import _prepare_reconstructor
from dirigo.records import RecordBatch
from dirigo.resonant_raster import ResonantLineReconstructor, ResonantRasterPlan
from dirigo.sources.raster_frame_acquisition import RasterFrameAcquisition


def _reconstruct_frame(
    batch: RecordBatch,
    *,
    plan: ResonantRasterPlan,
    reconstructor: ResonantLineReconstructor,
    sample_code_range: units.IntRange,
    use_numba: bool = False,
) -> FrameArray:
    """Select one frame's imaging records and encode oriented rows."""
    records_per_frame = plan.records_per_frame
    if (
        batch.records_per_frame != records_per_frame
        or batch.layout.records_per_batch != records_per_frame
    ):
        raise ValueError("Frame processing requires one complete frame per batch.")
    if batch.first_record < 0 or batch.first_record % records_per_frame != 0:
        raise ValueError("Record batch must start at a frame boundary.")
    if batch.samples.shape != batch.layout.shape:
        raise ValueError("Frame samples must match the capture layout shape.")

    samples = batch.samples[: plan.image_records]
    if use_numba:
        pixels = reconstructor.reconstruct_int16(
            samples, sample_code_range=sample_code_range
        )
    else:
        averages = reconstructor.reconstruct(samples)
        if averages.shape[2] == 1:
            averages = averages[:, :, 0]
        pixels = _encode_int16(averages, sample_code_range)

    if pixels.shape[:2] != plan.geometry.shape:
        raise ValueError("Reconstructed frame shape differs from the raster plan.")
    pixels.setflags(write=False)
    if pixels.ndim == 3 and pixels.shape[2] == 1:
        return pixels[:, :, 0]
    return pixels


class FrameProcessor(Worker[RecordBatch, FrameArray]):
    """Publish one owned, read-only int16 frame per nominal frame batch.

    The batch starts with the imaging records and ends with flyback records.
    Monochrome frames have shape (Y, X). Multichannel frames use (Y, X, C)
    in capture channel order.
    """

    def __init__(
        self, *, upstream: RasterFrameAcquisition, name: str | None = None
    ) -> None:
        """Connect to an acquisition whose frame plan resolves at preparation."""
        if not isinstance(upstream, RasterFrameAcquisition):
            raise TypeError("upstream must be a RasterFrameAcquisition.")

        super().__init__(upstream=upstream, name=name)
        self._source = upstream
        self._reconstructor: ResonantLineReconstructor | None = None
        self._use_numba: bool = False

    def _prepare(self) -> None:
        """Build nominal pixel bins and prepare the selected backend."""
        plan = self._source.plan
        self._reconstructor, self._use_numba = _prepare_reconstructor(
            self._source, plan.fast_axis, columns=plan.geometry.columns
        )

    def _work(self) -> None:
        """Reconstruct and publish frames until the acquisition ends."""
        reconstructor = self._reconstructor
        if reconstructor is None:
            raise RuntimeError("Frame processing requires successful preparation.")

        plan = self._source.plan
        sample_code_range = self._source.sample_code_range
        while True:
            try:
                with self._receive() as product:
                    frame = _reconstruct_frame(
                        product.payload,
                        plan=plan,
                        reconstructor=reconstructor,
                        sample_code_range=sample_code_range,
                        use_numba=self._use_numba,
                    )
                    metadata = product.metadata
                    if not isinstance(metadata, FrameMetadata):
                        metadata = FrameMetadata(sequence_index=metadata.sequence_index)
            except EndOfStream:
                return

            self._publish(Product(frame, metadata=metadata))

    def _close(self) -> None:
        """Release the prepared reconstruction tables."""
        self._reconstructor = None
