"""Reconstruct the useful sweeps from a raster line acquisition."""

from dirigo.lines import LineBatch
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.product import Product
from dirigo.pipeline.worker import Worker
from dirigo.processors._encoding import _encode_int16
from dirigo.records import RecordBatch
from dirigo.resonant_raster import ResonantLineReconstructor
from dirigo.sources.raster_line_acquisition import RasterLineAcquisition


class LineProcessor(Worker[RecordBatch, LineBatch]):
    """Publish one independently owned LineBatch for each raw batch."""

    def __init__(
        self, *, upstream: RasterLineAcquisition, name: str | None = None
    ) -> None:
        """Connect to a line acquisition whose plan will resolve at preparation."""
        if not isinstance(upstream, RasterLineAcquisition):
            raise TypeError("upstream must be a RasterLineAcquisition.")

        super().__init__(upstream=upstream, name=name)
        self._source = upstream
        self._reconstructor: ResonantLineReconstructor | None = None

    def _prepare(self) -> None:
        """Build nominal pixel bins from the prepared acquisition plan."""
        plan = self._source.plan
        self._reconstructor = ResonantLineReconstructor(
            plan.fast_axis, columns=plan.columns
        )

    def _work(self) -> None:
        """Drain the upstream record stream and publish signed pixels."""
        reconstructor = self._reconstructor
        if reconstructor is None:
            raise RuntimeError("Line processing requires successful preparation.")

        plan = self._source.plan
        pixel_width = plan.pixel_width
        lines_per_record = plan.fast_axis.lines_per_record
        sample_code_range = self._source.sample_code_range

        while True:
            try:
                with self._receive() as product:
                    batch = product.payload
                    pixels = _encode_int16(
                        reconstructor.reconstruct(batch.samples), sample_code_range
                    )
                    pixels.setflags(write=False)
                    output = Product(
                        LineBatch(
                            pixels=pixels,
                            pixel_width=pixel_width,
                            channel_indices=batch.layout.channel_indices,
                            first_record=batch.first_record,
                            lines_per_record=lines_per_record,
                            info=batch.info,
                        ),
                        metadata=product.metadata,
                    )
            except EndOfStream:
                return

            self._publish(output)

    def _close(self) -> None:
        """Release the prepared reconstruction tables."""
        self._reconstructor = None
