"""Connect scientific frame streams to a persistent Display."""

from typing import Any

import numpy as np

from dirigo.display import Display
from dirigo.frames import FrameArray
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.product import Product
from dirigo.pipeline.worker import Worker


class DisplayProcessor(Worker[FrameArray, FrameArray]):
    """Render fixed-layout scientific frames and publish owned display arrays.

    Shape and dtype must remain constant during execution. Output arrays are
    read-only and independently owned; their storage is never recycled by this
    worker. Every output preserves the corresponding scientific frame's metadata.

    Preparation reserves the Display. Cleanup releases that reservation without
    clearing its settings, retained scientific frame, or latest image.
    """

    def __init__(
        self,
        *,
        upstream: Worker[Any, FrameArray],
        display: Display,
        name: str | None = None,
    ) -> None:
        if upstream is None:
            raise TypeError("DisplayProcessor requires an upstream worker.")
        if not isinstance(display, Display):
            raise TypeError("display must be a Display.")

        super().__init__(upstream=upstream, name=name)
        self._display = display

    @property
    def display(self) -> Display:
        return self._display

    def _prepare(self) -> None:
        self._display._attach_processor(self)

    def _work(self) -> None:
        shape: tuple[int, ...] | None = None
        dtype: np.dtype[Any] | None = None

        while True:
            try:
                with self._receive() as product:
                    frame = product.payload
                    if not isinstance(frame, np.ndarray):
                        raise TypeError("Display frames must be NumPy arrays.")
                    if shape is not None and (
                        frame.shape != shape or frame.dtype != dtype
                    ):
                        raise ValueError(
                            "Display processor frame shape and dtype "
                            "must remain constant."
                        )

                    image = self._display._submit(
                        frame,
                        metadata=product.metadata,
                        processor=self,
                    )
                    shape, dtype = frame.shape, frame.dtype
            except EndOfStream:
                return

            # Display owns the scientific copy and output pixels. Release the
            # upstream borrow before waiting for downstream publication.
            self._publish(Product(image.pixels, metadata=image.metadata))

    def _close(self) -> None:
        self._display._detach_processor(self)
