from pathlib import Path

import numpy as np

from dirigo.frames import FrameArray, FrameMetadata
from dirigo.pipeline import Pipeline, Product, Source
from dirigo.sinks import TiffWriter


class SyntheticFrameSource(Source[FrameArray]):
    """Publish 100 monochrome frames with a rolling linear gradient."""

    def _prepare(self) -> None:
        """No resources need preparation for this source."""

    def _work(self) -> None:
        """Generate frames on the source's execution thread."""
        pattern = np.arange(
            256 * 256,
            dtype=np.uint16,
        ).reshape(256, 256)

        for index in range(100):
            # Addition creates a fresh array for each publication.
            frame = pattern + np.uint16(index * 500)

            product = Product(
                frame,
                metadata=FrameMetadata(
                    sequence_index=index,
                    time_index=index,
                ),
            )
            self._publish(product)

    def _close(self) -> None:
        """No external resources need cleanup."""


def main() -> None:
    destination = Path("synthetic_frames.tif")

    source = SyntheticFrameSource()
    writer = TiffWriter(
        upstream=source,
        path=destination,
        interpretation="monochrome",
    )

    # Including the writer also includes its upstream source.
    pipeline = Pipeline(writer)

    pipeline.start()

    try:
        pipeline.wait()
    except KeyboardInterrupt:
        # Cancellation is asynchronous; wait for cleanup before exiting.
        pipeline.abort()
        pipeline.wait()
        raise

    print(f"Saved 1000 frames to {destination.resolve()}")


if __name__ == "__main__":
    main()
