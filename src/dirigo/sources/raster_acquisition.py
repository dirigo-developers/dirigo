"""Shared execution and resource cleanup for raster acquisition workers."""

from time import monotonic
from typing import Never, Protocol

import numpy as np

from dirigo import units
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.pipeline.product import Metadata, Product
from dirigo.pipeline.worker import Worker
from dirigo.records import RecordBatch
from dirigo.system import System


class _RasterSyncSession(Protocol):
    """Operations needed by the shared acquisition execution path."""

    def start(self) -> None:
        """Start or arm the coordinated acquisition."""
        ...

    def close(self) -> None:
        """Stop activity and release owned synchronization resources."""
        ...


class RasterAcquisition(Worker[Never, RecordBatch]):
    """Share record delivery while subclasses prepare a particular raster scan.

    Subclasses implement _prepare(). Store capture and sync sessions as soon
    as they are created so cleanup can handle partial preparation. Set the
    settling deadline, read timeout, optional frame grouping, and finally
    _prepared after preparation succeeds.

    Session methods are called serially. The supplied System and its devices
    remain owned by the caller.
    """

    def __init__(
        self,
        *,
        system: System,
        name: str | None = None,
    ) -> None:
        """Store the borrowed system without accessing hardware."""
        if not isinstance(system, System):
            raise TypeError("system must be a System.")

        super().__init__(name=name)
        self._system = system
        self._capture: DigitizerCaptureSession | None = None
        self._sync: _RasterSyncSession | None = None
        self._read_timeout = 0.05
        self._records_per_frame: int | None = None
        self._sample_code_range: units.IntRange | None = None
        self._settled_at = 0.0
        self._prepared = False
        self._closed = False

    @property
    def sample_code_range(self) -> units.IntRange:
        """Inclusive digitizer code bounds captured during preparation."""
        bounds = self._sample_code_range
        if not self._prepared or bounds is None:
            raise RuntimeError("The sample code range requires successful preparation.")
        return bounds

    def _work(self) -> None:
        """Read and publish complete batches until the capture count or abort."""
        capture = self._capture
        sync = self._sync
        if not self._prepared or capture is None or sync is None:
            raise RuntimeError("Raster acquisition requires successful preparation.")

        remaining = max(0.0, self._settled_at - monotonic())
        if self._abort_requested.wait(remaining):
            return

        sync.start()
        layout = capture.layout
        batch_count = capture.settings.batch_count
        index = 0

        while (
            batch_count is None or index < batch_count
        ) and not self._abort_requested.is_set():
            samples = np.empty(layout.shape, dtype=layout.dtype)

            while not self._abort_requested.is_set():
                try:
                    info = capture.read_into(samples, timeout=self._read_timeout)
                except TimeoutError:
                    if capture.state is CaptureState.CLOSED:
                        raise
                    continue

                if self._abort_requested.is_set():
                    return
                if info is None:
                    if batch_count is None:
                        raise RuntimeError("Unlimited capture ended unexpectedly.")
                    raise RuntimeError(
                        "Capture ended before the requested batch count."
                    )

                samples.setflags(write=False)
                self._publish(
                    Product(
                        RecordBatch(
                            samples=samples,
                            layout=layout,
                            info=info,
                            first_record=index * layout.records_per_batch,
                            records_per_frame=self._records_per_frame,
                        ),
                        metadata=Metadata(sequence_index=index),
                    )
                )
                index += 1
                break

    def _close(self) -> None:
        """Attempt all session and scanner cleanup once, preserving failures."""
        if self._closed:
            return
        self._closed = True

        sync, self._sync = self._sync, None
        capture, self._capture = self._capture, None
        failures: list[BaseException] = []

        if sync is not None:
            try:
                sync.close()
            except BaseException as failure:
                failures.append(failure)

        if capture is not None:
            try:
                capture.close()
            except BaseException as failure:
                failures.append(failure)

        try:
            self._close_scanners()
        except BaseException as failure:
            failures.append(failure)

        if failures:
            raise BaseExceptionGroup("Raster acquisition cleanup failed.", failures)

    def _close_scanners(self) -> None:
        """Apply subclass scanner shutdown policy after session closure.

        Support partial preparation. Session cleanup has already been attempted
        when this hook runs.
        """
        return
