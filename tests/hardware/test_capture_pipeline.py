"""Exercise simulated digitizer capture through threaded pipeline delivery."""

from dataclasses import dataclass, field, replace
from typing import Never

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureInfo, CaptureSettings, SampleArray
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.hardware import Hardware
from dirigo.hardware.simulated_digitizer import (
    SimulatedDigitizer,
    SimulatedDigitizerConfig,
)
from dirigo.pipeline.exceptions import EndOfStream
from dirigo.pipeline.hardware_worker import HardwareWorker
from dirigo.pipeline.pipeline import Pipeline
from dirigo.pipeline.product import Metadata
from dirigo.pipeline.sink import Sink


@dataclass(slots=True, eq=False)
class CaptureBatch:
    """Pool samples and their current capture description as one payload.

    Both fields may be changed only while holding a producer lease. Consumers
    must copy samples if they need them after releasing their delivery.
    CaptureInfo itself is immutable and may be retained independently.
    """

    samples: SampleArray
    info: CaptureInfo = field(default_factory=CaptureInfo)


class CaptureAcquisition(HardwareWorker[Never, CaptureBatch]):
    """Test helper connecting a borrowed digitizer to the worker output pool."""

    def __init__(
        self,
        device_name: str,
        settings: CaptureSettings,
        *,
        pool_capacity: int = 1,
    ) -> None:
        super().__init__()
        self._device_name = device_name
        self._settings = settings
        self._pool_capacity = pool_capacity
        self.session: DigitizerCaptureSession | None = None

    def _prepare(self) -> None:
        """Prepare capture before allocating storage from its effective layout."""
        digitizer = self._hardware.require(self._device_name, Digitizer)
        session = digitizer.create_capture(self._settings)
        self.session = session
        session.prepare()

        layout = session.layout
        self._init_pool(
            capacity=self._pool_capacity,
            factory=lambda: CaptureBatch(np.empty(layout.shape, dtype=layout.dtype)),
        )

    def _work(self) -> None:
        """Publish complete batches; release unpublished leases on timeout or EOF."""
        session = self.session
        assert session is not None
        if self._abort_requested.is_set():
            return

        session.start()
        sequence_index = 0

        while not self._abort_requested.is_set():
            with self._acquire() as lease:
                try:
                    info = session.read_into(lease.payload.samples, timeout=0.05)
                except TimeoutError:
                    continue

                if info is None:
                    return

                lease.payload.info = info
                lease.metadata = Metadata(sequence_index=sequence_index)
                self._publish(lease)
                sequence_index += 1

    def _close(self) -> None:
        """Release this worker's reservation without closing its digitizer."""
        if self.session is not None:
            self.session.close()


class CaptureCollector(Sink[CaptureBatch]):
    """Copy sample arrays while deliveries still own their pooled storage."""

    def __init__(self, upstream: CaptureAcquisition) -> None:
        super().__init__(upstream=upstream)
        self.received: list[tuple[SampleArray, CaptureInfo, Metadata]] = []
        self.sample_ids: list[int] = []

    def _prepare(self) -> None:
        """Allocate no external resources."""
        return

    def _work(self) -> None:
        """Drain the stream, retaining owned samples and immutable descriptions."""
        while True:
            try:
                with self._receive() as product:
                    batch = product.payload
                    self.sample_ids.append(id(batch.samples))
                    self.received.append(
                        (batch.samples.copy(), batch.info, product.metadata)
                    )
            except EndOfStream:
                return

    def _close(self) -> None:
        """Keep collected copies available to the test after completion."""
        return


def _settings() -> CaptureSettings:
    """Select reversed channels and enough batches to force pool reuse."""
    return CaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=4,
        channel_indices=(1, 0),
        records_per_batch=2,
        batch_count=5,
        timestamps_enabled=True,
    )


def _run_capture(
    hardware: Hardware,
    settings: CaptureSettings,
) -> tuple[CaptureAcquisition, CaptureCollector]:
    """Run fresh workers and join them before the caller releases hardware."""
    acquisition = CaptureAcquisition("adc", settings, pool_capacity=1)
    collector = CaptureCollector(acquisition)
    pipeline = Pipeline(acquisition, collector, hardware=hardware)

    pipeline.start()
    try:
        pipeline.wait(timeout=5)
        assert pipeline.done
    finally:
        if not pipeline.done:
            pipeline.abort()
            pipeline.wait(timeout=5)

    return acquisition, collector


@pytest.mark.parametrize("timestamps_enabled", [False, True])
def test_capture_pipeline_delivers_complete_batches_with_reused_storage(
    timestamps_enabled: bool,
) -> None:
    """Preserve samples, channel order, capture metadata, and sequence identifiers.

    A one-buffer pool forces all five deliveries to share the same sample
    array. The collector's copies must retain earlier batches after that
    storage is overwritten. Optional timestamps must survive the same reuse.

    Normal exhaustion must close the session while leaving Hardware open.
    """
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())
    hardware = Hardware({"adc": digitizer})
    settings = replace(_settings(), timestamps_enabled=timestamps_enabled)

    try:
        acquisition, collector = _run_capture(hardware, settings)

        assert len(collector.received) == 5
        assert len(set(collector.sample_ids)) == 1
        assert [metadata.sequence_index for _, _, metadata in collector.received] == [
            0,
            1,
            2,
            3,
            4,
        ]

        for index, (samples, info, _) in enumerate(collector.received):
            assert samples.shape == (2, 4, 2)
            assert samples.dtype == np.dtype(np.int16)

            channel_zero = np.arange(
                -32768 + 8 * index,
                -32760 + 8 * index,
                dtype=np.int16,
            ).reshape(2, 4)
            np.testing.assert_array_equal(samples[:, :, 0], channel_zero + 257)
            np.testing.assert_array_equal(samples[:, :, 1], channel_zero)

            times = info.timestamps
            if timestamps_enabled:
                assert times is not None
                np.testing.assert_allclose(
                    times,
                    [0.008 * index, 0.008 * index + 0.004],
                )
            else:
                assert times is None

        session = acquisition.session
        assert session is not None
        assert session.state is CaptureState.CLOSED
        assert session.batches_acquired == 5
        assert session.layout.channel_indices == (1, 0)
        assert not digitizer.is_active
        assert hardware.require("adc", Digitizer) is digitizer
    finally:
        hardware.close()


def test_successive_pipelines_reuse_the_same_digitizer() -> None:
    """Release capture reservations while keeping the caller's device reusable.

    Each pipeline uses fresh single-use workers and a fresh capture session.
    The second preparation proves the first reservation was released.
    Matching outputs prove that sample positions and sequence indices restart
    independently of the persistent device.
    """
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())
    hardware = Hardware({"adc": digitizer})
    settings = _settings()

    try:
        first_worker, first = _run_capture(hardware, settings)
        second_worker, second = _run_capture(hardware, settings)

        assert first_worker.session is not second_worker.session
        assert len(first.received) == len(second.received) == 5

        for previous, current in zip(first.received, second.received, strict=True):
            first_samples, first_info, first_metadata = previous
            second_samples, second_info, second_metadata = current

            np.testing.assert_array_equal(first_samples, second_samples)

            first_times = first_info.timestamps
            second_times = second_info.timestamps
            assert first_times is not None
            assert second_times is not None
            np.testing.assert_array_equal(first_times, second_times)

            assert first_metadata == second_metadata

        assert hardware.require("adc", Digitizer) is digitizer
    finally:
        hardware.close()
