import math
import weakref
from typing import cast

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.digitizer import CaptureLayout, SampleArray

from ._demo_acquire import DemoAcquire

EXPECTED = np.array(
    [
        [[0, 2], [10, 12], [20, 22], [30, 32]],
        [[100, 102], [110, 112], [120, 122], [130, 132]],
    ],
    dtype=np.int16,
)


def destination_for(acquire: DemoAcquire) -> SampleArray:
    return np.empty(acquire.layout.shape, dtype=acquire.layout.dtype)


def test_prepare_establishes_layout_without_starting() -> None:
    acquire = DemoAcquire(channel_indices=(0, 2))

    assert not acquire.is_active

    with pytest.raises(RuntimeError):
        _ = acquire.layout

    acquire.prepare()

    assert acquire.layout.shape == (2, 4, 2)
    assert acquire.layout.channel_indices == (0, 2)
    assert acquire.layout.dtype == np.dtype(np.int16)
    assert acquire.buffers_acquired == 0
    assert not acquire.is_active

    acquire.stop()


def test_lifecycle_rejects_invalid_call_order() -> None:
    acquire = DemoAcquire()
    destination = np.empty((2, 4, 2), dtype=np.int16)

    with pytest.raises(RuntimeError):
        acquire.start()

    with pytest.raises(RuntimeError):
        acquire.read_into(destination)

    acquire.stop()
    acquire.prepare()

    with pytest.raises(RuntimeError):
        acquire.prepare()

    with pytest.raises(RuntimeError):
        acquire.read_into(destination)

    acquire.start()

    with pytest.raises(RuntimeError):
        acquire.start()

    acquire.stop()
    acquire.stop()

    with pytest.raises(RuntimeError):
        _ = acquire.layout

    with pytest.raises(RuntimeError):
        acquire.read_into(destination)


def test_preparation_locks_all_acquisition_settings() -> None:
    acquire = DemoAcquire()
    acquire.prepare()

    for name, value in (
        ("trigger_delay", 0),
        ("record_length", 8),
        ("records_per_buffer", 3),
        ("buffers_per_acquisition", 4),
        ("buffers_allocated", 3),
        ("timestamps_enabled", True),
    ):
        with pytest.raises(RuntimeError):
            setattr(acquire, name, value)

    acquire.stop()
    acquire.record_length = 8
    acquire.prepare()

    assert acquire.layout.shape == (2, 8, 2)

    acquire.stop()


def test_invalid_settings_are_rejected() -> None:
    acquire = DemoAcquire()

    for name, value, error in (
        ("record_length", 3, ValueError),
        ("record_length", 6, ValueError),
        ("trigger_delay", -3, ValueError),
        ("trigger_delay", 2, ValueError),
        ("trigger_delay", 20, ValueError),
        ("records_per_buffer", 0, ValueError),
        ("buffers_allocated", 0, ValueError),
        ("buffers_per_acquisition", -2, ValueError),
        ("record_length", True, TypeError),
        ("timestamps_enabled", 1, TypeError),
    ):
        with pytest.raises(error):
            setattr(acquire, name, value)

    acquire.trigger_delay = -2
    acquire.trigger_delay = 4

    assert acquire.trigger_delay == 4


def test_invalid_channel_configuration_and_clock_are_rejected() -> None:
    acquire = DemoAcquire(channel_indices=())

    with pytest.raises(ValueError):
        acquire.prepare()

    acquire.stop()

    for channels in ((0, 0), (-1, 0)):
        with pytest.raises(ValueError):
            DemoAcquire(channel_indices=channels)

    for rate in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            DemoAcquire(sample_rate=rate)


def test_finite_capture_delivers_ordered_complete_batches_and_timestamps() -> None:
    acquire = DemoAcquire(channel_indices=(0, 2))
    acquire.timestamps_enabled = True
    acquire.prepare()
    acquire.start()
    destination = destination_for(acquire)

    first = acquire.read_into(destination)

    assert first is not None
    np.testing.assert_array_equal(destination, EXPECTED)
    np.testing.assert_allclose(first.timestamps, [0.0, 0.004])

    second = acquire.read_into(destination)

    assert second is not None
    np.testing.assert_array_equal(destination, EXPECTED + 200)
    np.testing.assert_allclose(second.timestamps, [0.008, 0.012])

    # Previously returned timestamp storage remains valid.
    np.testing.assert_allclose(first.timestamps, [0.0, 0.004])

    assert acquire.buffers_acquired == 2
    assert acquire.read_into(destination) is None
    assert acquire.read_into(destination) is None
    assert acquire.is_active

    acquire.stop()

    assert not acquire.is_active
    assert acquire.buffers_acquired == 2


def test_success_without_timestamps_is_distinct_from_completion() -> None:
    acquire = DemoAcquire()
    acquire.buffers_per_acquisition = 1
    acquire.prepare()
    acquire.start()
    destination = destination_for(acquire)

    info = acquire.read_into(destination)

    assert info is not None
    assert info.timestamps is None
    assert acquire.read_into(destination) is None

    acquire.stop()


def test_unlimited_capture_and_reuse_after_stop() -> None:
    acquire = DemoAcquire(channel_indices=(0, 2))
    acquire.buffers_per_acquisition = -1

    for _ in range(2):
        acquire.prepare()
        assert acquire.buffers_acquired == 0

        acquire.start()
        destination = destination_for(acquire)

        for batch in range(5):
            assert acquire.read_into(destination) is not None
            np.testing.assert_array_equal(
                destination,
                EXPECTED + 200 * batch,
            )

        acquire.stop()
        assert acquire.buffers_acquired == 5


def test_destination_validation_happens_before_consuming_data() -> None:
    class Probe(DemoAcquire):
        def _wait_for_channel(
            self,
            channel: int,
            timeout: float | None,
        ) -> None:
            raise AssertionError(
                "Invalid destination reached the transfer hook."
            )

    acquire = Probe()
    acquire.prepare()
    acquire.start()

    readonly = destination_for(acquire)
    readonly.flags.writeable = False

    invalid_arrays = [
        np.empty((1, 4, 2), dtype=np.int16),
        np.empty((2, 4, 2), dtype=np.float64),
        np.empty((2, 8, 2), dtype=np.int16)[:, ::2, :],
        readonly,
    ]

    for destination in invalid_arrays:
        with pytest.raises(ValueError):
            acquire.read_into(destination)

    with pytest.raises(TypeError):
        acquire.read_into(cast(SampleArray, object()))

    assert acquire.buffers_acquired == 0

    acquire.stop()


def test_timeout_validation_and_ready_zero_timeout() -> None:
    acquire = DemoAcquire()
    acquire.prepare()
    acquire.start()
    destination = destination_for(acquire)

    for timeout in (
        units.Time(-1),
        units.Time(math.nan),
        units.Time(math.inf),
    ):
        with pytest.raises(ValueError):
            acquire.read_into(destination, timeout=timeout)

    with pytest.raises(TypeError):
        acquire.read_into(
            destination,
            timeout=cast(units.Time, 1.0),
        )

    assert acquire.buffers_acquired == 0
    assert acquire.read_into(
        destination,
        timeout=units.Time(0),
    ) is not None

    acquire.stop()


def test_timeout_retry_preserves_partial_batch_and_accepts_new_destination() -> None:
    class TimeoutOnce(DemoAcquire):
        def __init__(self) -> None:
            super().__init__(channel_indices=(0, 2))
            self.visited: list[int] = []
            self.failed = False

        def _wait_for_channel(
            self,
            channel: int,
            timeout: float | None,
        ) -> None:
            self.visited.append(channel)

            if channel == 2 and not self.failed:
                self.failed = True
                raise TimeoutError("Channel is not ready.")

    acquire = TimeoutOnce()
    acquire.prepare()
    acquire.start()

    abandoned = destination_for(acquire)
    reference = weakref.ref(abandoned)

    with pytest.raises(TimeoutError):
        acquire.read_into(abandoned, timeout=units.Time(0))

    del abandoned

    assert reference() is None
    assert acquire.buffers_acquired == 0

    destination = destination_for(acquire)

    assert acquire.read_into(
        destination,
        timeout=units.Time(0),
    ) is not None
    assert acquire.visited == [0, 2, 2]
    np.testing.assert_array_equal(destination, EXPECTED)
    assert acquire.buffers_acquired == 1

    acquire.stop()


def test_waiting_budget_is_shared_across_channels() -> None:
    class Timed(DemoAcquire):
        def __init__(self) -> None:
            super().__init__(channel_indices=(0, 1, 2))
            self.now = 0.0
            self.budgets: list[float] = []

        def _clock(self) -> float:
            return self.now

        def _wait_for_channel(
            self,
            channel: int,
            timeout: float | None,
        ) -> None:
            assert timeout is not None
            self.budgets.append(timeout)

            if timeout < 0.4:
                self.now += timeout
                raise TimeoutError("Not enough waiting budget.")

            self.now += 0.4

    acquire = Timed()
    acquire.prepare()
    acquire.start()
    destination = destination_for(acquire)

    with pytest.raises(TimeoutError):
        acquire.read_into(destination, timeout=units.Time(1))

    assert acquire.buffers_acquired == 0

    assert acquire.read_into(
        destination,
        timeout=units.Time(1),
    ) is not None

    np.testing.assert_allclose(
        acquire.budgets,
        [1.0, 0.6, 0.2, 1.0],
    )
    assert acquire.buffers_acquired == 1

    acquire.stop()


def test_output_is_detached_from_reused_transfer_buffers() -> None:
    acquire = DemoAcquire(channel_indices=(0, 2))
    acquire.buffers_per_acquisition = -1
    acquire.prepare()
    acquire.start()

    first = destination_for(acquire)
    reference = weakref.ref(first)
    acquire.read_into(first)

    # Reuse the first internal transfer buffer.
    for _ in range(2):
        acquire.read_into(destination_for(acquire))

    np.testing.assert_array_equal(first, EXPECTED)

    del first
    assert reference() is None

    acquire.stop()


def test_partial_preparation_failure_can_be_cleaned_and_retried() -> None:
    class AllocationFailure(DemoAcquire):
        def __init__(self) -> None:
            super().__init__()
            self.references: list[
                weakref.ReferenceType[SampleArray]
            ] = []
            self.fail_once = True

        def _allocate_buffer(
            self,
            layout: CaptureLayout,
        ) -> SampleArray:
            if self.references and self.fail_once:
                self.fail_once = False
                raise MemoryError("Allocation failed.")

            buffer = super()._allocate_buffer(layout)
            self.references.append(weakref.ref(buffer))
            return buffer

    acquire = AllocationFailure()

    with pytest.raises(MemoryError):
        acquire.prepare()

    with pytest.raises(RuntimeError):
        _ = acquire.layout

    acquire.stop()
    acquire.stop()

    assert all(
        reference() is None
        for reference in acquire.references
    )

    acquire.prepare()
    acquire.start()

    assert acquire.read_into(destination_for(acquire)) is not None

    acquire.stop()


def test_failed_start_is_cleaned_even_though_is_active_is_false() -> None:
    class StartFailure(DemoAcquire):
        def __init__(self) -> None:
            super().__init__()
            self.resource_open = False
            self.fail_once = True

        def _start_capture(self) -> None:
            self.resource_open = True

            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("Start failed.")

        def _stop_capture(self) -> None:
            self.resource_open = False

    acquire = StartFailure()
    acquire.prepare()

    with pytest.raises(RuntimeError):
        acquire.start()

    assert not acquire.is_active
    assert acquire.resource_open

    with pytest.raises(RuntimeError):
        acquire.start()

    acquire.stop()
    assert not acquire.resource_open

    acquire.prepare()
    acquire.start()
    acquire.stop()


def test_failed_stop_still_releases_transfer_buffers_and_invalidates_layout() -> None:
    class StopFailure(DemoAcquire):
        def __init__(self) -> None:
            super().__init__()
            self.references: list[
                weakref.ReferenceType[SampleArray]
            ] = []
            self.fail_once = True

        def _allocate_buffer(
            self,
            layout: CaptureLayout,
        ) -> SampleArray:
            buffer = super()._allocate_buffer(layout)
            self.references.append(weakref.ref(buffer))
            return buffer

        def _stop_capture(self) -> None:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("Stop failed.")

    acquire = StopFailure()
    acquire.prepare()
    acquire.start()

    with pytest.raises(RuntimeError):
        acquire.stop()

    assert not acquire.is_active

    with pytest.raises(RuntimeError):
        _ = acquire.layout

    assert all(
        reference() is None
        for reference in acquire.references
    )

    acquire.stop()