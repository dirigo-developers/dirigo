"""A deterministic acquisition demo with immediate data availability.

Samples are:
    (global_record * 100 + sample * 10 + physical_channel) % 32768

Records are contiguous at the fixed sample rate supplied during construction.
Protected transfer hooks allow deterministic timing and failure tests.
"""

from __future__ import annotations

import math
from time import monotonic

import numpy as np

from dirigo import units
from dirigo.digitizer import (
    Acquire,
    CaptureInfo,
    CaptureLayout,
    SampleArray,
)


class DemoAcquire(Acquire):
    def __init__(
        self,
        *,
        channel_indices: tuple[int, ...] = (0, 1),
        sample_rate: units.SampleRate | str | float = "1 kS/s",
    ) -> None:
        channels = tuple(channel_indices)

        for channel in channels:
            if not isinstance(channel, int) or isinstance(channel, bool):
                raise TypeError("Channel indices must be integers.")
            if channel < 0:
                raise ValueError("Channel indices must be nonnegative.")

        if len(set(channels)) != len(channels):
            raise ValueError("Channel indices must be distinct.")

        rate = units.SampleRate(sample_rate)
        if not math.isfinite(float(rate)) or float(rate) <= 0:
            raise ValueError("Sample rate must be finite and positive.")

        self._channel_indices = channels
        self._sample_rate = rate

        self._trigger_delay = 0
        self._record_length = 4
        self._records_per_buffer = 2
        self._buffers_per_acquisition = 2
        self._buffers_allocated = 2
        self._timestamps_enabled = False

        self._layout: CaptureLayout | None = None
        self._buffers: list[SampleArray] = []
        self._active = False
        self._start_attempted = False
        self._buffers_acquired = 0
        self._next_channel = 0

    def _ensure_idle(self) -> None:
        if self._layout is not None or self._buffers:
            raise RuntimeError(
                "Call stop() before changing or preparing settings."
            )

    @staticmethod
    def _integer(value: int, name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer.")
        return value

    def _positive(self, value: int, name: str) -> int:
        self._ensure_idle()
        value = self._integer(value, name)

        if value < 1:
            raise ValueError(f"{name} must be positive.")

        return value

    @property
    def trigger_delay(self) -> int:
        return self._trigger_delay

    @trigger_delay.setter
    def trigger_delay(self, value: int) -> None:
        self._ensure_idle()
        value = self._integer(value, "trigger_delay")

        step = (
            self.pre_trigger_delay_step
            if value < 0
            else self.post_trigger_delay_step
        )

        if not self.trigger_delay_range.within_range(value) or value % step:
            raise ValueError(
                "Trigger delay does not satisfy its limits and step."
            )

        self._trigger_delay = value

    @property
    def trigger_delay_range(self) -> units.IntRange:
        return units.IntRange(-16, 16)

    @property
    def pre_trigger_delay_step(self) -> int:
        return 2

    @property
    def post_trigger_delay_step(self) -> int:
        return 4

    @property
    def record_length(self) -> int:
        return self._record_length

    @record_length.setter
    def record_length(self, length: int) -> None:
        length = self._positive(length, "record_length")

        if (
            length < self.record_length_minimum
            or length % self.record_length_step
        ):
            raise ValueError(
                "Record length does not satisfy its minimum and step."
            )

        self._record_length = length

    @property
    def record_length_minimum(self) -> int:
        return 4

    @property
    def record_length_step(self) -> int:
        return 4

    @property
    def records_per_buffer(self) -> int:
        return self._records_per_buffer

    @records_per_buffer.setter
    def records_per_buffer(self, records: int) -> None:
        self._records_per_buffer = self._positive(
            records,
            "records_per_buffer",
        )

    @property
    def buffers_per_acquisition(self) -> int:
        return self._buffers_per_acquisition

    @buffers_per_acquisition.setter
    def buffers_per_acquisition(self, buffers: int) -> None:
        self._ensure_idle()
        buffers = self._integer(buffers, "buffers_per_acquisition")

        if buffers != -1 and buffers < 1:
            raise ValueError(
                "Batch count must be positive, or -1 for unlimited."
            )

        self._buffers_per_acquisition = buffers

    @property
    def buffers_allocated(self) -> int:
        return self._buffers_allocated

    @buffers_allocated.setter
    def buffers_allocated(self, buffers: int) -> None:
        self._buffers_allocated = self._positive(
            buffers,
            "buffers_allocated",
        )

    @property
    def timestamps_enabled(self) -> bool:
        return self._timestamps_enabled

    @timestamps_enabled.setter
    def timestamps_enabled(self, enable: bool) -> None:
        self._ensure_idle()

        if not isinstance(enable, bool):
            raise TypeError("timestamps_enabled must be a boolean.")

        self._timestamps_enabled = enable

    @property
    def buffers_acquired(self) -> int:
        return self._buffers_acquired

    @property
    def n_channels_enabled(self) -> int:
        return len(self._channel_indices)

    @property
    def layout(self) -> CaptureLayout:
        layout = self._layout

        if layout is None:
            raise RuntimeError("Acquisition is not prepared.")

        return layout

    @property
    def is_active(self) -> bool:
        return self._active

    def prepare(self) -> None:
        self._ensure_idle()

        if not self.n_channels_enabled:
            raise ValueError("At least one channel must be enabled.")

        layout = CaptureLayout(
            shape=(
                self._records_per_buffer,
                self._record_length,
                self.n_channels_enabled,
            ),
            channel_indices=self._channel_indices,
            dtype=np.dtype(np.int16),
        )

        for _ in range(self._buffers_allocated):
            self._buffers.append(self._allocate_buffer(layout))

        self._layout = layout
        self._buffers_acquired = 0
        self._next_channel = 0
        self._start_attempted = False

    def start(self) -> None:
        if self._layout is None or self._start_attempted:
            raise RuntimeError(
                "Acquisition must be prepared and not started."
            )

        # A failed start still requires stop() before another attempt.
        self._start_attempted = True
        self._start_capture()
        self._active = True

    def read_into(
        self,
        destination: SampleArray,
        *,
        timeout: units.Time | None = None,
    ) -> CaptureInfo | None:
        if not self._active:
            raise RuntimeError("Acquisition has not started.")

        layout = self.layout

        if not isinstance(destination, np.ndarray):
            raise TypeError("destination must be a NumPy array.")

        if (
            destination.shape != layout.shape
            or destination.dtype != layout.dtype
            or not destination.flags.c_contiguous
            or not destination.flags.writeable
        ):
            raise ValueError(
                "destination must match the writable contiguous layout."
            )

        seconds: float | None = None

        if timeout is not None:
            if not isinstance(timeout, units.Time):
                raise TypeError("timeout must be Time or None.")

            seconds = float(timeout)

            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(
                    "timeout must be finite and nonnegative."
                )

        if (
            self._buffers_per_acquisition != -1
            and self._buffers_acquired >= self._buffers_per_acquisition
        ):
            return None

        deadline = None if seconds is None else self._clock() + seconds
        pending = self._buffers[
            self._buffers_acquired % len(self._buffers)
        ]
        first_record = self._buffers_acquired * layout.shape[0]

        while self._next_channel < len(layout.channel_indices):
            channel = layout.channel_indices[self._next_channel]
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - self._clock())
            )

            self._wait_for_channel(channel, remaining)

            records = (
                np.arange(layout.shape[0], dtype=np.int64)
                + first_record
            )
            samples = np.arange(layout.shape[1], dtype=np.int64)

            pending[:, :, self._next_channel] = (
                records[:, None] * 100
                + samples[None, :] * 10
                + channel
            ) % 32768

            self._next_channel += 1

        np.copyto(destination, pending)

        timestamps = None
        if self._timestamps_enabled:
            period = layout.shape[1] / float(self._sample_rate)
            timestamps = (
                np.arange(layout.shape[0], dtype=np.float64)
                + first_record
            ) * period

        info = CaptureInfo(timestamps=timestamps)
        self._buffers_acquired += 1
        self._next_channel = 0

        return info

    def stop(self) -> None:
        try:
            self._stop_capture()
        finally:
            self._active = False
            self._start_attempted = False
            self._layout = None
            self._buffers.clear()
            self._next_channel = 0

    def _allocate_buffer(self, layout: CaptureLayout) -> SampleArray:
        return np.empty(layout.shape, dtype=layout.dtype)

    def _start_capture(self) -> None:
        """Simulate starting capture; tests can inject failure here."""
        return None

    def _stop_capture(self) -> None:
        """Simulate stopping capture; tests can inject failure here."""
        return None

    def _wait_for_channel(
        self,
        channel: int,
        timeout: float | None,
    ) -> None:
        """Default data is immediately ready; tests can simulate a wait."""
        return None

    def _clock(self) -> float:
        return monotonic()
