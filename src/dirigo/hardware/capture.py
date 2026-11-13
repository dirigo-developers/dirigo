"""Descriptions and metadata for complete digitizer capture batches."""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from dirigo import units
from dirigo.hardware.timing import DigitalStartTrigger, ExternalSampleClock

type SampleArray = npt.NDArray[np.integer[Any]]


@dataclass(frozen=True, slots=True)
class CaptureLayout:
    """Describe the sample storage for one complete logical batch.

    shape is (records_per_batch, record_length, channel_count). All dimensions
    are positive. channel_indices lists distinct, nonnegative physical channel
    indices in the order represented by the final array axis.

    dtype is the exact native-endian integer storage dtype, independent of
    ADC resolution. Destinations used for capture must also be writable and
    C-contiguous; the capture session will validate those array properties.

    This description owns no acquisition resources and remains usable after
    the run that established it has closed.
    """

    shape: tuple[int, int, int]
    channel_indices: tuple[int, ...]
    dtype: np.dtype[np.integer[Any]]

    def __post_init__(self) -> None:
        """Validate dimensions, physical-channel order, and sample storage."""
        if not isinstance(self.shape, tuple):
            raise TypeError("shape must be a tuple.")
        if len(self.shape) != 3:
            raise ValueError("shape must have three dimensions.")
        if any(not isinstance(n, int) or isinstance(n, bool) for n in self.shape):
            raise TypeError("shape dimensions must be integers.")
        if any(n <= 0 for n in self.shape):
            raise ValueError("shape dimensions must be positive.")

        if not isinstance(self.channel_indices, tuple):
            raise TypeError("channel_indices must be a tuple.")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in self.channel_indices
        ):
            raise TypeError("channel_indices must contain integers.")
        if any(index < 0 for index in self.channel_indices):
            raise ValueError("channel_indices must be nonnegative.")
        if len(self.channel_indices) != self.channel_count:
            raise ValueError("channel_indices must match the shape's channel count.")
        if len(set(self.channel_indices)) != len(self.channel_indices):
            raise ValueError("channel_indices must be distinct.")

        if not isinstance(self.dtype, np.dtype):
            raise TypeError("dtype must be a NumPy dtype.")
        if self.dtype.kind not in "iu":
            raise TypeError("dtype must describe integer sample storage.")
        if not self.dtype.isnative:
            raise ValueError("dtype must use native byte order.")

    @property
    def records_per_batch(self) -> int:
        """Number of complete records represented by each channel."""
        return self.shape[0]

    @property
    def record_length(self) -> int:
        """Number of samples per record and channel."""
        return self.shape[1]

    @property
    def channel_count(self) -> int:
        """Number of channels represented by the final sample-array axis."""
        return self.shape[2]


@dataclass(frozen=True, slots=True, init=False, eq=False)
class CaptureInfo:
    """Own metadata accompanying one successfully transferred complete batch.

    timestamps contains first-sample times for the batch's records, in seconds
    relative to the first sample of the first record in the capture session.
    Values are finite, nonnegative, and nondecreasing; ties are permitted for
    clocks whose timestamp resolution is coarser than the record interval.

    Construction copies timestamps into immutable storage. Each timestamps
    access returns a read-only float64 view independent of SDK-buffer lifetime.
    None represents omitted timestamps. The capture session will validate the
    timestamp count against its layout and requested timestamp settings.
    """

    _timestamp_data: bytes | None = field(init=False, repr=False)

    def __init__(self, timestamps: npt.ArrayLike | None = None) -> None:
        """Copy an optional nonempty vector of real-valued record times."""
        if timestamps is None:
            object.__setattr__(self, "_timestamp_data", None)
            return

        values = np.asarray(timestamps)
        if values.ndim != 1:
            raise ValueError("timestamps must be one-dimensional.")
        if values.size == 0:
            raise ValueError("timestamps must not be empty.")
        if values.dtype.kind not in "iuf":
            raise TypeError("timestamps must contain real numeric values.")

        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("timestamps must be finite.")
        if np.any(values < 0):
            raise ValueError("timestamps must be nonnegative.")
        if np.any(values[1:] < values[:-1]):
            raise ValueError("timestamps must be nondecreasing.")

        object.__setattr__(self, "_timestamp_data", values.tobytes())

    @property
    def timestamps(self) -> npt.NDArray[np.float64] | None:
        """Read-only record times in seconds, or None when omitted."""
        if self._timestamp_data is None:
            return None
        return np.frombuffer(self._timestamp_data, dtype=np.float64)


@dataclass(frozen=True, slots=True, kw_only=True)
class CaptureSettings:
    """Request uniformly sampled records and complete-batch delivery.

    channel_indices selects zero-based physical inputs in delivered order.
    Records are consecutive windows of record_length samples; per-record
    triggering is not modeled here yet.

    batch_count=None requests unlimited delivery. External sample clocks and
    whole-run start triggers are optional capabilities that a backend may
    reject.
    """

    sample_rate: units.SampleRate
    record_length: int
    channel_indices: tuple[int, ...] = (0,)
    records_per_batch: int = 1
    batch_count: int | None = 1
    timestamps_enabled: bool = False
    external_clock: ExternalSampleClock | None = None
    start_trigger: DigitalStartTrigger | None = None

    def __post_init__(self) -> None:
        """Validate record geometry, delivery policy, and timing descriptions."""
        if not isinstance(self.sample_rate, units.SampleRate):
            raise TypeError("sample_rate must be a SampleRate.")
        if not math.isfinite(float(self.sample_rate)) or float(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be finite and positive.")

        if not isinstance(self.channel_indices, tuple):
            raise TypeError("channel_indices must be a tuple.")
        if not self.channel_indices:
            raise ValueError("At least one channel index is required.")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in self.channel_indices
        ):
            raise TypeError("channel_indices must contain integers.")
        if any(index < 0 for index in self.channel_indices):
            raise ValueError("channel_indices must be nonnegative.")
        if len(set(self.channel_indices)) != len(self.channel_indices):
            raise ValueError("channel_indices must be distinct.")

        for name in ("record_length", "records_per_batch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        if self.batch_count is not None:
            if not isinstance(self.batch_count, int) or isinstance(
                self.batch_count, bool
            ):
                raise TypeError("batch_count must be an integer or None.")
            if self.batch_count <= 0:
                raise ValueError("batch_count must be positive.")

        if not isinstance(self.timestamps_enabled, bool):
            raise TypeError("timestamps_enabled must be a bool.")

        if self.external_clock is not None:
            if not isinstance(self.external_clock, ExternalSampleClock):
                raise TypeError(
                    "external_clock must be an ExternalSampleClock or None."
                )
            if float(self.external_clock.max_rate) < float(self.sample_rate):
                raise ValueError(
                    "External clock max_rate must be at least sample_rate."
                )

        if self.start_trigger is not None and not isinstance(
            self.start_trigger, DigitalStartTrigger
        ):
            raise TypeError("start_trigger must be a DigitalStartTrigger or None.")
