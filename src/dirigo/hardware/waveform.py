"""Owned, uniformly sampled optical-angle waveforms."""

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from dirigo import units


@dataclass(frozen=True, slots=True, init=False)
class AngularWaveform:
    """A nonempty sequence of optical-angle commands at a uniform sample rate.

    Samples are stored as native float64 radians. Real integer and floating
    inputs are normalized without interpreting them as degrees or voltages.
    The sample at index i has nominal time i / float(sample_rate).

    Construction copies the samples into an independent immutable buffer.
    Each samples access exposes a fresh read-only NumPy view of that buffer.
    """

    sample_rate: units.SampleRate
    _data: bytes = field(init=False, repr=False)

    def __init__(
        self,
        samples: npt.ArrayLike,
        *,
        sample_rate: units.SampleRate,
    ) -> None:
        """Validate the sample rate and take an owned snapshot of the samples.

        Args:
            samples: A one-dimensional real numeric array or sequence in radians.
            sample_rate: The nominal uniform sampling rate.

        Raises:
            TypeError: sample_rate is not a SampleRate, or the sample dtype
                is not a real integer or floating type.
            ValueError: The rate is nonfinite or nonpositive, the samples
                are empty or not one-dimensional, or a stored sample is
                nonfinite.
        """
        if not isinstance(sample_rate, units.SampleRate):
            raise TypeError("sample_rate must be a SampleRate.")
        rate = float(sample_rate)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate must be finite and positive.")

        array = np.asarray(samples)
        if array.ndim != 1:
            raise ValueError("samples must be one-dimensional.")
        if array.size == 0:
            raise ValueError("samples must not be empty.")
        if array.dtype.kind not in "iuf":
            raise TypeError("samples must contain real integer or floating values.")

        data = np.asarray(array, dtype=np.float64).tobytes(order="C")
        stored = np.frombuffer(data, dtype=np.float64)
        if not np.isfinite(stored).all():
            raise ValueError("samples must be finite.")

        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "_data", data)

    @property
    def samples(self) -> npt.NDArray[np.float64]:
        """Read-only contiguous float64 optical angles in radians.

        The returned view shares immutable storage with the waveform, but
        its array metadata is independent of views returned by other calls.
        """
        return np.frombuffer(self._data, dtype=np.float64)
