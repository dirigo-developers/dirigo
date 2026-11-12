"""Test angular waveform representation, ownership, and validation."""

from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from dirigo import units
from dirigo.hardware.waveform import AngularWaveform


def test_waveform_copies_strided_samples_and_normalizes_storage() -> None:
    """Keep independent contiguous float64 data from a strided float32 input.

    The source is a noncontiguous view into another array. Modifying that
    backing array after construction must not change the waveform.
    Sample values remain radians, with no scaling during normalization.
    """
    backing = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    source = backing[::2]
    rate = units.SampleRate("20 kS/s")

    waveform = AngularWaveform(source, sample_rate=rate)
    backing[:] = -99.0

    assert waveform.sample_rate is rate
    assert waveform.samples.dtype == np.dtype(np.float64)
    assert waveform.samples.flags.c_contiguous
    assert not np.shares_memory(waveform.samples, source)
    np.testing.assert_array_equal(waveform.samples, [0.0, 0.5, 1.0])


@pytest.mark.parametrize(
    "samples",
    [
        [0.25],
        np.array([0.25], dtype=np.float64),
        np.array([1], dtype=np.int16),
    ],
)
def test_waveform_accepts_single_sample_and_real_numeric_inputs(
    samples: npt.ArrayLike,
) -> None:
    """Accept nonempty sequences and real arrays, including one-sample buffers."""
    waveform = AngularWaveform(
        samples,
        sample_rate=units.SampleRate("1 kS/s"),
    )

    assert waveform.samples.shape == (1,)
    assert waveform.samples.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        waveform.samples,
        np.asarray(samples, dtype=np.float64),
    )


def test_waveform_samples_cannot_be_made_writable() -> None:
    """Protect stored samples through the public NumPy view.

    Direct assignment must fail. Re-enabling the write flag must also fail,
    so the exposed view cannot invalidate previously validated sample data.
    """
    waveform = AngularWaveform(
        [0.0, 0.25],
        sample_rate=units.SampleRate("1 kS/s"),
    )
    samples = waveform.samples

    assert not samples.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        samples[0] = 0.5

    with pytest.raises(ValueError, match="WRITEABLE"):
        samples.setflags(write=True)

    np.testing.assert_array_equal(waveform.samples, [0.0, 0.25])


def test_samples_access_returns_distinct_views_of_shared_storage() -> None:
    """Expose independent array headers without copying the sample data.

    Each access must return a distinct view while sharing the waveform's
    immutable storage. This preserves independent array metadata without
    allocating another sample buffer.
    """
    waveform = AngularWaveform(
        [0.0, 0.25],
        sample_rate=units.SampleRate("1 kS/s"),
    )

    first = waveform.samples
    second = waveform.samples

    assert first is not second
    assert np.shares_memory(first, second)
    assert first.shape == second.shape == (2,)
    np.testing.assert_array_equal(first, [0.0, 0.25])
    np.testing.assert_array_equal(second, first)


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.array([]), "must not be empty"),
        (np.array(0.0), "must be one-dimensional"),
        (np.zeros((2, 3)), "must be one-dimensional"),
    ],
)
def test_waveform_rejects_empty_or_non_vector_samples(
    samples: npt.ArrayLike,
    message: str,
) -> None:
    """Require a nonempty one-dimensional sample sequence."""
    with pytest.raises(ValueError, match=message):
        AngularWaveform(samples, sample_rate=units.SampleRate("1 kS/s"))


@pytest.mark.parametrize(
    "samples",
    [
        np.array([True, False]),
        np.array(["0.0", "0.25"]),
        np.array([0.0 + 0.25j]),
        np.array([0.0, 0.25], dtype=object),
    ],
)
def test_waveform_rejects_non_real_numeric_dtypes(
    samples: npt.ArrayLike,
) -> None:
    """Reject booleans, strings, complex values, and object arrays.

    These inputs must fail before float64 conversion could reinterpret their
    meaning, parse numeric strings, or discard imaginary components.
    """
    with pytest.raises(TypeError, match="real integer or floating"):
        AngularWaveform(samples, sample_rate=units.SampleRate("1 kS/s"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_waveform_rejects_nonfinite_samples(value: float) -> None:
    """Reject a nonfinite sample anywhere in an otherwise finite sequence."""
    with pytest.raises(ValueError, match="samples must be finite"):
        AngularWaveform(
            [0.0, value, 0.25],
            sample_rate=units.SampleRate("1 kS/s"),
        )


@pytest.mark.parametrize(
    ("rate", "error_type"),
    [
        (1000.0, TypeError),
        ("1 kS/s", TypeError),
        (units.Frequency("1 kHz"), TypeError),
        (units.SampleRate(0), ValueError),
        (units.SampleRate(-1), ValueError),
        (units.SampleRate(float("nan")), ValueError),
        (units.SampleRate(float("inf")), ValueError),
        (units.SampleRate(float("-inf")), ValueError),
    ],
)
def test_waveform_requires_a_finite_positive_sample_rate(
    rate: object,
    error_type: type[Exception],
) -> None:
    """Require an explicitly typed SampleRate with a finite positive value."""
    with pytest.raises(error_type, match="sample_rate"):
        AngularWaveform(
            [0.0],
            sample_rate=cast(units.SampleRate, rate),
        )
