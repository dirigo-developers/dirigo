"""Test capture layouts and timestamp ownership without device hardware."""

from dataclasses import replace
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from dirigo.hardware.capture import CaptureInfo, CaptureLayout


@pytest.fixture
def layout() -> CaptureLayout:
    """Describe three records with an explicitly ordered pair of channels."""
    return CaptureLayout(
        shape=(3, 64, 2),
        channel_indices=(2, 0),
        dtype=np.dtype(np.int16),
    )


@pytest.mark.parametrize("dtype", [np.dtype(np.int16), np.dtype(np.uint16)])
def test_layout_preserves_channel_order_and_storage_dtype(
    layout: CaptureLayout,
    dtype: np.dtype[np.integer[Any]],
) -> None:
    """Describe record geometry without sorting channels or changing signedness."""
    configured = replace(layout, dtype=dtype)

    assert configured.shape == (3, 64, 2)
    assert configured.records_per_batch == 3
    assert configured.record_length == 64
    assert configured.channel_count == 2
    assert configured.channel_indices == (2, 0)
    assert configured.dtype == dtype


@pytest.mark.parametrize(
    ("changes", "error_type", "message"),
    [
        ({"shape": [3, 64, 2]}, TypeError, "shape must be a tuple"),
        ({"shape": (3, 64)}, ValueError, "three dimensions"),
        ({"shape": (3, 0, 2)}, ValueError, "dimensions must be positive"),
        ({"shape": (3, 64.0, 2)}, TypeError, "dimensions must be integers"),
        ({"shape": (3, True, 2)}, TypeError, "dimensions must be integers"),
        ({"channel_indices": [2, 0]}, TypeError, "must be a tuple"),
        ({"channel_indices": (2,)}, ValueError, "channel count"),
        ({"channel_indices": (-1, 0)}, ValueError, "nonnegative"),
        ({"channel_indices": (0, 0)}, ValueError, "distinct"),
        ({"channel_indices": (0.0, 1)}, TypeError, "must contain integers"),
        ({"channel_indices": (False, 1)}, TypeError, "must contain integers"),
        ({"dtype": np.int16}, TypeError, "must be a NumPy dtype"),
        ({"dtype": np.dtype(np.float64)}, TypeError, "integer sample storage"),
        (
            {"dtype": np.dtype(np.int16).newbyteorder("S")},
            ValueError,
            "native byte order",
        ),
    ],
)
def test_layout_rejects_inconsistent_or_invalid_fields(
    layout: CaptureLayout,
    changes: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject invalid dimensions, channel mappings, and unsupported storage.

    Swap the native integer dtype's byte order to exercise the foreign-endian
    check on either little-endian or big-endian hosts.
    """
    with pytest.raises(error_type, match=message):
        replace(layout, **changes)


def test_capture_info_defaults_to_omitted_timestamps() -> None:
    """Allow a complete batch to carry no optional timestamp vector."""
    assert CaptureInfo().timestamps is None


def test_capture_info_detaches_timestamps_and_exposes_read_only_storage() -> None:
    """Keep record times independent of a reused SDK or caller array.

    Mutate the supplied float64 array after constructing the result. Its
    recorded values must remain unchanged, and callers must not be able to
    enable writes on the returned view of immutable timestamp storage.
    """
    original = np.array([0.1, 0.2], dtype=np.float64)
    info = CaptureInfo(timestamps=original)
    original[:] = 99.0

    timestamps = info.timestamps
    assert timestamps is not None
    assert timestamps.dtype == np.float64
    assert not timestamps.flags.writeable
    np.testing.assert_array_equal(timestamps, [0.1, 0.2])

    with pytest.raises(ValueError, match="cannot set"):
        timestamps.setflags(write=True)


def test_capture_info_accepts_later_batches_and_equal_timestamp_ticks() -> None:
    """Allow positive batch origins and ties from finite timestamp resolution."""
    info = CaptureInfo([1.0, 1.0, 1.002])

    timestamps = info.timestamps
    assert timestamps is not None
    np.testing.assert_array_equal(timestamps, [1.0, 1.0, 1.002])


@pytest.mark.parametrize(
    ("timestamps", "error_type", "message"),
    [
        (0.0, ValueError, "one-dimensional"),
        ([[0.0, 1.0]], ValueError, "one-dimensional"),
        ([], ValueError, "must not be empty"),
        ([False, True], TypeError, "real numeric values"),
        (["0", "1"], TypeError, "real numeric values"),
        ([0j, 1j], TypeError, "real numeric values"),
        ([float("nan")], ValueError, "finite"),
        ([float("inf")], ValueError, "finite"),
        ([-0.1, 0.0], ValueError, "nonnegative"),
        ([0.2, 0.1], ValueError, "nondecreasing"),
    ],
)
def test_capture_info_rejects_invalid_timestamps(
    timestamps: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Require a nonempty, finite, ordered vector of real record times."""
    with pytest.raises(error_type, match=message):
        CaptureInfo(cast(npt.ArrayLike, timestamps))
