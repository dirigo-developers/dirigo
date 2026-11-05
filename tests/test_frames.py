"""Test immutable image-frame coordinates and pipeline metadata compatibility."""

from dataclasses import FrozenInstanceError, replace

import pytest

from dirigo.frames import FrameMetadata
from dirigo.pipeline import Metadata, Product


def test_frame_metadata_defaults_to_unspecified_indices() -> None:
    """Default indices remain unspecified and the record is valid Product metadata."""
    metadata = FrameMetadata()
    product = Product(False, metadata=metadata)

    assert isinstance(metadata, Metadata)
    assert product.metadata is metadata
    assert metadata.sequence_index is None
    assert metadata.time_index is None
    assert metadata.z_index is None


@pytest.mark.parametrize(
    ("time_index", "z_index"),
    [(0, None), (None, 0), (3, 5)],
)
def test_frame_metadata_preserves_supplied_indices(
    time_index: int | None,
    z_index: int | None,
) -> None:
    """Preserve explicit indices, including zero and independently missing axes."""
    metadata = FrameMetadata(
        sequence_index=17,
        time_index=time_index,
        z_index=z_index,
    )

    assert metadata.sequence_index == 17
    assert metadata.time_index == time_index
    assert metadata.z_index == z_index


def test_frame_metadata_is_immutable_and_replaceable() -> None:
    """Replacing coordinates creates a new record without changing the original."""
    original = FrameMetadata(sequence_index=7, time_index=1, z_index=2)

    with pytest.raises(FrozenInstanceError):
        original.z_index = 8  # ty: ignore[invalid-assignment]

    updated = replace(original, time_index=4, z_index=None)

    assert original == FrameMetadata(
        sequence_index=7,
        time_index=1,
        z_index=2,
    )
    assert updated == FrameMetadata(
        sequence_index=7,
        time_index=4,
        z_index=None,
    )


@pytest.mark.parametrize("field", ["time_index", "z_index"])
@pytest.mark.parametrize("value", [True, False, 1.5, "2"])
def test_frame_coordinates_reject_non_integer_values(
    field: str,
    value: object,
) -> None:
    """Time and Z indices reject booleans and other non-integer values."""
    with pytest.raises(TypeError, match=field):
        FrameMetadata(**{field: value})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("field", ["time_index", "z_index"])
def test_frame_coordinates_reject_negative_values(field: str) -> None:
    """Time and Z indices must be nonnegative when specified."""
    with pytest.raises(ValueError, match=field):
        FrameMetadata(**{field: -1})
