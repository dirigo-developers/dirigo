"""Test product construction and immutable publication metadata.

Products preserve their supplied payload and metadata. Their fields cannot be
reassigned, and metadata changes require constructing a replacement record.
Delivery, buffer ownership, and worker behavior are tested in other modules.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from dirigo.pipeline import Metadata, Product


def test_product_preserves_supplied_payload() -> None:
    """Construction stores the original payload without copying it."""
    payload = bytearray(b"data")

    product = Product(payload)

    assert product.payload is payload


def test_product_defaults_to_empty_metadata() -> None:
    """Omitting metadata leaves the sequence index unspecified."""
    product = Product(False)

    assert product.metadata == Metadata()
    assert product.metadata.sequence_index is None


def test_product_preserves_supplied_metadata() -> None:
    """Explicit metadata is attached without copying or modifying it."""
    metadata = Metadata(sequence_index=35)

    product = Product(False, metadata=metadata)

    assert product.metadata is metadata
    assert product.metadata.sequence_index == 35


def test_product_fields_cannot_be_reassigned() -> None:
    """A product's payload and metadata references are immutable."""
    metadata = Metadata(sequence_index=17)
    product = Product(False, metadata=metadata)

    with pytest.raises(FrozenInstanceError):
        product.payload = True  # ty: ignore[invalid-assignment]

    with pytest.raises(FrozenInstanceError):
        product.metadata = Metadata()  # ty: ignore[invalid-assignment]

    assert product.payload is False
    assert product.metadata is metadata


def test_metadata_is_immutable() -> None:
    """A shared metadata record cannot be changed through assignment."""
    metadata = Metadata(sequence_index=97)

    with pytest.raises(FrozenInstanceError):
        metadata.sequence_index = 1  # ty: ignore[invalid-assignment]

    assert metadata.sequence_index == 97


@pytest.mark.parametrize("sequence_index", [30, None])
def test_metadata_can_be_explicitly_updated(
    sequence_index: int | None,
) -> None:
    """Replacing metadata can update or clear an index without changing it."""
    original = Metadata(sequence_index=7)

    updated = replace(original, sequence_index=sequence_index)

    assert updated is not original
    assert updated == Metadata(sequence_index=sequence_index)
    assert original.sequence_index == 7


def test_product_can_be_replaced_with_different_metadata() -> None:
    """A replacement product shares the payload and leaves the original intact."""
    payload = bytearray(b"data")
    original_metadata = Metadata(sequence_index=127)
    updated_metadata = replace(original_metadata, sequence_index=128)
    original = Product(payload, metadata=original_metadata)

    updated = replace(original, metadata=updated_metadata)

    assert updated is not original
    assert updated.payload is payload
    assert updated.metadata is updated_metadata
    assert original.payload is payload
    assert original.metadata is original_metadata
    assert original.metadata.sequence_index == 127


@pytest.mark.parametrize("invalid_metadata", [{}, {"sequence_index": 1}, None])
def test_product_rejects_invalid_metadata(invalid_metadata: object) -> None:
    """Explicit metadata must be a Metadata record, including for empty values."""
    with pytest.raises(TypeError, match="metadata must be Metadata"):
        Product(
            False,
            metadata=invalid_metadata,  # ty: ignore[invalid-argument-type]
        )
