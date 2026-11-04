"""Test metadata across publication and producer-lease lifetimes.

Verify that fanout shares immutable metadata, buffer reuse starts with empty
metadata, and publication captures the record supplied by the producer.

Failed publication must preserve editable lease metadata. Released or
transferred leases must reject further metadata access.

Product construction and metadata handling inside workers are tested separately.
"""

from dataclasses import replace

import pytest

from dirigo.pipeline import Metadata, OperationCancelled, Product
from dirigo.pipeline.pool import Pool
from dirigo.pipeline.publisher import Publisher


def test_unpooled_fanout_shares_product_and_immutable_metadata() -> None:
    """Every subscriber receives the same Product and metadata record.

    Publication must preserve the supplied objects rather than create separate
    copies for each subscriber. Sharing metadata is safe because Metadata
    records are immutable.
    """
    publisher = Publisher[bool]()
    subscriptions = [
        publisher.subscribe(),
        publisher.subscribe(),
    ]
    metadata = Metadata(sequence_index=1)
    product = Product(False, metadata=metadata)

    publisher.publish(product, timeout=0.0)

    for subscription in subscriptions:
        with subscription.receive(timeout=0.0) as received:
            assert received is product
            assert received.metadata is metadata


def test_reusing_pooled_storage_does_not_reuse_publication_metadata() -> None:
    """Reusing a buffer must not carry metadata into the next acquisition.

    Publish twice with different sequence indices, then publish without setting
    metadata. A one-buffer pool ensures that every publication uses the same
    storage.

    Each fresh lease must start with empty metadata. Previously received
    metadata records must retain their values after the buffer is reused.
    """
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()
    records = [
        Metadata(sequence_index=0),
        Metadata(sequence_index=1),
        Metadata(),
    ]
    snapshots: list[Metadata] = []
    buffer_ids: list[int] = []

    for index, metadata in enumerate(records):
        with pool.acquire(timeout=0.0) as lease:
            assert lease.metadata == Metadata()

            # Leave the final acquisition's default metadata untouched.
            if index < 2:
                lease.metadata = metadata

            publisher.publish_lease(lease, timeout=0.0)

        with subscription.receive(timeout=0.0) as product:
            snapshots.append(product.metadata)
            buffer_ids.append(id(product.payload))

            assert product.metadata == metadata

            if index < 2:
                assert product.metadata is metadata

    assert len(set(buffer_ids)) == 1
    assert snapshots == records
    assert snapshots[0] == Metadata(sequence_index=0)


def test_publication_retry_can_supply_different_metadata() -> None:
    """A publication retry captures the producer's revised metadata record.

    Fill the subscriber queue so the first publication attempt times out.
    The producer must retain its buffer and metadata after that failure.

    After consuming the blocker, replace the lease metadata and retry. The
    delivered product must contain the revised record and the original buffer.
    """
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()

    publisher.publish(Product(bytearray(4)), timeout=0.0)

    attempted = Metadata(sequence_index=1)
    accepted = Metadata(sequence_index=2)

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        lease.metadata = attempted

        with pytest.raises(TimeoutError):
            publisher.publish_lease(lease, timeout=0.0)

        assert lease.payload is original
        assert lease.metadata is attempted

        with subscription.receive(timeout=0.0):
            pass

        lease.metadata = accepted
        publisher.publish_lease(lease, timeout=0.0)

    with subscription.receive(timeout=0.0) as product:
        assert product.metadata is accepted
        assert product.payload is original

    with pool.acquire(timeout=0.0) as lease:
        assert lease.payload is original


@pytest.mark.parametrize("invalid", [{"sequence_index": 1}, None])
def test_invalid_metadata_does_not_consume_producer_lease(
    invalid: object,
) -> None:
    """Rejecting invalid metadata preserves the producer's existing lease state.

    Assign valid metadata, then attempt to replace it with an invalid value.
    Rejection must preserve both the buffer and the original metadata record,
    without publishing a delivery.

    Leaving the acquisition context must still return the buffer normally.
    """
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        metadata = Metadata(sequence_index=7)
        lease.metadata = metadata

        with pytest.raises(TypeError, match="metadata must be Metadata"):
            lease.metadata = invalid  # ty: ignore[invalid-assignment]

        assert lease.payload is original
        assert lease.metadata is metadata

        with pytest.raises(TimeoutError), subscription.receive(timeout=0.0):
            pass

    with pool.acquire(timeout=0.0) as lease:
        assert lease.payload is original


@pytest.mark.parametrize("expiration", ["release", "publication"])
def test_expired_lease_rejects_metadata_access_and_updates(
    expiration: str,
) -> None:
    """Released and transferred leases reject metadata reads and assignments.

    Publication without subscribers immediately recycles the buffer, allowing
    the same storage to be reacquired in either scenario.

    The new lease must have empty metadata. The expired lease's later context
    cleanup must remain harmless.
    """
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    publisher = Publisher[bytearray]()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        lease.metadata = Metadata(sequence_index=1)

        if expiration == "release":
            lease.release()
        else:
            publisher.publish_lease(lease, timeout=0.0)

        with pytest.raises(RuntimeError, match="released or transferred"):
            _ = lease.metadata

        with pytest.raises(RuntimeError, match="released or transferred"):
            lease.metadata = Metadata(sequence_index=9)

        with pool.acquire(timeout=0.0) as next_lease:
            assert next_lease.payload is original
            assert next_lease.metadata == Metadata()


@pytest.mark.parametrize("operation", ["close", "abort"])
def test_rejected_publication_preserves_editable_lease_metadata(
    operation: str,
) -> None:
    """Publisher closure or cancellation preserves editable producer metadata.

    Attempt publication after the publisher stops accepting products. Rejection
    must leave the lease valid and preserve its existing metadata record.

    The producer must still be able to replace that record. After releasing
    the lease, reacquiring the same buffer must yield empty metadata.
    """
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    publisher = Publisher[bytearray]()
    original_metadata = Metadata(sequence_index=3)
    expected_error = RuntimeError if operation == "close" else OperationCancelled

    with pool.acquire(timeout=0.0) as lease:
        original_buffer = lease.payload
        lease.metadata = original_metadata

        if operation == "close":
            publisher.close()
        else:
            publisher.abort()

        with pytest.raises(expected_error):
            publisher.publish_lease(lease, timeout=0.0)

        assert lease.payload is original_buffer
        assert lease.metadata is original_metadata

        updated = replace(lease.metadata, sequence_index=5)
        lease.metadata = updated

        assert lease.metadata is updated

    with pool.acquire(timeout=0.0) as fresh:
        assert fresh.payload is original_buffer
        assert fresh.metadata == Metadata()
