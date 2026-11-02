from dataclasses import FrozenInstanceError, replace
from typing import Any, Never

import pytest

from dirigo.exceptions import EndOfStream, OperationCancelled
from dirigo.pipeline import Pipeline
from dirigo.pool import Pool
from dirigo.product import Metadata, Product
from dirigo.publisher import Publisher
from dirigo.worker import Worker

from .stubs import StubWorker


def test_product_defaults_to_empty_metadata() -> None:
    product = Product(False)

    assert product.metadata == Metadata()
    assert product.metadata.sequence_index is None


def test_metadata_is_immutable_and_can_be_explicitly_updated() -> None:
    original = Metadata(sequence_index=7)
    updated = replace(original, sequence_index=8)
    cleared = replace(original, sequence_index=None)

    assert original == Metadata(sequence_index=7)
    assert updated == Metadata(sequence_index=8)
    assert cleared == Metadata()

    with pytest.raises(FrozenInstanceError):
        original.sequence_index = 9  # ty: ignore[invalid-assignment]


def test_unpooled_fanout_shares_product_and_immutable_metadata() -> None:
    publisher = Publisher[bool]()
    subscriptions = [publisher.subscribe(), publisher.subscribe()]
    metadata = Metadata(sequence_index=1)
    product = Product(False, metadata=metadata)

    publisher.publish(product, timeout=0.0)

    for subscription in subscriptions:
        with subscription.receive(timeout=0.0) as received:
            assert received is product
            assert received.metadata is metadata


def test_reusing_pooled_storage_does_not_reuse_publication_metadata() -> None:
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
def test_invalid_metadata_does_not_consume_producer_lease(invalid: object) -> None:
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


def test_product_rejects_mutable_mapping_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be Metadata"):
        Product(False, metadata={})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("expiration", ["release", "publication"])
def test_expired_lease_rejects_metadata_access_and_updates(expiration: str) -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    publisher = Publisher[bytearray]()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        lease.metadata = Metadata(sequence_index=1)

        if expiration == "release":
            lease.release()
        else:
            # Without subscribers, publication immediately returns the buffer.
            publisher.publish_lease(lease, timeout=0.0)

        with pytest.raises(RuntimeError, match="released or transferred"):
            _ = lease.metadata

        with pytest.raises(RuntimeError, match="released or transferred"):
            lease.metadata = Metadata(sequence_index=9)

        with pool.acquire(timeout=0.0) as next_lease:
            assert next_lease.payload is original
            assert next_lease.metadata == Metadata()


@pytest.mark.parametrize("operation", ["close", "abort"])
def test_rejected_publication_preserves_editable_lease_metadata(operation: str) -> None:
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


class MetadataSource(StubWorker[Never, bytearray]):
    def __init__(self) -> None:
        super().__init__()
        self.sent_metadata: list[Metadata] = []

    def _prepare(self) -> None:
        self._init_pool(capacity=1, factory=lambda: bytearray(4))

    def _work(self) -> None:
        for sequence_index, value in enumerate((b"one!", b"two!")):
            metadata = Metadata(sequence_index=sequence_index)
            self.sent_metadata.append(metadata)

            with self._acquire(timeout=2.0) as lease:
                lease.payload[:] = value
                lease.metadata = metadata
                self._publish(lease, timeout=2.0)


class MetadataCollector(StubWorker[bytearray, Never]):
    def __init__(self, *, upstream: Worker[Any, bytearray]) -> None:
        super().__init__(upstream=upstream)
        self.records: list[tuple[bytes, Metadata]] = []
        self.buffer_ids: list[int] = []
        self.product_ids: list[int] = []

    def _work(self) -> None:
        try:
            while True:
                with self._receive(timeout=2.0) as product:
                    self.records.append((bytes(product.payload), product.metadata))
                    self.buffer_ids.append(id(product.payload))
                    self.product_ids.append(id(product))
        except EndOfStream:
            return


def _abort_and_join(pipeline: Pipeline) -> None:
    pipeline.abort()
    for worker in pipeline._launched:
        worker._wait(timeout=2.0)


@pytest.mark.parametrize("change_sequence", [False, True])
def test_processor_explicitly_inherits_or_updates_metadata(
    change_sequence: bool,
) -> None:
    class Processor(StubWorker[bytearray, bytearray]):
        def _prepare(self) -> None:
            self._init_pool(capacity=1, factory=lambda: bytearray(4))

        def _work(self) -> None:
            try:
                while True:
                    with (
                        self._receive(timeout=2.0) as incoming,
                        self._acquire(timeout=2.0) as outgoing,
                    ):
                        outgoing.payload[:] = incoming.payload.upper()

                        sequence_index = incoming.metadata.sequence_index
                        assert sequence_index is not None
                        outgoing.metadata = (
                            replace(
                                incoming.metadata,
                                sequence_index=sequence_index + 100,
                            )
                            if change_sequence
                            else incoming.metadata
                        )
                        self._publish(outgoing, timeout=2.0)
            except EndOfStream:
                return

    source = MetadataSource()
    processor = Processor(upstream=source)
    sink = MetadataCollector(upstream=processor)
    pipeline = Pipeline(sink)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        offset = 100 if change_sequence else 0
        assert sink.records == [
            (b"ONE!", Metadata(sequence_index=offset)),
            (b"TWO!", Metadata(sequence_index=offset + 1)),
        ]
        assert source.sent_metadata == [
            Metadata(sequence_index=0),
            Metadata(sequence_index=1),
        ]

        z = zip(sink.records, source.sent_metadata, strict=True)
        for (_, received), original in z:
            if change_sequence:
                assert received is not original
            else:
                assert received is original

        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


@pytest.mark.parametrize("mode", ["preserve", "update", "clear"])
def test_forwarding_preserves_ownership_and_keeps_branch_metadata_independent(
    mode: str,
) -> None:
    class Forwarder(StubWorker[bytearray, bytearray]):
        def _work(self) -> None:
            try:
                while True:
                    with self._receive(timeout=2.0) as incoming:
                        if mode == "preserve":
                            outgoing = incoming
                        elif mode == "update":
                            sequence_index = incoming.metadata.sequence_index
                            assert sequence_index is not None
                            outgoing = replace(
                                incoming,
                                metadata=replace(
                                    incoming.metadata,
                                    sequence_index=sequence_index + 100,
                                ),
                            )
                        else:
                            outgoing = replace(incoming, metadata=Metadata())

                        self._publish(outgoing, timeout=2.0)
            except EndOfStream:
                return

    source = MetadataSource()
    original = MetadataCollector(upstream=source)
    forwarder = Forwarder(upstream=source)
    modified = MetadataCollector(upstream=forwarder)
    pipeline = Pipeline(original, modified)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        assert original.records == [
            (b"one!", Metadata(sequence_index=0)),
            (b"two!", Metadata(sequence_index=1)),
        ]
        assert original.buffer_ids == modified.buffer_ids
        assert [value for value, _ in modified.records] == [b"one!", b"two!"]

        if mode == "preserve":
            assert modified.records == original.records
            assert modified.product_ids == original.product_ids
        elif mode == "update":
            assert [metadata for _, metadata in modified.records] == [
                Metadata(sequence_index=100),
                Metadata(sequence_index=101),
            ]
        else:
            assert [metadata for _, metadata in modified.records] == [
                Metadata(),
                Metadata(),
            ]

        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)


def test_new_payload_publication_does_not_implicitly_inherit_metadata() -> None:
    class Source(StubWorker[Never, bool]):
        def _work(self) -> None:
            self._publish(
                Product(False, metadata=Metadata(sequence_index=5)),
                timeout=2.0,
            )

    class Processor(StubWorker[bool, bool]):
        def _work(self) -> None:
            with self._receive(timeout=2.0) as incoming:
                assert incoming.metadata == Metadata(sequence_index=5)
                self._publish(not incoming.payload, timeout=2.0)

    class Sink(StubWorker[bool, Never]):
        def __init__(self, *, upstream: Worker[Any, bool]) -> None:
            super().__init__(upstream=upstream)
            self.received: tuple[bool, Metadata] | None = None

        def _work(self) -> None:
            with self._receive(timeout=2.0) as product:
                self.received = (product.payload, product.metadata)

    source = Source()
    processor = Processor(upstream=source)
    sink = Sink(upstream=processor)
    pipeline = Pipeline(sink)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)
        assert sink.received == (True, Metadata())
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)

