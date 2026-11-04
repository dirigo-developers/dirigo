"""Test metadata handling by processors and forwarding workers.

Processors producing new payloads must explicitly inherit or replace metadata.
Forwarding workers may preserve, update, or clear metadata while sharing the
original pooled storage without changing another branch's metadata.

Publishing a plain payload starts with empty metadata rather than implicitly
inheriting the most recently received record.
"""

from dataclasses import replace
from typing import Any, Never

import pytest

from dirigo.pipeline import EndOfStream, Metadata, Pipeline, Product, Worker

from .stubs import StubWorker


class MetadataSource(StubWorker[Never, bytearray]):
    """Publish two indexed records through a single reusable buffer."""

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
    """Record payload snapshots, metadata, and identities during valid deliveries."""

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
    """Cancel and join launched workers without re-raising recorded failures."""
    pipeline.abort()

    for worker in pipeline._launched:
        worker._wait(timeout=2.0)


@pytest.mark.parametrize("change_sequence", [False, True])
def test_processor_explicitly_inherits_or_updates_metadata(
    change_sequence: bool,
) -> None:
    """A processor explicitly chooses metadata for its newly produced payload.

    Transform each input into uppercase using a separate output pool. Assign
    either the incoming metadata record or a replacement with an adjusted index
    to the outgoing lease.

    Inheritance must preserve metadata identity. Updating must create a distinct
    record while leaving the source's original metadata unchanged.
    """
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

                        if change_sequence:
                            outgoing.metadata = replace(
                                incoming.metadata,
                                sequence_index=sequence_index + 100,
                            )
                        else:
                            outgoing.metadata = incoming.metadata

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

        records = zip(sink.records, source.sent_metadata, strict=True)

        for (_, received), original in records:
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
    """Forwarding can change metadata while retaining the original pooled buffer.

    Split the source into a direct collector and a forwarding branch. Forward
    either the original Product or a replacement with updated or empty metadata.

    Both branches must receive the same payload storage and original contents.
    Changes on the forwarding branch must not alter the direct branch's records.
    Preserving the Product must also preserve its identity across branches.

    Forwarding occurs inside the receive context so downstream claims are
    established before the forwarder's incoming claim is released.
    """
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

                            # Replace the envelope while sharing its payload
                            # and ownership record.
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
    """Publishing a plain payload creates a Product with empty metadata.

    Receive an indexed Product, then publish a transformed boolean directly.
    The outgoing payload must have its new value and empty metadata, proving
    that receiving a record does not implicitly establish publication metadata.
    """
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
                self._publish(not incoming.payload, timeout=2.0) # publishes w/out md

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
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        _abort_and_join(pipeline)
