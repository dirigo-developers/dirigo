"""Test ownership transfer when pooled buffers are published and forwarded.

Successful publication invalidates the producer lease and gives each active
recipient a delivery claim. The buffer returns to its pool only after every
claim has ended.

Verify ownership across publication failures, retries, consumer exceptions,
closure, cancellation, and forwarding. Metadata behavior is tested separately.
"""

from threading import Condition, Event, Thread

import pytest

from dirigo.pipeline import EndOfStream, OperationCancelled, Product
from dirigo.pipeline.pool import Pool
from dirigo.pipeline.publisher import Publisher


def _pool() -> Pool[bytearray]:
    """Create a one-buffer pool so availability reveals outstanding ownership."""
    return Pool[bytearray](capacity=1, factory=lambda: bytearray(4))


def _assert_unavailable(pool: Pool[bytearray]) -> None:
    """Verify that the pool's only buffer cannot currently be acquired."""
    with pytest.raises(TimeoutError), pool.acquire(timeout=0.0):
        pass


def _assert_reusable(pool: Pool[bytearray], expected: bytearray) -> None:
    """Verify that the original buffer was returned exactly once.

    Reacquiring proves that it was returned. Exhausting the pool while holding
    it proves that no duplicate available entry was introduced.
    """
    with pool.acquire(timeout=0.0) as lease:
        assert lease.payload is expected
        _assert_unavailable(pool)


def _observe_wait(
    condition: Condition,
    monkeypatch: pytest.MonkeyPatch,
) -> Event:
    """Signal when an operation reaches the condition's wait method."""
    waiting = Event()
    original_wait = condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", observed_wait)
    return waiting


def test_transfer_keeps_buffer_until_final_delivery_exits() -> None:
    """Publication transfers ownership from the producer to both subscribers.

    The producer lease becomes invalid immediately. Leaving its acquisition
    context must not recycle the buffer while deliveries remain.

    Completing the first delivery still leaves the second subscriber's claim.
    Only exiting the final receive context may return the buffer to the pool.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    first = publisher.subscribe()
    second = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        original[:] = b"data"

        publisher.publish_lease(lease, timeout=0.0)

        with pytest.raises(RuntimeError, match="released or transferred"):
            _ = lease.payload

        _assert_unavailable(pool)

    # Producer context cleanup must not release the consumers' claims.
    _assert_unavailable(pool)

    with first.receive(timeout=0.0) as first_product:
        assert first_product.payload is original
        assert first_product.payload == bytearray(b"data")

    # The second subscriber still holds a queued delivery claim.
    _assert_unavailable(pool)

    with second.receive(timeout=0.0) as second_product:
        assert second_product is first_product
        assert second_product.payload is original
        _assert_unavailable(pool)

    _assert_reusable(pool, original)


def test_publication_timeout_preserves_producer_lease_and_is_atomic() -> None:
    """A timed-out publication transfers ownership to no subscriber.

    Leave one subscriber empty and another full. The attempted publication must
    reach neither, and the producer must retain its valid lease until release.
    The full subscriber must retain only its previously queued product.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    first = publisher.subscribe()
    second = publisher.subscribe()
    blocker = Product(bytearray(b"busy"))

    publisher.publish(blocker, timeout=0.0)

    with first.receive(timeout=0.0):
        pass

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        with pytest.raises(TimeoutError):
            publisher.publish_lease(lease, timeout=0.0)

        assert lease.payload is original
        _assert_unavailable(pool)

        with pytest.raises(TimeoutError), first.receive(timeout=0.0):
            pass

    _assert_reusable(pool, original)

    with second.receive(timeout=0.0) as received:
        assert received is blocker


def test_lease_can_be_retried_after_publication_timeout() -> None:
    """A failed publication leaves the lease available for a successful retry.

    Fill the subscriber queue, attempt publication, then consume the blocker.
    Retrying with the same lease must deliver the original buffer and recycle
    it after the consumer finishes.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()

    publisher.publish(Product(bytearray(4)), timeout=0.0)

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        with pytest.raises(TimeoutError):
            publisher.publish_lease(lease, timeout=0.0)

        with subscription.receive(timeout=0.0):
            pass

        publisher.publish_lease(lease, timeout=0.0)

    _assert_unavailable(pool)

    with subscription.receive(timeout=0.0) as product:
        assert product.payload is original

    _assert_reusable(pool, original)


@pytest.mark.parametrize("detached_subscribers", [0, 1, 3])
def test_transfer_without_active_subscribers_returns_immediately(
    detached_subscribers: int,
) -> None:
    """Publication without recipients immediately recycles the buffer.

    Both an empty subscriber list and a list containing only a detached
    subscriber leave no delivery claims. Successful publication must still
    invalidate the producer lease.

    Later acquisition-context cleanup must not return the buffer a second time.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()

    for _ in range(detached_subscribers):
        publisher.subscribe().close()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        publisher.publish_lease(lease, timeout=0.0)

        with pytest.raises(RuntimeError, match="released or transferred"):
            _ = lease.payload

        _assert_reusable(pool, original)

    _assert_reusable(pool, original)


def test_detached_subscriber_does_not_leave_an_extra_claim() -> None:
    """Only active subscribers receive ownership claims during publication.

    Publish with one detached subscriber and one active subscriber. Completing
    the active delivery must return the buffer; an erroneous extra claim for
    the detached subscriber would leave the pool exhausted.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    publisher.subscribe().close()
    active = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)

    with active.receive(timeout=0.0):
        pass

    _assert_reusable(pool, original)


def test_consumer_exception_releases_its_delivery() -> None:
    """Exceptional receive-context exit releases the consumer's ownership claim.

    The consumer's original exception must propagate. Its delivery claim must
    still be released, allowing the buffer to return to the pool.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()
    failure = RuntimeError("consumer failed")

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)

    with (
        pytest.raises(RuntimeError, match="consumer failed") as caught,
        subscription.receive(timeout=0.0),
    ):
        raise failure

    assert caught.value is failure
    _assert_reusable(pool, original)


@pytest.mark.parametrize("operation", ["close", "abort", "publisher_abort"])
def test_discarding_pending_delivery_returns_buffer(operation: str) -> None:
    """Discarding the only queued delivery releases its claim and recycles storage.

    Subscription closure, subscription cancellation, and publisher cancellation
    must all release discarded delivery claims. Subsequent lifecycle calls must
    not release those claims again or create duplicate pool entries.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)

    if operation == "close":
        subscription.close()
    elif operation == "abort":
        subscription.abort()
    else:
        publisher.abort()

    _assert_reusable(pool, original)

    subscription.close()
    subscription.abort()
    publisher.abort()

    _assert_reusable(pool, original)


def test_publisher_close_preserves_queued_ownership_until_drain() -> None:
    """Normal publisher closure preserves queued claims until consumers finish.

    Closing the publisher must not return a buffer that is still queued.
    Receiving it must also preserve ownership throughout the receive context.
    The buffer becomes reusable only after that context exits.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)

    publisher.close()

    _assert_unavailable(pool)

    with subscription.receive(timeout=0.0) as product:
        assert product.payload is original
        _assert_unavailable(pool)

    _assert_reusable(pool, original)

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass


def test_abort_discards_queue_but_preserves_a_borrowed_delivery() -> None:
    """Cancellation releases queued claims without releasing a borrowed claim.

    Publish one buffer to two subscribers, then borrow one delivery and abort.
    The queued delivery must be cancelled, but the borrowed delivery must keep
    the buffer unavailable until its receive context exits.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    borrowed_subscription = publisher.subscribe()
    queued_subscription = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)

    with borrowed_subscription.receive(timeout=0.0) as product:
        publisher.abort()

        assert product.payload is original
        _assert_unavailable(pool)

        with (
            pytest.raises(OperationCancelled),
            queued_subscription.receive(timeout=0.0),
        ):
            pass

    _assert_reusable(pool, original)


@pytest.mark.parametrize("operation", ["close", "abort"])
def test_rejected_publication_keeps_producer_ownership(operation: str) -> None:
    """Publisher closure or cancellation rejects transfer without consuming the lease.

    The producer must retain access to its buffer after rejection. The buffer
    must remain unavailable to other acquisitions until the producer's context
    releases its claim.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    expected_error = RuntimeError if operation == "close" else OperationCancelled

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        if operation == "close":
            publisher.close()
        else:
            publisher.abort()

        with pytest.raises(expected_error):
            publisher.publish_lease(lease, timeout=0.0)

        assert lease.payload is original
        _assert_unavailable(pool)

    _assert_reusable(pool, original)


@pytest.mark.parametrize("transferred", [False, True])
def test_expired_producer_lease_cannot_be_published(transferred: bool) -> None:
    """A released or previously transferred lease cannot transfer ownership again.

    Publication without subscribers immediately returns the buffer, just as
    explicit release does. Attempting to publish the expired handle must raise
    and leave exactly one available copy of the buffer.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        if transferred:
            publisher.publish_lease(lease, timeout=0.0)
        else:
            lease.release()

        with pytest.raises(RuntimeError, match="released or transferred"):
            publisher.publish_lease(lease, timeout=0.0)

    _assert_reusable(pool, original)


def test_forwarding_pooled_product_retains_the_original_buffer() -> None:
    """Forwarding adds a downstream claim without copying the pooled payload.

    The middle consumer forwards its received Product through another publisher.
    Leaving the middle receive context must not recycle the buffer while the
    downstream delivery remains outstanding.
    """
    pool = _pool()
    upstream = Publisher[bytearray]()
    middle = upstream.subscribe()
    downstream = Publisher[bytearray]()
    sink = downstream.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        upstream.publish_lease(lease, timeout=0.0)

    with middle.receive(timeout=0.0) as product:
        downstream.publish(product, timeout=0.0)

    _assert_unavailable(pool)

    with sink.receive(timeout=0.0) as received:
        assert received is product
        assert received.payload is original

    _assert_reusable(pool, original)


def test_expired_pooled_product_cannot_be_forwarded() -> None:
    """A Product whose final claim has ended cannot establish a new delivery.

    Keeping a Python reference to the Product does not extend its ownership.
    After its receive context releases the final claim, forwarding must raise
    rather than revive the exhausted ownership record.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    subscription = publisher.subscribe()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)

    with subscription.receive(timeout=0.0) as expired_product:
        pass

    with pytest.raises(RuntimeError, match="ownership has been released"):
        publisher.publish(expired_product, timeout=0.0)

    _assert_reusable(pool, original)


@pytest.mark.parametrize("abort_publisher", [False, True])
def test_waiting_publication_handles_detachment_or_abort(
    monkeypatch: pytest.MonkeyPatch,
    abort_publisher: bool,
) -> None:
    """A waiting lease transfer handles detachment and cancellation correctly.

    Fill the slow subscriber's queue while leaving the active subscriber empty.
    The producer must then wait for capacity before it can transfer its lease.

    Detaching the slow subscriber must allow transfer to the active subscriber
    and invalidate the producer lease. Aborting the publisher must instead
    reject transfer and preserve the producer's claim.

    The publishing thread must finish before its publication timeout expires.
    """
    pool = _pool()
    publisher = Publisher[bytearray]()
    slow = publisher.subscribe()
    active = publisher.subscribe()

    publisher.publish(Product(bytearray(4)), timeout=0.0)

    with active.receive(timeout=0.0):
        pass

    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        def publish() -> None:
            try:
                publisher.publish_lease(lease, timeout=5.0)
            except BaseException as exc:
                errors.append(exc)

        thread = Thread(target=publish)
        thread.start()

        try:
            assert waiting.wait(timeout=2.0)

            if abort_publisher:
                publisher.abort()
            else:
                slow.close()

            thread.join(timeout=2.0)

            assert not thread.is_alive()

            if abort_publisher:
                assert len(errors) == 1
                assert isinstance(errors[0], OperationCancelled)
                assert lease.payload is original
            else:
                assert errors == []

                with pytest.raises(RuntimeError, match="released or transferred"):
                    _ = lease.payload

                with active.receive(timeout=0.0) as product:
                    assert product.payload is original
        finally:
            publisher.abort()
            thread.join(timeout=6.0)

    _assert_reusable(pool, original)


def test_unpooled_delivery_remains_unchanged() -> None:
    """Ordinary products still support fanout and draining after publisher closure.

    This regression check verifies that the ownership-transfer machinery also
    permits delivery of products that have no pooled ownership record.
    """
    publisher = Publisher[bool]()
    subscriptions = [
        publisher.subscribe(),
        publisher.subscribe(),
    ]
    product = Product(False)

    publisher.publish(product, timeout=0.0)
    publisher.close()

    for subscription in subscriptions:
        with subscription.receive(timeout=0.0) as received:
            assert received is product
            assert received.payload is False

        with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
            pass


def test_publish_lease_rejects_nonlease() -> None:
    """Lease publication rejects objects that cannot supply a producer claim."""
    publisher = Publisher[bytearray]()

    with pytest.raises(TypeError, match="lease must be a Lease"):
        publisher.publish_lease(
            object(),  # ty: ignore[invalid-argument-type]
            timeout=0.0,
        )
