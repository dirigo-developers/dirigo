from threading import Condition, Event, Thread

import pytest

from dirigo.pipeline import EndOfStream, OperationCancelled, Product
from dirigo.pipeline.pool import Pool
from dirigo.pipeline.publisher import Publisher


def _pool() -> Pool[bytearray]:
    return Pool[bytearray](capacity=1, factory=lambda: bytearray(4))


def _assert_unavailable(pool: Pool[bytearray]) -> None:
    with pytest.raises(TimeoutError): # ruff: ignore[multiple-with-statements]
        with pool.acquire(timeout=0.0):
            pass


def _assert_reusable(pool: Pool[bytearray], expected: bytearray) -> None:
    with pool.acquire(timeout=0.0) as lease:
        assert lease.payload is expected
        # The buffer must have been returned once, rather than twice.
        _assert_unavailable(pool)


def _observe_wait(
    condition: Condition, monkeypatch: pytest.MonkeyPatch
) -> Event:
    waiting = Event()
    original_wait = condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", observed_wait)
    return waiting


def test_transfer_keeps_buffer_until_final_delivery_exits() -> None:
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

    # Exiting the producer context must not return a transferred buffer.
    _assert_unavailable(pool)

    with first.receive(timeout=0.0) as first_product:
        assert first_product.payload is original
        assert first_product.payload == bytearray(b"data")

    # The second subscriber still owns a queued delivery.
    _assert_unavailable(pool)

    with second.receive(timeout=0.0) as second_product:
        assert second_product is first_product
        assert second_product.payload is original
        _assert_unavailable(pool)

    _assert_reusable(pool, original)


def test_publication_timeout_preserves_producer_lease_and_is_atomic() -> None:
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


@pytest.mark.parametrize("detached_subscribers", [0, 1])
def test_transfer_without_active_subscribers_returns_immediately(
    detached_subscribers: int,
) -> None:
    pool = _pool()
    publisher = Publisher[bytearray]()
    for _ in range(detached_subscribers):
        publisher.subscribe().close()

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload
        publisher.publish_lease(lease, timeout=0.0)
        with pytest.raises(RuntimeError):
            _ = lease.payload
        _assert_reusable(pool, original)

    _assert_reusable(pool, original)


def test_detached_subscriber_does_not_leave_an_extra_claim() -> None:
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
    # Later lifecycle calls must not release the delivery a second time.
    subscription.close()
    subscription.abort()
    publisher.abort()
    _assert_reusable(pool, original)


def test_publisher_close_preserves_queued_ownership_until_drain() -> None:
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
        with pytest.raises(OperationCancelled), queued_subscription.receive(0.0):
            pass

    _assert_reusable(pool, original)


@pytest.mark.parametrize("operation", ["close", "abort"])
def test_rejected_publication_keeps_producer_ownership(operation: str) -> None:
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
    monkeypatch: pytest.MonkeyPatch, abort_publisher: bool
) -> None:
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
                publisher.publish_lease(lease, timeout=2.0)
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
            thread.join(timeout=3.0)
            assert not thread.is_alive()

            if abort_publisher:
                assert len(errors) == 1
                assert isinstance(errors[0], OperationCancelled)
                assert lease.payload is original
            else:
                assert errors == []
                with pytest.raises(RuntimeError):
                    _ = lease.payload
                with active.receive(timeout=0.0) as product:
                    assert product.payload is original
        finally:
            publisher.abort()
            thread.join(timeout=3.0)

    _assert_reusable(pool, original)


def test_unpooled_delivery_remains_unchanged() -> None:
    publisher = Publisher[bool]()
    first = publisher.subscribe()
    second = publisher.subscribe()
    product = Product(False)
    publisher.publish(product, timeout=0.0)
    publisher.close()
    for subscription in (first, second):
        with subscription.receive(timeout=0.0) as received:
            assert received is product
            assert received.payload is False
        with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
            pass


def test_publish_lease_rejects_nonlease() -> None:
    publisher = Publisher[bytearray]()

    with pytest.raises(TypeError, match="lease must be a Lease"):
        publisher.publish_lease(
            object(),  # ty: ignore[invalid-argument-type]
            timeout=0.0,
        )
