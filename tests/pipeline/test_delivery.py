"""Test ordinary publication and reception through bounded subscriptions.

Verify delivery identity, ordering, input validation, receive-context behavior,
and atomic publication when subscriber capacity is exhausted.

Closure, cancellation, pooled ownership, and worker execution are tested in
separate modules.
"""

import pytest

from dirigo.pipeline import Product
from dirigo.pipeline.publisher import Publisher


@pytest.mark.parametrize("payload", [False, True])
def test_publish_delivers_once_to_each_subscriber(payload: bool) -> None:
    """Each subscriber receives the original product exactly once."""
    publisher = Publisher[bool]()
    subscriptions = [
        publisher.subscribe(capacity=1),
        publisher.subscribe(capacity=1),
    ]
    product = Product(payload)

    publisher.publish(product, timeout=0.0)

    for subscription in subscriptions:
        with subscription.receive(timeout=0.0) as received:
            assert received is product
            assert received.payload is payload

        with pytest.raises(TimeoutError), subscription.receive(timeout=0.0):
            pass


def test_subscribers_receive_products_in_publication_order() -> None:
    """Every subscriber receives products in the order they were published."""
    publisher = Publisher[bool]()
    subscriptions = [
        publisher.subscribe(capacity=3),
        publisher.subscribe(capacity=3),
    ]
    products = [
        Product(True),
        Product(False),
        Product(True),
    ]

    for product in products:
        publisher.publish(product, timeout=0.0)

    for subscription in subscriptions:
        for expected in products:
            with subscription.receive(timeout=0.0) as received:
                assert received is expected


def test_none_payload_is_delivered_as_data() -> None:
    """A None payload is ordinary data and does not signal end of stream."""
    publisher = Publisher[None]()
    subscription = publisher.subscribe()
    product = Product(None)

    publisher.publish(product, timeout=0.0)

    with subscription.receive(timeout=0.0) as received:
        assert received is product
        assert received.payload is None


@pytest.mark.parametrize("capacity", [0, -1])
def test_subscribe_rejects_nonpositive_capacity(capacity: int) -> None:
    """Subscription queues must have capacity for at least one product."""
    publisher = Publisher[bool]()

    with pytest.raises(ValueError, match="must be positive"):
        publisher.subscribe(capacity=capacity)


def test_publish_rejects_invalid_product() -> None:
    """The publisher requires a Product wrapper rather than a raw payload."""
    publisher = Publisher[int]()

    with pytest.raises(TypeError, match="must be a Product"):
        publisher.publish(42, timeout=0.0)  # ty: ignore[invalid-argument-type]


def test_publish_rejects_negative_timeout() -> None:
    """A negative publication timeout is invalid even without subscribers."""
    publisher = Publisher[None]()
    product = Product(None)

    with pytest.raises(ValueError, match="must be nonnegative"):
        publisher.publish(product, timeout=-1.0)


def test_receive_rejects_negative_timeout() -> None:
    """Entering a receive context rejects a negative timeout."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    with (
        pytest.raises(ValueError, match="must be nonnegative"),
        subscription.receive(timeout=-1.0),
    ):
        pass


def test_empty_subscription_times_out() -> None:
    """A zero-timeout receive fails immediately when no product is queued."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    with pytest.raises(TimeoutError), subscription.receive(timeout=0.0):
        pass


def test_receive_begins_on_context_entry() -> None:
    """Creating a receive context does not try to receive a product."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    # The queue is empty here; an eager receive would raise TimeoutError.
    delivery = subscription.receive(timeout=0.0)

    product = Product(False)
    publisher.publish(product, timeout=0.0)

    with delivery as received:
        assert received is product


def test_receiving_frees_capacity_before_context_exit() -> None:
    """A borrowed product no longer occupies a slot in the subscriber queue."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=1)
    first = Product(False)
    second = Product(True)

    publisher.publish(first, timeout=0.0)

    with subscription.receive(timeout=0.0) as received:
        # Publication succeeds while the consumer still holds the first product.
        publisher.publish(second, timeout=0.0)

        assert received is first
        assert received.payload is False

    with subscription.receive(timeout=0.0) as received:
        assert received is second


def test_receive_context_does_not_suppress_consumer_exception() -> None:
    """Consumer exceptions propagate, and subsequent delivery still works."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=1)
    failure = RuntimeError("consumer failed")
    first = Product(False)

    publisher.publish(first, timeout=0.0)

    with (
        pytest.raises(RuntimeError, match="consumer failed") as caught,
        subscription.receive(timeout=0.0) as received,
    ):
        raise failure

    assert received is first
    assert caught.value is failure

    second = Product(True)
    publisher.publish(second, timeout=0.0)

    with subscription.receive(timeout=0.0) as received:
        assert received is second


def test_publish_timeout_does_not_partially_deliver() -> None:
    """If any recipient queue is full, a timed-out publication reaches none."""
    publisher = Publisher[bool]()
    first = publisher.subscribe(capacity=1)
    second = publisher.subscribe(capacity=1)
    original = Product(False)
    following = Product(True)

    publisher.publish(original, timeout=0.0)

    # Leave the first subscriber empty and the second subscriber full.
    with first.receive(timeout=0.0) as received:
        assert received is original

    with pytest.raises(TimeoutError, match="Publication exceeded the timeout"):
        publisher.publish(following, timeout=0.0)

    # The subscriber with available capacity must not receive a partial delivery.
    with pytest.raises(TimeoutError), first.receive(timeout=0.0):
        pass

    # The full subscriber retains only its original product.
    with second.receive(timeout=0.0) as received:
        assert received is original

    with pytest.raises(TimeoutError), second.receive(timeout=0.0):
        pass
