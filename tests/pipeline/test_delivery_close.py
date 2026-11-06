"""Test normal closure of publishers and subscriptions.

Publisher closure preserves queued deliveries until consumers drain them.
Subscription closure detaches a consumer and discards its queued deliveries
while preserving any product already borrowed through a receive context.

Verify that closure wakes blocked operations and repeated closure is harmless.
Cancellation and pooled-buffer recycling are tested in separate modules.
"""

import gc
import weakref
from threading import Condition, Event, Thread

import pytest

from dirigo.pipeline import EndOfStream, Product
from dirigo.pipeline.publisher import Publisher


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


def test_closing_empty_publisher_ends_stream() -> None:
    """Closing a publisher with no queued products ends reception immediately."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    publisher.close()

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass


def test_publisher_close_allows_queued_products_to_drain() -> None:
    """Each subscriber receives its queued products before reaching end of stream."""
    publisher = Publisher[bool]()
    subscriptions = [
        publisher.subscribe(capacity=2),
        publisher.subscribe(capacity=2),
    ]
    products = [Product(False), Product(True)]

    for product in products:
        publisher.publish(product, timeout=0.0)

    publisher.close()

    for subscription in subscriptions:
        for expected in products:
            with subscription.receive(timeout=0.0) as received:
                assert received is expected

        with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
            pass


@pytest.mark.parametrize("scope", ["publisher", "subscription"])
def test_close_wakes_blocked_receiver(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    """Closing either end wakes a waiting receiver with EndOfStream."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
    close = publisher.close if scope == "publisher" else subscription.close
    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def receive() -> None:
        try:
            with subscription.receive(timeout=5.0):
                pass
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=receive)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        close()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], EndOfStream)
    finally:
        close()
        thread.join(timeout=6.0)


def test_closed_publisher_rejects_publish_and_subscribe() -> None:
    """A closed publisher accepts neither new products nor new subscriptions."""
    publisher = Publisher[bool]()
    publisher.close()

    with pytest.raises(RuntimeError, match="closed"):
        publisher.publish(Product(False), timeout=0.0)

    with pytest.raises(RuntimeError, match="closed"):
        publisher.subscribe()


def test_repeated_publisher_close_is_harmless() -> None:
    """Repeated closure preserves queued products and the eventual end of stream."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
    product = Product(False)
    publisher.publish(product, timeout=0.0)

    publisher.close()
    publisher.close()

    with subscription.receive(timeout=0.0) as received:
        assert received is product

    publisher.close()

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass


def test_publisher_close_wakes_blocked_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the publisher wakes and rejects a publication waiting for capacity.

    One subscriber has space while another has a full queue, preventing the
    pending publication from proceeding. Closing the publisher must wake the
    publishing thread with RuntimeError before its publication timeout expires.

    The pending product must reach neither subscriber. The product queued
    before closure must remain available to its original recipient.
    """
    publisher = Publisher[bool]()
    first = publisher.subscribe(capacity=1)
    second = publisher.subscribe(capacity=1)
    original = Product(False)
    following = Product(True)

    publisher.publish(original, timeout=0.0)

    # Leave the first subscriber empty and the second subscriber full.
    with first.receive(timeout=0.0):
        pass

    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish(following, timeout=5.0)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=publish)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        publisher.close()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "closed" in str(errors[0])

        # Available capacity must not permit a partial delivery.
        with pytest.raises(EndOfStream), first.receive(timeout=0.0):
            pass

        # The original queued product remains available.
        with second.receive(timeout=0.0) as received:
            assert received is original

        with pytest.raises(EndOfStream), second.receive(timeout=0.0):
            pass
    finally:
        publisher.close()
        thread.join(timeout=6.0)


def test_subscription_close_discards_queued_products() -> None:
    """Closing a subscription releases the payload held by its queued product.

    A weak reference lets us check whether the payload still exists without
    keeping it alive. After deleting our local strong reference, the queued
    product keeps the payload alive until the subscription is closed.

    Explicit garbage collection avoids relying on immediate object destruction.
    Once the queue is discarded and collection runs, the weak reference must
    return None, proving that the delivery channel no longer retains the payload.
    """

    class Payload:
        pass

    publisher = Publisher[Payload]()
    subscription = publisher.subscribe()
    payload = Payload()
    reference = weakref.ref(payload)

    publisher.publish(Product(payload), timeout=0.0)
    del payload

    assert reference() is not None

    subscription.close()
    gc.collect()

    assert reference() is None

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass


def test_closing_one_subscription_preserves_other_deliveries() -> None:
    """Detaching one consumer preserves another's queued and future deliveries."""
    publisher = Publisher[bool]()
    first = publisher.subscribe()
    second = publisher.subscribe()
    original = Product(False)

    publisher.publish(original, timeout=0.0)
    first.close()

    with second.receive(timeout=0.0) as received:
        assert received is original

    following = Product(True)
    publisher.publish(following, timeout=0.0)

    with pytest.raises(EndOfStream), first.receive(timeout=0.0):
        pass

    with second.receive(timeout=0.0) as received:
        assert received is following


def test_subscription_close_unblocks_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Detaching a full subscriber lets a waiting publication proceed.

    Both subscribers initially receive a product, but only the active subscriber
    consumes it. The slow subscriber's full queue then blocks the next
    publication, even though the active subscriber has available capacity.

    Closing the slow subscription removes that obstacle. The publishing thread
    must finish successfully before its publication timeout expires, delivering
    the pending product exactly once to the active subscriber. The detached
    subscriber must report EndOfStream.
    """
    publisher = Publisher[bool]()
    slow = publisher.subscribe(capacity=1)
    active = publisher.subscribe(capacity=1)
    original = Product(False)
    following = Product(True)

    publisher.publish(original, timeout=0.0)

    # Only the slow subscriber's queue remains full.
    with active.receive(timeout=0.0):
        pass

    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish(following, timeout=5.0)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=publish)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        slow.close()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert errors == []

        with pytest.raises(EndOfStream), slow.receive(timeout=0.0):
            pass

        with active.receive(timeout=0.0) as received:
            assert received is following

        with pytest.raises(TimeoutError), active.receive(timeout=0.0):
            pass
    finally:
        slow.close()
        active.close()
        thread.join(timeout=6.0)


def test_subscription_close_preserves_borrowed_product() -> None:
    """Detachment discards queued products while leaving a borrowed one usable."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=2)
    first = Product(False)

    publisher.publish(first, timeout=0.0)
    publisher.publish(Product(True), timeout=0.0)

    with subscription.receive(timeout=0.0) as borrowed:
        subscription.close()

        assert borrowed is first
        assert borrowed.payload is False

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass


def test_repeated_subscription_close_is_harmless() -> None:
    """Repeated detachment remains harmless and does not obstruct publication."""
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    subscription.close()
    subscription.close()

    publisher.publish(Product(False), timeout=0.0)

    subscription.close()

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass

    publisher.close()
    subscription.close()
