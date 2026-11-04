"""Test cancellation of publishers and subscriptions.

Aborting discards queued deliveries and wakes blocked operations. A subscriber
abort detaches only that consumer; a publisher abort cancels every subscriber.

Borrowed products remain usable until their receive contexts exit. Cancellation
takes precedence over normal closure and remains effective after repeated calls.
Worker cancellation and pooled-buffer recycling are tested separately.
"""

import gc
import weakref
from threading import Condition, Event, Thread

import pytest

from dirigo.pipeline import OperationCancelled, Product
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


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_discards_queued_products(scope: str) -> None:
    """Aborting either end releases payloads retained by queued deliveries.

    The weak reference observes the payload without keeping it alive. After
    deleting the local strong reference, the queued product keeps it alive.

    Aborting must discard that product. Explicit garbage collection avoids
    relying on immediate destruction; afterward, the weak reference must return
    None. Further reception must report cancellation.
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

    target = subscription if scope == "subscription" else publisher
    target.abort()
    gc.collect()

    assert reference() is None

    with pytest.raises(OperationCancelled), subscription.receive(timeout=0.0):
        pass


def test_subscription_abort_preserves_other_deliveries() -> None:
    """Aborting one subscriber preserves another's queued and future deliveries.

    Both subscribers initially have a queued product. Aborting one must leave
    the other subscriber's copy available and allow subsequent publication to
    continue. The aborted subscriber must report cancellation.
    """
    publisher = Publisher[bool]()
    aborted = publisher.subscribe()
    active = publisher.subscribe()
    original = Product(False)

    publisher.publish(original, timeout=0.0)
    aborted.abort()

    with active.receive(timeout=0.0) as received:
        assert received is original

    following = Product(True)
    publisher.publish(following, timeout=0.0)

    with active.receive(timeout=0.0) as received:
        assert received is following

    with pytest.raises(OperationCancelled), aborted.receive(timeout=0.0):
        pass


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_wakes_blocked_receiver(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    """Aborting either end wakes an empty subscription's waiting receiver.

    Observe the receiver entering its wait before issuing the abort. It must
    finish with OperationCancelled before its receive timeout expires, proving
    that cancellation actively wakes the blocked operation.
    """
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
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

        target = subscription if scope == "subscription" else publisher
        target.abort()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], OperationCancelled)
    finally:
        subscription.close()
        thread.join(timeout=6.0)


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_wakes_blocked_publisher(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    """Aborting a subscriber or publisher wakes publication waiting for capacity.

    The slow subscriber's queue is full while the active subscriber has space.
    This prevents the pending publication from proceeding.

    Aborting only the slow subscriber removes the obstacle: publication must
    succeed and deliver the pending product to the active subscriber.

    Aborting the publisher instead must reject the pending publication with
    OperationCancelled and cancel both subscriptions. In either case, the
    publishing thread must finish before its publication timeout expires.
    """
    publisher = Publisher[bool]()
    slow = publisher.subscribe(capacity=1)
    active = publisher.subscribe(capacity=1)
    following = Product(True)

    publisher.publish(Product(False), timeout=0.0)

    # Leave only the slow subscriber's queue full.
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

        target = slow if scope == "subscription" else publisher
        target.abort()
        thread.join(timeout=2.0)

        assert not thread.is_alive()

        with pytest.raises(OperationCancelled), slow.receive(timeout=0.0):
            pass

        if scope == "subscription":
            assert errors == []

            with active.receive(timeout=0.0) as received:
                assert received is following

            with pytest.raises(TimeoutError), active.receive(timeout=0.0):
                pass
        else:
            assert len(errors) == 1
            assert isinstance(errors[0], OperationCancelled)

            with pytest.raises(OperationCancelled), active.receive(timeout=0.0):
                pass
    finally:
        publisher.close()
        slow.close()
        active.close()
        thread.join(timeout=6.0)


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_preserves_borrowed_product(scope: str) -> None:
    """Cancellation preserves a borrowed product while discarding queued ones.

    Receive the first of two products, then abort while its context is active.
    The borrowed product must remain readable until context exit. A subsequent
    receive must report cancellation rather than deliver the second product.
    """
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=2)
    first = Product(False)

    publisher.publish(first, timeout=0.0)
    publisher.publish(Product(True), timeout=0.0)

    with subscription.receive(timeout=0.0) as borrowed:
        target = subscription if scope == "subscription" else publisher
        target.abort()

        assert borrowed is first
        assert borrowed.payload is False

    with pytest.raises(OperationCancelled), subscription.receive(timeout=0.0):
        pass


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_overrides_normal_close_and_is_idempotent(scope: str) -> None:
    """Cancellation takes precedence over closure and survives repeated calls.

    Normal publisher closure would allow the queued product to drain. Aborting
    afterward must instead discard it and make reception report cancellation.
    Repeating abort, or calling close afterward, must preserve that outcome.
    """
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    publisher.publish(Product(False), timeout=0.0)
    publisher.close()

    target = subscription if scope == "subscription" else publisher
    target.abort()
    target.abort()

    subscription.close()
    publisher.close()

    with pytest.raises(OperationCancelled), subscription.receive(timeout=0.0):
        pass


def test_aborted_publisher_rejects_publish_and_subscribe() -> None:
    """An aborted publisher rejects new operations with OperationCancelled."""
    publisher = Publisher[bool]()
    publisher.abort()

    with pytest.raises(OperationCancelled):
        publisher.publish(Product(False), timeout=0.0)

    with pytest.raises(OperationCancelled):
        publisher.subscribe()
