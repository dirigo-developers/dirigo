import gc
import weakref
from threading import Condition, Event, Thread
from typing import Never

import pytest

from dirigo.exceptions import EndOfStream, OperationCancelled
from dirigo.pipeline import Pipeline
from dirigo.product import Product
from dirigo.publisher import Publisher
from dirigo.subscription import Subscription

from .stubs import BooleanSource, CollectingWorker, StubWorker


def _observe_wait(
    condition: Condition,
    monkeypatch: pytest.MonkeyPatch,
) -> Event:
    waiting = Event()
    original_wait = condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(target=condition, name="wait", value=observed_wait)
    return waiting


@pytest.mark.parametrize(argnames="payload", argvalues=[False, True])
def test_publish_delivers_once_to_each_subscriber(payload: bool) -> None:
    publisher: Publisher[bool] = Publisher[bool]()
    first: Subscription[bool] = publisher.subscribe(capacity=1)
    second: Subscription[bool] = publisher.subscribe(capacity=1)
    product: Product[bool] = Product(payload)

    publisher.publish(product, timeout=0.0)

    with first.receive(timeout=0.0) as received:
        assert received is product
        assert received.payload is payload

    with second.receive(timeout=0.0) as received:
        assert received is product
        assert received.payload is payload

    for subscription in (first, second):
        with pytest.raises(expected_exception=TimeoutError):  # ruff: ignore[multiple-with-statements]
            with subscription.receive(timeout=0.0):
                pass


def test_publisher_subcribe_negative_capacity_fails() -> None:
    publisher: Publisher[bool] = Publisher[bool]()

    with pytest.raises(expected_exception=ValueError, match="must be positive"):
        publisher.subscribe(capacity=0)

    with pytest.raises(expected_exception=ValueError, match="must be positive"):
        publisher.subscribe(capacity=-1)


def test_empty_subscription_times_out() -> None:
    publisher: Publisher[bool] = Publisher[bool]()
    subscription: Subscription[bool] = publisher.subscribe(capacity=1)

    with pytest.raises(expected_exception=TimeoutError):  # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


def test_subscribe_receive_negative_timeout_fails() -> None:
    publisher: Publisher[bool] = Publisher[bool]()
    subscription: Subscription[bool] = publisher.subscribe(capacity=1)

    with pytest.raises(expected_exception=ValueError, match="must be nonnegative"):  # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=-1.0):
            pass


def test_subscribers_receive_products_in_publication_order() -> None:
    publisher: Publisher[bool] = Publisher[bool]()
    subscriptions: list[Subscription[bool]] = [
        publisher.subscribe(capacity=3),
        publisher.subscribe(capacity=3),
    ]
    products: list[Product[bool]] = [
        Product(payload=True),
        Product(payload=False),
        Product(payload=True),
    ]

    for product in products:
        publisher.publish(product, timeout=0.0)

    for subscription in subscriptions:
        for expected in products:
            with subscription.receive(timeout=0.0) as received:
                assert received is expected


def test_none_payload_is_delivered_as_data() -> None:
    publisher: Publisher[None] = Publisher[None]()
    subscription: Subscription[None] = publisher.subscribe(capacity=1)
    product: Product[None] = Product(payload=None)

    publisher.publish(product, timeout=0.0)

    with subscription.receive(timeout=0.0) as received:
        assert received is product
        assert received.payload is None


def test_receive_context_does_not_suppress_consumer_exception() -> None:
    publisher: Publisher[bool] = Publisher[bool]()
    subscription: Subscription[bool] = publisher.subscribe(capacity=1)
    failure = RuntimeError("consumer failed")
    first: Product[bool] = Product(payload=False)

    publisher.publish(product=first, timeout=0.0)

    with pytest.raises(RuntimeError, match="consumer failed") as caught:  # ruff: ignore[multiple-with-statements, pytest-raises-with-multiple-statements]
        with subscription.receive(timeout=0.0) as received:
            assert received is first
            raise failure

    assert caught.value is failure

    second = Product(True)
    publisher.publish(second, timeout=0.0)

    with subscription.receive(timeout=0.0) as received:
        assert received is second


def test_publish_rejects_invalid_product() -> None:
    publisher: Publisher[None] = Publisher[None]()
    product = 42

    with pytest.raises(expected_exception=TypeError, match="must be a Product"):
        publisher.publish(product, timeout=0.0) # ty: ignore[invalid-argument-type]


def test_publish_rejects_negative_timeout() -> None:
    publisher: Publisher[None] = Publisher[None]()
    product: Product[None] = Product(payload=None)

    with pytest.raises(expected_exception=ValueError, match="must be nonnegative"):
        publisher.publish(product, timeout=-1)


def test_closing_empty_publisher_ends_stream() -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    publisher.close()

    with pytest.raises(EndOfStream), subscription.receive(timeout=0.0):
        pass


def test_publisher_close_allows_queued_products_to_drain() -> None:
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


def test_publisher_close_wakes_blocked_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def receive() -> None:
        try:
            with subscription.receive(timeout=2.0):
                pass
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=receive)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        publisher.close()
        thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], EndOfStream)
    finally:
        publisher.close()
        thread.join(timeout=3.0)


def test_closed_publisher_rejects_publish_and_subscribe() -> None:
    publisher = Publisher[bool]()
    publisher.close()

    with pytest.raises(RuntimeError, match="closed"):
        publisher.publish(Product(False), timeout=0.0)

    with pytest.raises(RuntimeError, match="closed"):
        publisher.subscribe()


def test_repeated_publisher_close_is_harmless() -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
    product = Product(False)
    publisher.publish(product, timeout=0.0)

    publisher.close()
    publisher.close()

    with subscription.receive(timeout=0.0) as received:
        assert received is product

    publisher.close()

    with pytest.raises(EndOfStream): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


def test_publisher_close_wakes_blocked_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = Publisher[bool]()
    first = publisher.subscribe(capacity=1)
    second = publisher.subscribe(capacity=1)
    original = Product(False)

    publisher.publish(original, timeout=0.0)

    # Leave first empty and second full.
    with first.receive(timeout=0.0):
        pass

    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish(Product(True), timeout=2.0)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=publish)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        publisher.close()
        thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "closed" in str(errors[0])

        # The pending publication wasn't delivered to first.
        with pytest.raises(EndOfStream), first.receive(timeout=0.0):
            pass

        # Second retains only its original queued product.
        with second.receive(timeout=0.0) as received:
            assert received is original

        with pytest.raises(EndOfStream), second.receive(timeout=0.0):
            pass
    finally:
        publisher.close()
        thread.join(timeout=3.0)


def test_publish_timeout_does_not_partially_deliver() -> None:
    publisher = Publisher[bool]()
    first = publisher.subscribe(capacity=1)
    second = publisher.subscribe(capacity=1)
    original = Product(False)

    publisher.publish(original, timeout=0.0)

    # First now has capacity, but second remains full.
    with first.receive(timeout=0.0) as received:
        assert received is original

    with pytest.raises(TimeoutError, match="Publication exceeded the timeout"):
        publisher.publish(Product(True), timeout=0.0)

    # The failed publication wasn't delivered to first.
    with pytest.raises(TimeoutError): # ruff: ignore[multiple-with-statements]
        with first.receive(timeout=0.0):
            pass

    # Second still contains only the original product.
    with second.receive(timeout=0.0) as received:
        assert received is original

    with pytest.raises(TimeoutError): # ruff: ignore[multiple-with-statements]
        with second.receive(timeout=0.0):
            pass


def test_subscription_close_discards_queued_products() -> None:
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

    with pytest.raises(EndOfStream): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


def test_subscription_close_wakes_blocked_receiver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def receive() -> None:
        try:
            with subscription.receive(timeout=2.0):
                pass
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=receive)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        subscription.close()
        thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], EndOfStream)
    finally:
        subscription.close()
        thread.join(timeout=3.0)


def test_closing_one_subscription_preserves_other_deliveries() -> None:
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

    with pytest.raises(EndOfStream): # ruff: ignore[multiple-with-statements]
        with first.receive(timeout=0.0):
            pass

    with second.receive(timeout=0.0) as received:
        assert received is following


def test_subscription_close_unblocks_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publisher = Publisher[bool]()
    slow = publisher.subscribe(capacity=1)
    active = publisher.subscribe(capacity=1)
    original = Product(False)
    following = Product(True)

    publisher.publish(original, timeout=0.0)

    # Only slow remains full.
    with active.receive(timeout=0.0):
        pass

    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish(following, timeout=2.0)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=publish)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        slow.close()
        thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert errors == []

        with pytest.raises(EndOfStream): # ruff: ignore[multiple-with-statements]
            with slow.receive(timeout=0.0):
                pass

        with active.receive(timeout=0.0) as received:
            assert received is following

        with pytest.raises(TimeoutError): # ruff: ignore[multiple-with-statements]
            with active.receive(timeout=0.0):
                pass
    finally:
        slow.close()
        active.close()
        thread.join(timeout=3.0)


def test_subscription_close_preserves_borrowed_product() -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=2)
    first = Product(False)

    publisher.publish(first, timeout=0.0)
    publisher.publish(Product(True), timeout=0.0)

    with subscription.receive(timeout=0.0) as borrowed:
        subscription.close()

        assert borrowed is first
        assert borrowed.payload is False

    with pytest.raises(EndOfStream): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


def test_repeated_subscription_close_is_harmless() -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    subscription.close()
    subscription.close()

    # A detached subscriber must not block publication.
    publisher.publish(Product(False), timeout=0.0)

    subscription.close()

    with pytest.raises(EndOfStream): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass

    publisher.close()
    subscription.close()


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_discards_queued_products(scope: str) -> None:
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

    with pytest.raises(OperationCancelled): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


def test_subscription_abort_preserves_other_deliveries() -> None:
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

    with pytest.raises(OperationCancelled): # ruff: ignore[multiple-with-statements]
        with aborted.receive(timeout=0.0):
            pass


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_wakes_blocked_receiver(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()
    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def receive() -> None:
        try:
            with subscription.receive(timeout=2.0):
                pass
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=receive)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        target = subscription if scope == "subscription" else publisher
        target.abort()
        thread.join(timeout=3.0)

        assert not thread.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], OperationCancelled)
    finally:
        subscription.close()
        thread.join(timeout=3.0)


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_wakes_blocked_publisher(
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
) -> None:
    publisher = Publisher[bool]()
    slow = publisher.subscribe(capacity=1)
    active = publisher.subscribe(capacity=1)
    following = Product(True)

    publisher.publish(Product(False), timeout=0.0)

    # Leave only slow's queue full.
    with active.receive(timeout=0.0):
        pass

    waiting = _observe_wait(publisher._condition, monkeypatch)
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            publisher.publish(following, timeout=2.0)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=publish)
    thread.start()

    try:
        assert waiting.wait(timeout=2.0)

        target = slow if scope == "subscription" else publisher
        target.abort()
        thread.join(timeout=3.0)

        assert not thread.is_alive()

        with pytest.raises(OperationCancelled): # ruff: ignore[multiple-with-statements]
            with slow.receive(timeout=0.0):
                pass

        if scope == "subscription":
            assert errors == []

            with active.receive(timeout=0.0) as received:
                assert received is following
        else:
            assert len(errors) == 1
            assert isinstance(errors[0], OperationCancelled)

            with pytest.raises(OperationCancelled): # ruff: ignore[multiple-with-statements]
                with active.receive(timeout=0.0):
                    pass
    finally:
        publisher.close()
        slow.close()
        active.close()
        thread.join(timeout=3.0)


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_preserves_borrowed_product(scope: str) -> None:
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

    with pytest.raises(OperationCancelled): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


@pytest.mark.parametrize("scope", ["subscription", "publisher"])
def test_abort_overrides_normal_close_and_is_idempotent(scope: str) -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe()

    publisher.publish(Product(False), timeout=0.0)
    publisher.close()

    target = subscription if scope == "subscription" else publisher
    target.abort()
    target.abort()

    subscription.close()
    publisher.close()

    with pytest.raises(OperationCancelled): # ruff: ignore[multiple-with-statements]
        with subscription.receive(timeout=0.0):
            pass


def test_aborted_publisher_rejects_publish_and_subscribe() -> None:
    publisher = Publisher[bool]()
    publisher.abort()

    with pytest.raises(OperationCancelled):
        publisher.publish(Product(False), timeout=0.0)

    with pytest.raises(OperationCancelled):
        publisher.subscribe()


@pytest.mark.parametrize("operation", ["receive", "publish"])
def test_worker_abort_interrupts_channel_operation(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    publisher = Publisher[bool]()
    subscription = publisher.subscribe(capacity=1)

    if operation == "publish":
        publisher.publish(Product(False), timeout=0.0)

    class ChannelWorker(StubWorker):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        def _work(self) -> None:
            if operation == "receive":
                with subscription.receive(timeout=2.0):
                    pass
            else:
                publisher.publish(Product(True), timeout=2.0)

        def _request_abort(self) -> None:
            # Record the request before waking the worker.
            super()._request_abort()

            if operation == "receive":
                subscription.abort()
            else:
                publisher.abort()

        def _close(self) -> None:
            self.close_calls += 1

            if operation == "receive":
                subscription.close()
            else:
                publisher.close()

    worker = ChannelWorker()
    pipeline = Pipeline(worker)
    waiting = _observe_wait(publisher._condition, monkeypatch)
    pipeline.start()

    try:
        assert waiting.wait(timeout=2.0)

        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert worker._exception is None
        assert worker.close_calls == 1
    finally:
        worker._abort_requested.set()
        publisher.abort()
        worker._wait(timeout=2.0)


def test_unrequested_cancellation_is_a_worker_failure() -> None:
    failure = OperationCancelled("unexpected cancellation")
    calls: list[str] = []

    class UnexpectedCancellationWorker(StubWorker):
        def _work(self) -> None:
            calls.append("work")
            raise failure

        def _close(self) -> None:
            calls.append("close")

    worker = UnexpectedCancellationWorker()
    pipeline = Pipeline(worker)
    pipeline.start()

    with pytest.raises(OperationCancelled) as caught:
        pipeline.wait(timeout=2.0)

    assert caught.value is failure
    assert pipeline.exception is failure
    assert pipeline.done
    assert calls == ["work", "close"]


def test_work_failure_after_abort_is_not_suppressed() -> None:
    failure = RuntimeError("work failed during cancellation")

    class FailingWorker(StubWorker):
        def _work(self) -> None:
            self._abort_requested.wait()
            raise failure

    worker = FailingWorker()
    pipeline = Pipeline(worker)
    pipeline.start()

    try:
        pipeline.abort()

        with pytest.raises(RuntimeError) as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is failure
        assert pipeline.exception is failure
        assert pipeline.done
    finally:
        worker._abort_requested.set()
        worker._wait(timeout=2.0)


@pytest.mark.parametrize(
    "cleanup_failure",
    [
        OSError("cleanup failed"),
        OperationCancelled("cleanup interrupted"),
    ],
)
def test_cleanup_failure_after_cancellation_is_reported(
    cleanup_failure: BaseException,
) -> None:
    class CleanupFailureWorker(StubWorker):
        def _work(self) -> None:
            self._abort_requested.wait()
            raise OperationCancelled("worker cancelled")

        def _close(self) -> None:
            raise cleanup_failure

    worker = CleanupFailureWorker()
    pipeline = Pipeline(worker)
    pipeline.start()

    try:
        pipeline.abort()

        with pytest.raises(type(cleanup_failure)) as caught:
            pipeline.wait(timeout=2.0)

        assert caught.value is cleanup_failure
        assert pipeline.exception is cleanup_failure
        assert worker._close_exception is cleanup_failure
        assert pipeline.done
    finally:
        worker._abort_requested.set()
        worker._wait(timeout=2.0)


def test_workers_deliver_products_and_finish() -> None:
    source = BooleanSource(values=(False, True))
    sink = CollectingWorker(upstream=source)
    pipeline = Pipeline(sink)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        assert sink.values == [False, True]
        assert source.published == 2
        assert pipeline.done
        assert pipeline.exception is None

        # Successful completion must close the source's output.
        with pytest.raises(RuntimeError, match="closed"):
            source._publish(False, timeout=0.0)
    finally:
        pipeline.abort()
        for worker in pipeline.workers:
            worker._wait(timeout=2.0)


def test_consumer_exit_does_not_block_source() -> None:
    class FirstOnlyWorker(CollectingWorker):
        def _work(self) -> None:
            with self._receive(timeout=2.0) as product:
                self.values.append(product.payload)

    source = BooleanSource(values=(False, True, False, True))
    sink = FirstOnlyWorker(upstream=source)
    pipeline = Pipeline(sink)
    pipeline.start()

    try:
        pipeline.wait(timeout=2.0)

        assert sink.values == [False]
        assert source.published == 4
        assert pipeline.done
        assert pipeline.exception is None
    finally:
        pipeline.abort()
        for worker in pipeline.workers:
            worker._wait(timeout=2.0)


def test_pipeline_abort_wakes_worker_receiving_from_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IdleSource(StubWorker[Never, bool]):
        def _work(self) -> None:
            self._abort_requested.wait(timeout=2.0)

    source = IdleSource()
    sink = CollectingWorker(upstream=source)
    pipeline = Pipeline(sink)

    # Observe the sink waiting on its upstream channel.
    waiting = _observe_wait(source._publisher._condition, monkeypatch)
    pipeline.start()

    try:
        assert waiting.wait(timeout=2.0)

        pipeline.abort()
        pipeline.wait(timeout=2.0)

        assert pipeline.done
        assert pipeline.exception is None
        assert source._exception is None
        assert sink._exception is None
    finally:
        pipeline.abort()
        for worker in pipeline.workers:
            worker._wait(timeout=2.0)


def test_pipeline_marks_all_workers_before_waking_channels() -> None:
    class CheckingWorker(StubWorker):
        def _request_abort(self) -> None:
            assert all(
                worker._abort_requested.is_set()
                for worker in workers
            )
            super()._request_abort()

    first = CheckingWorker()
    second = CheckingWorker(upstream=first)
    workers = (first, second)
    pipeline = Pipeline(second)

    pipeline.start()
    pipeline.wait(timeout=2.0)
    pipeline.abort()


def test_worker_receive_requires_upstream() -> None:
    worker = StubWorker()

    with pytest.raises(RuntimeError, match="no upstream"):
        worker._receive(timeout=0.0)

