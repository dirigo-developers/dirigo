from threading import Condition

from .exceptions import OperationCancelled
from .product import Product
from .subscription import Subscription


class Publisher[T]:
    """Broadcast products to bounded subscriber queues."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._subscriptions: list[Subscription[T]] = []

        self._closed: bool = False
        self._aborted: bool = False

    def subscribe(self, *, capacity: int = 1) -> Subscription[T]:
        """Create a subscription for subsequent publications.

        Publications already waiting for capacity exclude this subscription.
        """
        with self._condition:
            self._raise_if_aborted_or_closed()

            subscription = Subscription[T](
                capacity=capacity,
                _condition=self._condition,
            )
            self._subscriptions.append(subscription)
            return subscription

    def publish(
        self,
        product: Product[T],
        timeout: float | None = None,
    ) -> None:
        """Broadcast a product once every recipient queue has capacity.

        Recipients are selected before waiting. On timeout, no recipient
        receives the product. With no subscribers, nothing is retained.

        None waits indefinitely for capacity; zero does not wait for capacity.

        Raises:
            TypeError: If product is not a Product.
            ValueError: If timeout is negative.
            TimeoutError: If waiting for capacity exceeds the timeout.
            RuntimeError: If the publisher is closed.
        """
        if not isinstance(product, Product):
            raise TypeError("product must be a Product.")

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative or None.")

        with self._condition:
            self._raise_if_aborted_or_closed()
            subscriptions = tuple(self._subscriptions)

            available: bool = self._condition.wait_for(
                lambda: self._closed or all(
                    subscription._ready_for_publish()
                    for subscription in subscriptions
                ),
                timeout=timeout,
            )

            self._raise_if_aborted_or_closed()

            if not available:
                raise TimeoutError("Publication exceeded the timeout.")

            for subscription in subscriptions:
                subscription._enqueue(product)

            self._condition.notify_all()

    def close(self) -> None:
        """End publication while allowing queued products to drain.

        Wake blocked receivers and publishers. Repeated calls are harmless.
        """
        with self._condition:
            if self._closed:
                return

            self._closed = True

            for subscription in self._subscriptions:
                subscription._finish()

            self._condition.notify_all()

    def abort(self) -> None:
        """Cancel publication and every subscription.

        Discard queued deliveries and wake blocked operations.
        Repeated calls are harmless.
        """
        with self._condition:
            if self._aborted:
                return

            self._aborted = True
            self._closed = True

            for subscription in self._subscriptions:
                subscription._abort()

            self._condition.notify_all()

    def _raise_if_aborted_or_closed(self) -> None:
        """Raise if publication has ended or been cancelled."""
        if self._aborted:
            raise OperationCancelled("Publisher was aborted.")

        if self._closed:
            raise RuntimeError("Publisher is closed.")
