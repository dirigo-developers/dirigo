from collections import deque
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from threading import Condition

from .exceptions import EndOfStream, OperationCancelled
from .product import Product


class Subscription[T]:
    """A bounded delivery channel from one publisher to one consumer."""

    def __init__(
        self,
        *,
        capacity: int = 1,
        _condition: Condition,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self._capacity = capacity
        self._products: deque[Product[T]] = deque() # used as FIFO container
        self._condition = _condition
        self._closed: bool = False
        self._aborted: bool = False
        self._producer_closed: bool = False

    def receive(
        self,
        timeout: float | None = None,
    ) -> AbstractContextManager[Product[T]]:
        """Receive the next queued product through a context manager.

        Waiting begins on context entry. None waits indefinitely; zero
        checks without waiting. Receiving frees queue capacity before
        consumer processing begins.

        Exceptions raised inside the context are not suppressed.

        Raises:
            ValueError: If timeout is negative.
            TimeoutError: If no product arrives before the timeout.
            EndOfStream: If receiving has ended and the queue is empty.
        """
        return self._receive(timeout)

    @contextmanager
    def _receive(
        self,
        timeout: float | None,
    ) -> Generator[Product[T]]:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative or None.")

        with self._condition:
            available = self._condition.wait_for(
                lambda: (
                    self._closed
                    or self._producer_closed
                    or bool(self._products)
                ),
                timeout=timeout,
            )

            if self._aborted:
                raise OperationCancelled("Subscription was aborted.")

            if not available:
                raise TimeoutError("No product arrived before the timeout.")

            if self._closed or not self._products:
                raise EndOfStream

            product = self._products.popleft()
            self._condition.notify_all()

        # Consumer processing takes place outside the channel lock.
        yield product

    def close(self) -> None:
        """Detach this consumer and discard its queued deliveries.

        Borrowed products remain valid until their contexts exit.
        Repeated calls are harmless.
        """
        with self._condition:
            if self._closed:
                return

            self._closed = True
            self._products.clear()
            self._condition.notify_all()

    def abort(self) -> None:
        """Cancel receiving, discard queued deliveries, and wake waiters.

        Borrowed products remain valid. Repeated calls are harmless.
        """
        with self._condition:
            self._abort()
            self._condition.notify_all()

    def _abort(self) -> None:
        """Record cancellation. Caller must hold the shared condition lock."""
        if self._aborted:
            return

        self._aborted = True
        self._closed = True
        self._products.clear()

    def _ready_for_publish(self) -> bool:
        """Whether this subscription permits publication to proceed."""
        return self._closed or len(self._products) < self._capacity

    def _enqueue(self, product: Product[T]) -> None:
        """Queue a delivery unless this consumer has detached."""
        if not self._closed:
            self._products.append(product)

    def _finish(self) -> None:
        """Record normal producer completion without discarding products."""
        self._producer_closed = True
