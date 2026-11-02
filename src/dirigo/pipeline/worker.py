from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from threading import Event, Thread
from typing import Any

from .exceptions import OperationCancelled
from .pool import Lease, Pool
from .product import Product
from .publisher import Publisher
from .subscription import Subscription


class Worker[InputT = object, OutputT = object](ABC):
    """A single-use pipeline stage with a dedicated execution thread."""

    def __init__(
        self,
        *,
        upstream: Worker[Any, InputT] | None = None,
        name: str | None = None,
    ) -> None:
        if upstream is not None and not isinstance(upstream, Worker):
            raise TypeError("upstream must be a Worker or None.")

        self._upstream: Worker[Any, InputT] | None = upstream
        self._name: str = name if name is not None else type(self).__name__

        self._pipeline_token: object | None = None

        self._thread: Thread | None = None
        self._finished = Event()
        self._abort_requested = Event()
        self._exception: BaseException | None = None
        self._close_exception: BaseException | None = None

        self._pool: Pool[OutputT] | None = None

        self._publisher = Publisher[OutputT]()
        self._subscription: Subscription[InputT] | None = None

        if upstream is not None:
            self._subscription = upstream._publisher.subscribe(capacity=1)

    @property
    def upstream(self) -> Worker[Any, InputT] | None:
        return self._upstream

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _prepare(self) -> None:
        """Prepare resources on the thread calling Pipeline.start()."""

    @abstractmethod
    def _work(self) -> None:
        """On dedicated thread, produce or consume products until completion."""

    @abstractmethod
    def _close(self) -> None:
        """Release resources, including after partial preparation.

        May run on the worker thread or the pipeline's calling thread.
        """

    def _init_pool(self, *, capacity: int, factory: Callable[[], OutputT]) -> None:
        """Create this worker's exclusively owned output pool.

        Usually called from _prepare(). Initialization is allowed once.
        """
        if self._pool is not None:
            raise RuntimeError("Output pool is already initialized.")

        pool = Pool[OutputT](capacity=capacity, factory=factory)
        self._pool = pool

        # Cancellation may have arrived while the factory was running.
        if self._abort_requested.is_set():
            pool.abort()

    def _acquire(
        self,
        timeout: float | None = None,
    ) -> AbstractContextManager[Lease[OutputT]]:
        """Borrow writable output storage through a context manager."""
        pool = self._pool
        if pool is None:
            raise RuntimeError("Worker has no output pool.")

        return pool.acquire(timeout=timeout)

    def _start(
        self,
        *,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Launch execution on this worker's thread."""
        if self._thread is not None:
            raise RuntimeError("Worker execution has already started.")

        self._thread = Thread(
            target=self._run,
            args=(on_failure,),
            name=self.name,
            daemon=False,
        )
        self._thread.start()

    def _wait(self, timeout: float | None = None) -> None:
        """Join this worker's execution thread."""
        if self._thread is None:
            raise RuntimeError("Worker execution has not started.")

        self._thread.join(timeout)

        if self._thread.is_alive():
            raise TimeoutError("Worker did not finish before the timeout.")

    def _run(
        self,
        on_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Wait for the execution thread and its cleanup to finish."""
        try:
            try:
                self._work()
            except BaseException as exc:
                expected_cancellation = (
                    isinstance(exc, OperationCancelled)
                    and self._abort_requested.is_set()
                )

                if not expected_cancellation:
                    self._exception = exc

                    if on_failure is not None:
                        on_failure(exc)
        finally:
            try:
                self._close_worker()
            except BaseException as exc:
                self._close_exception = exc

                if self._exception is None:
                    self._exception = exc

                    if on_failure is not None:
                        on_failure(exc)
                else:
                    self._exception.add_note(
                        f"Cleanup also failed in worker {self.name!r}: {exc!r}"
                    )
            finally:
                self._finished.set()

    def _publish(
        self,
        payload: OutputT | Lease[OutputT] | Product[OutputT],
        timeout: float | None = None,
    ) -> None:
        """Publish a new value/lease, or forward an existing borrowed product.

        Leases and Products already carry metadata. Plain payloads are shorthand
        for an unpooled Product with empty metadata. Use dataclasses.replace()
        to create a new metadata envelope while borrowing an existing Product.
        Product and Lease are envelope/ownership handles, not plain payload types.
        """
        if isinstance(payload, Product):
            self._publisher.publish(payload, timeout=timeout)
        elif isinstance(payload, Lease):
            self._publisher.publish_lease(payload, timeout=timeout)
        else:
            self._publisher.publish(Product(payload), timeout=timeout)

    def _receive(
        self,
        timeout: float | None = None,
    ) -> AbstractContextManager[Product[InputT]]:
        """Borrow the next product from this worker's upstream."""
        if self._subscription is None:
            raise RuntimeError("Worker has no upstream subscription.")

        return self._subscription.receive(timeout=timeout)

    def _mark_abort_requested(self) -> None:
        """Record cancellation without waking channel operations."""
        self._abort_requested.set()

    def _request_abort(self) -> None:
        """Record cancellation and wake this worker's channel operations."""
        self._mark_abort_requested()

        pool = self._pool
        if pool is not None:
            pool.abort()

        self._publisher.abort()

        if self._subscription is not None:
            self._subscription.abort()

    def _close_worker(self) -> None:
        """Close resources and channels, including after partial preparation."""
        try:
            self._close()
        finally:
            try:
                if self._subscription is not None:
                    self._subscription.close()
            finally:
                try:
                    self._publisher.close()
                finally:
                    pool = self._pool
                    if pool is not None:
                        pool.abort()
