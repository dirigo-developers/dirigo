from collections import deque
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from threading import Condition

from ._ownership import _Ownership
from .exceptions import OperationCancelled
from .product import _EMPTY_METADATA, Metadata, Product


class Lease[T]:
    """A producer's claim to a pooled payload.

    Successful publish_lease() transfers this claim. Releasing or leaving the
    acquisition context afterward is harmless. Raw aliases must not be used
    after release or transfer. Every fresh lease starts with empty metadata,
    even when its payload storage has been used before.
    """

    def __init__(self, *, _pool: Pool[T], _ownership: _Ownership[T]) -> None:
        self._pool = _pool
        self._ownership = _ownership
        self._released = False
        self._metadata = _EMPTY_METADATA

    @property
    def payload(self) -> T:
        """Return the payload while this lease remains active."""
        with self._pool._condition:
            self._raise_if_released()
            return self._ownership.payload

    @property
    def metadata(self) -> Metadata:
        """Return this acquisition's metadata while the producer lease is active."""
        with self._pool._condition:
            self._raise_if_released()
            return self._metadata

    @metadata.setter
    def metadata(self, metadata: Metadata) -> None:
        """Replace metadata before publication; the record itself stays frozen."""
        with self._pool._condition:
            self._raise_if_released()
            if not isinstance(metadata, Metadata):
                raise TypeError("metadata must be Metadata.")
            self._metadata = metadata

    def _raise_if_released(self) -> None:
        """Caller must hold the pool condition lock."""
        if self._released:
            raise RuntimeError("Lease has been released or transferred.")

    def release(self) -> None:
        """Return this lease's claim exactly once."""
        with self._pool._condition:
            if self._released:
                return
            self._released = True
            self._ownership._release_locked()

    def _transfer(self, deliveries: int) -> Product[T]:
        """Capture payload/metadata and reserve deliveries under the pool lock."""
        with self._pool._condition:
            self._raise_if_released()
            product = Product(
                self._ownership.payload,
                _ownership=self._ownership,
                metadata=self._metadata,
            )
            self._ownership._retain_locked(deliveries)
            self._released = True
            self._ownership._release_locked()
            return product


class Pool[T]:
    """A fixed-capacity pool of independently reusable payloads."""

    def __init__(self, *, capacity: int, factory: Callable[[], T]) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self._condition = Condition()
        self._aborted = False
        self._available: deque[T] = deque(factory() for _ in range(capacity))

    def acquire(
        self,
        timeout: float | None = None,
    ) -> AbstractContextManager[Lease[T]]:
        """Return a context manager granting an exclusive producer lease.

        Waiting begins on entering the context. None waits indefinitely;
        zero checks without waiting.

        Exiting releases the lease, including when the body raises.
        """
        return self._acquire(timeout)

    @contextmanager
    def _acquire(
        self,
        timeout: float | None,
    ) -> Generator[Lease[T]]:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative or None.")

        with self._condition:
            available = self._condition.wait_for(
                lambda: self._aborted or bool(self._available), timeout=timeout
            )
            if self._aborted:
                raise OperationCancelled("Pool was aborted.")
            if not available:
                raise TimeoutError("Pool acquisition exceeded the timeout.")

            ownership = _Ownership(
                payload=self._available.popleft(),
                condition=self._condition,
                recycle=self._return,
            )
            lease = Lease(_pool=self, _ownership=ownership)

        try:
            yield lease
        finally:
            lease.release()

    def _return(self, payload: T) -> None:
        """Return a payload. Caller must hold the condition lock."""
        self._available.append(payload)
        self._condition.notify_all()

    def abort(self) -> None:
        """Permanently cancel acquisition and wake blocked acquirers.

        Existing leases remain valid until released.
        Repeated calls are harmless.
        """
        with self._condition:
            if self._aborted:
                return

            self._aborted = True
            self._condition.notify_all()
