from collections.abc import Callable
from threading import Condition


class _Ownership[T]:
    """One acquisition's shared ownership record, initially held by its producer.

    The condition is the pool's condition. Recycling runs with that lock held.
    This record is never reused for a subsequent acquisition of the same buffer.
    """

    def __init__(
        self,
        *,
        payload: T,
        condition: Condition,
        recycle: Callable[[T], None],
    ) -> None:
        self.payload = payload
        self._condition = condition
        self._recycle = recycle
        self._references = 1

    def retain(self, count: int) -> None:
        with self._condition:
            self._retain_locked(count)

    def _retain_locked(self, count: int) -> None:
        """Caller must hold the shared condition lock."""
        if count < 0:
            raise ValueError("Reference count must be nonnegative.")
        if self._references == 0:
            raise RuntimeError("Product ownership has been released.")
        self._references += count

    def release(self) -> None:
        with self._condition:
            self._release_locked()

    def _release_locked(self) -> None:
        """Caller must hold the shared condition lock."""
        if self._references == 0:
            raise RuntimeError("Product ownership has already been released.")
        self._references -= 1
        if self._references == 0:
            self._recycle(self.payload)
