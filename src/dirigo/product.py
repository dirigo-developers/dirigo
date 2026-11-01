from dataclasses import dataclass, field

from ._ownership import _Ownership


@dataclass(frozen=True, slots=True, eq=False)
class Product[T]:
    """A payload shared by downstream deliveries.

    Treat the payload as read-only after publication. A pooled payload is valid
    only while an ownership claim is held, normally inside receive().
    """

    payload: T
    _ownership: _Ownership[T] | None = field(default=None, repr=False)

    def _retain_deliveries(self, count: int) -> None:
        if self._ownership is not None:
            self._ownership.retain(count)

    def _release_delivery(self) -> None:
        if self._ownership is not None:
            self._ownership.release()
