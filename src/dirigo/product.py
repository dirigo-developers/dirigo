from dataclasses import dataclass, field

from ._ownership import _Ownership


@dataclass(frozen=True, slots=True)
class Metadata:
    """Immutable publication identifiers; None means unspecified.

    Use dataclasses.replace() to inherit a record with selected changes.
    """

    sequence_index: int | None = None


_EMPTY_METADATA = Metadata()


@dataclass(frozen=True, slots=True, eq=False)
class Product[T]:
    """A payload shared by downstream deliveries.

    Treat the payload as read-only after publication. A pooled payload is valid
    only while an ownership claim is held, normally inside receive().
    """

    payload: T
    metadata: Metadata = field(default=_EMPTY_METADATA, kw_only=True)
    _ownership: _Ownership[T] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, Metadata):
            raise TypeError("metadata must be Metadata.")

    def _retain_deliveries(self, count: int) -> None:
        if self._ownership is not None:
            self._ownership.retain(count)

    def _release_delivery(self) -> None:
        if self._ownership is not None:
            self._ownership.release()
