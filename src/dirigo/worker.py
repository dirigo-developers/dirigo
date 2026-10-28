from abc import ABC, abstractmethod


class Worker(ABC):
    def __init__(
        self,
        *,
        upstream: Worker | None = None,
        name: str | None = None,
    ) -> None:
        self._upstream: Worker | None = upstream
        self._name: str = name if name is not None else type(self).__name__

    @property
    def upstream(self) -> Worker | None:
        return self._upstream

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def _prepare(self) -> None:
        """Prepare resources needed for execution."""

    @abstractmethod
    def _work(self) -> None:
        """Produce or consume products until completion."""

    @abstractmethod
    def _close(self) -> None:
        """Flush and release resources acquired by this worker."""
