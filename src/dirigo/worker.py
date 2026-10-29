from abc import ABC, abstractmethod
from collections.abc import Callable
from threading import Event, Thread


class Worker(ABC):
    def __init__(
        self,
        *,
        upstream: Worker | None = None,
        name: str | None = None,
    ) -> None:
        if upstream is not None and not isinstance(upstream, Worker):
            raise TypeError("upstream must be a Worker or None.")

        self._upstream: Worker | None = upstream
        self._name: str = name if name is not None else type(self).__name__

        self._thread: Thread | None = None
        self._abort_requested = Event()
        self._finished = Event()

        self._exception: BaseException | None = None
        self._close_exception: BaseException | None = None

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
        """Flush and release resources, including after partial preparation."""

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
        """Execute work, then close resources and signal completion."""
        try:
            try:
                self._work()
            except BaseException as exc:
                self._exception: BaseException = exc
                if on_failure is not None:
                    on_failure(exc)
        finally:
            try:
                self._close()
            except BaseException as exc:
                self._close_exception: BaseException = exc

                if self._exception is None:
                    self._exception: BaseException = exc
                    if on_failure is not None:
                        on_failure(exc)
                else:
                    self._exception.add_note(
                        f"Cleanup also failed in worker {self.name!r}: {exc!r}"
                    )
            finally:
                self._finished.set()

    def _request_abort(self) -> None:
        """Request cooperative cancellation."""
        self._abort_requested.set()
