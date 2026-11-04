from threading import Event

from dirigo.pipeline import Worker


class StubWorker[InputT = object, OutputT = object](Worker[InputT, OutputT]):
    """Implement the abstract hooks without adding application behavior."""

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        pass

    def _close(self) -> None:
        pass


class BlockingWorker(StubWorker):
    """Keep execution active until the test explicitly permits completion."""

    def __init__(self) -> None:
        super().__init__()
        self.release = Event()

    def _work(self) -> None:
        self.release.wait()


class AbortableWorker(StubWorker):
    """Wait for cancellation and count completed cleanup calls."""

    def __init__(
        self,
        *,
        upstream: Worker | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(upstream=upstream, name=name)
        self.entered = Event()
        self.close_calls = 0

    def _work(self) -> None:
        self.entered.set()
        self._abort_requested.wait()

    def _close(self) -> None:
        self.close_calls += 1
