from threading import Event

from dirigo.worker import Worker


class StubWorker(Worker):
    """Minimal concrete worker for testing inherited behavior."""

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        pass

    def _close(self) -> None:
        pass


class BlockingWorker(StubWorker):
    """Tiny concrete worker that blocks in work until .release Event is set."""
    def __init__(self) -> None:
        super().__init__()
        self.release = Event()

    def _work(self) -> None:
        self.release.wait()


class PreparationWorker(StubWorker):
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        upstream: Worker | None = None,
        prepare_failure: BaseException | None = None,
        close_failure: BaseException | None = None,
    ) -> None:
        super().__init__(name=name, upstream=upstream)
        self.calls: list[str] = calls
        self.prepare_failure: BaseException | None = prepare_failure
        self.close_failure: BaseException | None = close_failure

    def _prepare(self) -> None:
        self.calls.append(f"{self.name}.prepare")
        if self.prepare_failure is not None:
            raise self.prepare_failure

    def _work(self) -> None:
        self.calls.append(f"{self.name}.work")

    def _close(self) -> None:
        self.calls.append(f"{self.name}.close")
        if self.close_failure is not None:
            raise self.close_failure


class AbortableWorker(StubWorker):
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


class SlowClosingWorker(AbortableWorker):
    def __init__(
        self,
        *,
        upstream: Worker | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(upstream=upstream, name=name)
        self.closing = Event()
        self.release_close = Event()

    def _close(self) -> None:
        self.closing.set()
        # Bounded so an incorrectly blocking abort won't hang the test.
        self.release_close.wait(timeout=2.0)
        super()._close()


class TriggeredFailureWorker(StubWorker):
    def __init__(
        self,
        failure: BaseException,
        *,
        fail_on_close: bool = False,
    ) -> None:
        super().__init__()
        self.failure = failure
        self.fail_on_close = fail_on_close
        self.entered = Event()
        self.fail_now = Event()
        self.close_calls = 0

    def _work(self) -> None:
        self.entered.set()
        self.fail_now.wait()

        if not self.fail_on_close:
            raise self.failure

    def _close(self) -> None:
        self.close_calls += 1

        if self.fail_on_close:
            raise self.failure

