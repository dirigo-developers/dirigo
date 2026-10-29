from threading import Lock
from time import monotonic

from .worker import Worker


class Pipeline:
    """A single-use execution graph of workers."""

    def __init__(self, *workers: Worker) -> None:
        self._started: bool = False
        self._workers: list[Worker] = []
        self._launched: list[Worker] = []

        self._exception: BaseException | None = None
        self._exception_lock = Lock()

        self._preparation_failed = False # preparation failed and cleanup has completed

        for worker in workers:
            self.add(worker)

    @property
    def workers(self) -> tuple[Worker, ...]:
        """All included workers in topological order.

        Each addition appends new workers upstream first, skipping duplicates.
        Adding branches a2 and b2 with a shared source yields:
            (source, a1, a2, b1, b2)
        """
        return tuple(self._workers)

    def add(self, worker: Worker) -> None:
        """Include a worker and its upstream dependencies.

        Workers are included once, by identity, with dependencies before their
        consumers.

        Raises:
            ValueError: If the upstream chain contains a cycle.
        """
        if self._started:
            raise RuntimeError("Cannot add workers after execution has started.")

        if not isinstance(worker, Worker):
            raise TypeError("worker must be a Worker.")

        existing_ids: set[int] = {id(item) for item in self._workers}
        visited_ids: set[int] = set()
        pending: list[Worker] = []

        current: Worker | None = worker

        while current is not None:
            current_id: int = id(current)

            if current_id in visited_ids:
                raise ValueError("Worker upstream chain contains a cycle.")

            if current_id in existing_ids:
                break

            visited_ids.add(current_id)
            pending.append(current)
            current: Worker | None = current.upstream

        self._workers.extend(reversed(pending))

    def start(self) -> None:
        """Prepare and launch the graph without waiting for completion.

        Prepare workers upstream first, then launch consumers before
        producers. The graph becomes fixed when execution starts.

        Raises:
            RuntimeError: If execution has already started.
        """
        if self._started:
            raise RuntimeError("Pipeline execution has already started.")

        if not self._workers:
            raise ValueError("Cannot start an empty pipeline.")

        self._started = True
        attempted: list[Worker] = []

        try:
            for worker in self._workers:
                attempted.append(worker)
                worker._prepare()

        except BaseException as exc:
            self._record_exception(exc)

            try:
                self._close_workers(workers=attempted, failure=exc)
            finally:
                self._preparation_failed = True

            raise

        try:
            for worker in reversed(self._workers):
                worker._start(on_failure=self._record_exception)
                self._launched.append(worker)
        except BaseException as exc:
            # Recording the failure also requests cancellation.
            self._record_exception(exc)

            launched_ids: set[int] = {id(worker) for worker in self._launched}
            unlaunched: list[Worker] = [
                worker for worker in self._workers
                if id(worker) not in launched_ids
            ]

            self._close_workers(workers=unlaunched, failure=exc)
            raise

    def stop(self) -> None:
        """Request graceful completion and return without waiting.

        Stop acquisition while allowing downstream workers to drain
        pending products, flush output, and release resources.
        """
        raise NotImplementedError

    def abort(self) -> None:
        """Request cancellation without waiting for cleanup.

        Repeated requests, including after completion, are harmless.

        Raises:
            RuntimeError: If execution has not started.
        """
        if not self._started:
            raise RuntimeError("Pipeline execution has not started.")

        for worker in self._workers:
            worker._request_abort()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for all workers to finish and cleanup to complete.

        A timeout does not stop or abort the pipeline.

        Raises:
            RuntimeError: If execution has not started.
            TimeoutError: If completion exceeds the timeout.

        A worker failure is re-raised to the caller.
        """
        if not self._started:
            raise RuntimeError("Pipeline execution has not started.")

        deadline: float | None = None if timeout is None else monotonic() + timeout

        for worker in self._launched:
            remaining: float | None = (
                None
                if deadline is None
                else max(0.0, deadline - monotonic())
            )
            worker._wait(timeout=remaining)

        failure: BaseException | None = self.exception
        if failure is not None:
            raise failure

    @property
    def done(self) -> bool:
        """Whether execution and cleanup have completed.

        False before execution starts.
        """
        return self._started and (
            self._preparation_failed
            or all(worker._finished.is_set() for worker in self._workers)
        )

    @property
    def exception(self) -> BaseException | None:
        """The first recorded worker failure, or None."""
        with self._exception_lock:
            return self._exception

    def _record_exception(self, exc: BaseException) -> None:
        """Record the first failure and request graph-wide cancellation."""
        with self._exception_lock:
            if self._exception is not None:
                return

            self._exception: BaseException = exc

        self.abort()

    def _close_workers(
        self,
        workers: list[Worker],
        failure: BaseException,
    ) -> None:
        """Close prepared, unlaunched workers in reverse order."""
        for worker in reversed(workers):
            try:
                worker._close()
            except BaseException as exc:
                worker._close_exception = exc
                failure.add_note(
                    f"Cleanup also failed in worker {worker.name!r}: {exc!r}"
                )
            finally:
                worker._finished.set()

