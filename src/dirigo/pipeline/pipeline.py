from threading import Lock
from time import monotonic
from typing import Any

from dirigo.hardware.hardware import Hardware

from .hardware_worker import HardwareWorker
from .worker import Worker

type _AnyWorker = Worker[Any, Any]


class Pipeline:
    """A single-use execution graph of workers."""

    def __init__(self, *workers: _AnyWorker, hardware: Hardware | None = None) -> None:
        if hardware is not None and not isinstance(hardware, Hardware):
            raise TypeError("hardware must be a Hardware instance or None.")

        self._hardware = hardware

        self._ownership_token = object()

        self._started: bool = False
        self._workers: list[_AnyWorker] = []
        self._launched: list[_AnyWorker] = []

        self._exception: BaseException | None = None
        self._exception_lock = Lock()

        self._preparation_failed = False

        try:
            for worker in workers:
                self.add(worker)
        except BaseException:
            # A failed constructor must not strand earlier accepted workers.
            for worker in self._workers:
                worker._pipeline_token = None
            raise

    @property
    def workers(self) -> tuple[_AnyWorker, ...]:
        """All included workers in topological order.

        Each addition appends new workers upstream first, skipping duplicates.
        Adding branches a2 and b2 with a shared source yields:
            (source, a1, a2, b1, b2)
        """
        return tuple(self._workers)

    def add(self, worker: _AnyWorker) -> None:
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
        pending: list[_AnyWorker] = []

        current: _AnyWorker | None = worker

        while current is not None:
            current_id: int = id(current)

            if current_id in visited_ids:
                raise ValueError("Worker upstream chain contains a cycle.")

            if current_id in existing_ids:
                break

            if (
                current._pipeline_token is not None
                and current._pipeline_token is not self._ownership_token
            ):
                raise RuntimeError(
                    f"Worker {current.name!r} already belongs to another pipeline."
                )

            visited_ids.add(current_id)
            pending.append(current)
            current: Worker | None = current.upstream

        for item in pending:
            item._pipeline_token = self._ownership_token

        self._workers.extend(reversed(pending))

    def start(self) -> None:
        """Prepare and launch the graph without waiting for completion.

        Prepare workers upstream first, then launch consumers before
        producers. The graph becomes fixed when execution starts.

        Finish connecting workers before calling this method. Every active
        subscription must belong to an included worker. An incomplete graph
        is rejected before preparation and can be repaired with add().

        Raises:
            RuntimeError: If execution has already started.
            ValueError: If the graph is empty or has an omitted consumer.
        """
        if self._started:
            raise RuntimeError("Pipeline execution has already started.")

        if not self._workers:
            raise ValueError("Cannot start an empty pipeline.")

        self._validate_graph()

        self._started = True
        attempted: list[Worker] = []

        try:
            self._bind_hardware_workers()

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
                worker for worker in self._workers if id(worker) not in launched_ids
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
        """Request graph-wide cancellation without waiting for completion.

        Raises:
            RuntimeError: If execution has not started.
        """
        if not self._started:
            raise RuntimeError("Pipeline execution has not started.")

        for worker in self._workers:
            worker._mark_abort_requested()

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
                None if deadline is None else max(0.0, deadline - monotonic())
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
                worker._close_worker()
            except BaseException as exc:
                worker._close_exception = exc
                failure.add_note(
                    f"Cleanup also failed in worker {worker.name!r}: {exc!r}"
                )
            finally:
                worker._finished.set()

    def _validate_graph(self) -> None:
        """Reject active subscriptions whose consumers will not be launched."""
        subscription_ids = {
            id(worker._subscription)
            for worker in self._workers
            if worker._subscription is not None
        }

        for worker in self._workers:
            if any(
                id(subscription) not in subscription_ids
                for subscription in worker._publisher._active_subscriptions()
            ):
                raise ValueError(
                    f"Worker {worker.name!r} has an active subscription "
                    "outside this pipeline. Include every connected consumer "
                    "with Pipeline.add() before starting."
                )

    def _bind_hardware_workers(self) -> None:
        """Bind all hardware workers before any worker prepares."""
        hardware_workers = [
            worker for worker in self._workers if isinstance(worker, HardwareWorker)
        ]

        if not hardware_workers:
            return

        hardware = self._hardware

        if hardware is None:
            names = ", ".join(repr(worker.name) for worker in hardware_workers)
            raise RuntimeError(f"Hardware collection required by worker(s): {names}.")

        for worker in hardware_workers:
            worker._bind_hardware(hardware)
