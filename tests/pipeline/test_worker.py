"""Test worker construction and basic execution rules.

Verify upstream validation, naming, abstract construction, and single-use
execution. Starting and waiting are tested directly through the internal hooks
used by Pipeline.

Cancellation, pooled output, metadata handling, and graph execution are tested
in separate modules.
"""

import pytest

from dirigo.pipeline import Worker

from .stubs import BlockingWorker, StubWorker


def test_worker_cannot_be_instantiated_directly() -> None:
    """Worker remains abstract; subclasses must implement its lifecycle hooks."""
    with pytest.raises(TypeError):
        Worker()  # ty: ignore[call-non-callable]


def test_upstream_defaults_to_none() -> None:
    """A worker constructed without an upstream has no input dependency."""
    worker = StubWorker()

    assert worker.upstream is None


def test_worker_rejects_invalid_upstream() -> None:
    """An upstream dependency must be a Worker instance or None."""
    with pytest.raises(TypeError, match="upstream"):
        StubWorker(upstream=object())  # ty: ignore[invalid-argument-type]


def test_upstream_preserves_the_supplied_worker() -> None:
    """Construction retains the exact worker supplied as the upstream."""
    upstream = StubWorker()

    worker = StubWorker(upstream=upstream)

    assert worker.upstream is upstream


def test_name_defaults_to_concrete_class_name() -> None:
    """Omitting a name uses the concrete subclass name."""
    worker = StubWorker()

    assert worker.name == "StubWorker"


def test_explicit_none_name_uses_concrete_class_name() -> None:
    """Passing name=None has the same naming behavior as omitting the argument."""
    worker = StubWorker(name=None)

    assert worker.name == "StubWorker"


@pytest.mark.parametrize("name", ["acquisition", ""])
def test_explicit_name_is_preserved(name: str) -> None:
    """An explicit string is preserved, including an empty string."""
    worker = StubWorker(name=name)

    assert worker.name == name


def test_worker_receive_requires_upstream() -> None:
    """Receiving without an upstream fails immediately when _receive is called."""
    worker = StubWorker()

    with pytest.raises(RuntimeError, match="no upstream"):
        worker._receive(timeout=0.0)


@pytest.mark.parametrize("finish_first", [False, True])
def test_worker_cannot_start_twice(finish_first: bool) -> None:
    """A worker cannot restart while running or after normal completion.

    Keep the worker blocked to test a second start during execution. In the
    other case, release it and wait for completion before attempting restart.

    Always release and join the original thread, even if an assertion fails.
    """
    worker = BlockingWorker()
    worker._start()

    try:
        if finish_first:
            worker.release.set()
            worker._wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="already started"):
            worker._start()
    finally:
        worker.release.set()
        worker._wait(timeout=2.0)


def test_worker_cannot_wait_before_start() -> None:
    """Waiting before a worker has an execution thread raises RuntimeError."""
    worker = StubWorker()

    with pytest.raises(RuntimeError, match="has not started"):
        worker._wait()


def test_worker_wait_raises_on_timeout() -> None:
    """Waiting times out when execution has not finished.

    The worker cannot complete until its release event is set, so a zero-timeout
    wait must fail. Cleanup then releases it and waits for the thread to finish.
    """
    worker = BlockingWorker()
    worker._start()

    try:
        with pytest.raises(TimeoutError, match="did not finish"):
            worker._wait(timeout=0.0)
    finally:
        worker.release.set()
        worker._wait(timeout=2.0)
