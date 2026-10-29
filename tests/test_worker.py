import pytest

from dirigo.worker import Worker

from .stubs import BlockingWorker, StubWorker


def test_worker_cannot_be_instantiated_directly() -> None:
    with pytest.raises(expected_exception=TypeError):
        Worker() # ty: ignore[call-non-callable]


def test_upstream_defaults_to_none() -> None:
    worker = StubWorker()

    assert worker.upstream is None


def test_worker_rejects_invalid_upstream() -> None:
    with pytest.raises(expected_exception=TypeError, match="upstream"):
        StubWorker(upstream=object())  # ty: ignore[invalid-argument-type]


def test_upstream_preserves_the_supplied_worker() -> None:
    upstream = StubWorker()
    worker = StubWorker(upstream=upstream)

    assert worker.upstream is upstream


def test_name_defaults_to_concrete_class_name() -> None:
    worker = StubWorker()

    assert worker.name == "StubWorker"


def test_explicit_none_name_uses_concrete_class_name() -> None:
    worker = StubWorker(name=None)

    assert worker.name == "StubWorker"


@pytest.mark.parametrize("name", ["acquisition", ""])
def test_explicit_name_is_preserved(name: str) -> None:
    worker = StubWorker(name=name)

    assert worker.name == name


def test_worker_cannot_start_twice() -> None:
    worker = StubWorker()
    worker._start()

    try:
        with pytest.raises(expected_exception=RuntimeError, match="already started"):
            worker._start()
    finally:
        worker._wait(timeout=2.0)


def test_worker_cannot_wait_before_start() -> None:
    worker = StubWorker()

    with pytest.raises(expected_exception=RuntimeError, match="has not started"):
        worker._wait()


def test_worker_wait_raises_on_timeout() -> None:
    worker = BlockingWorker()
    worker._start()

    try:
        with pytest.raises(expected_exception=TimeoutError, match="did not finish"):
            worker._wait(timeout=0.0)
    finally:
        worker.release.set()
        worker._wait(timeout=2.0)

