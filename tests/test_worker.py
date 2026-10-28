import pytest

from dirigo.worker import Worker


class StubWorker(Worker):
    """Minimal concrete worker for testing inherited behavior."""

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        pass

    def _close(self) -> None:
        pass


def test_worker_cannot_be_instantiated_directly() -> None:
    with pytest.raises(expected_exception=TypeError):
        Worker() # ty: ignore[call-non-callable]


def test_upstream_defaults_to_none() -> None:
    worker = StubWorker()

    assert worker.upstream is None


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

