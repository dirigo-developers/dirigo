import gc
import weakref
from typing import Any

import pytest

from dirigo.pipeline import Pipeline, Worker


class StubWorker(Worker[Any, Any]):
    def __init__(
        self,
        *,
        upstream: Worker[Any, Any] | None = None,
        fail_prepare: bool = False,
    ) -> None:
        super().__init__(upstream=upstream)
        self.fail_prepare = fail_prepare
        self.prepare_count = 0
        self.close_count = 0

    def _prepare(self) -> None:
        self.prepare_count += 1
        if self.fail_prepare:
            raise RuntimeError("Preparation failed.")

    def _work(self) -> None:
        pass

    def _close(self) -> None:
        self.close_count += 1


def test_constructor_rejects_worker_owned_by_another_pipeline() -> None:
    worker = StubWorker()
    first = Pipeline(worker)

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)

    assert worker.prepare_count == 0

    first.start()
    first.wait(timeout=1.0)
    assert worker.prepare_count == worker.close_count == 1


def test_add_rejects_owned_worker_without_changing_destination() -> None:
    owned = StubWorker()
    first = Pipeline(owned)
    independent = StubWorker()
    second = Pipeline(independent)

    with pytest.raises(RuntimeError, match="another pipeline"):
        second.add(owned)

    assert first.workers == (owned,)
    assert second.workers == (independent,)


def test_shared_upstream_is_rejected_without_claiming_new_sink() -> None:
    source = StubWorker()
    first = Pipeline(source)
    sink = StubWorker(upstream=source)
    second = Pipeline()

    with pytest.raises(RuntimeError, match="another pipeline"):
        second.add(sink)

    assert second.workers == ()

    first.add(sink)
    assert first.workers == (source, sink)


def test_duplicate_additions_and_shared_branches_are_allowed() -> None:
    source = StubWorker()
    left = StubWorker(upstream=source)
    right = StubWorker(upstream=source)
    pipeline = Pipeline(left, right, left)

    pipeline.add(source)
    pipeline.add(right)

    assert pipeline.workers == (source, left, right)


def test_failed_constructor_releases_earlier_claims() -> None:
    owned = StubWorker()
    first = Pipeline(owned)
    fresh = StubWorker()

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(fresh, owned)

    assert Pipeline(fresh).workers == (fresh,)

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(owned)

    assert first.workers == (owned,)


def test_completion_does_not_release_ownership() -> None:
    worker = StubWorker()
    pipeline = Pipeline(worker)
    pipeline.start()
    pipeline.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)

    assert worker.prepare_count == worker.close_count == 1


def test_preparation_failure_does_not_release_ownership() -> None:
    worker = StubWorker(fail_prepare=True)
    pipeline = Pipeline(worker)

    with pytest.raises(RuntimeError, match="Preparation failed"):
        pipeline.start()

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)

    assert worker.prepare_count == worker.close_count == 1


def test_discarding_pipeline_does_not_make_worker_reusable() -> None:
    worker = StubWorker()
    pipeline = Pipeline(worker)
    reference = weakref.ref(pipeline)
    del pipeline
    gc.collect()

    assert reference() is None

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)