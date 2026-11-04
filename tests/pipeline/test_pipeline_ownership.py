"""Test exclusive and permanent worker membership in pipelines.

Including a worker reserves it for one pipeline, including its upstream
dependencies. Repeated inclusion within that pipeline remains valid.

Failed additions must leave existing membership unchanged. Failed construction
must release newly acquired claims without disturbing another pipeline's claims.

Completion, preparation failure, and garbage collection do not make an accepted
worker reusable. Buffer ownership is tested separately.
"""

import gc
import weakref
from typing import Any

import pytest

from dirigo.pipeline import Pipeline, Worker

from .stubs import StubWorker


class RecordingWorker(StubWorker):
    """Count lifecycle calls and optionally fail during preparation."""

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

    def _close(self) -> None:
        self.close_count += 1


def _abort_and_join(pipeline: Pipeline) -> None:
    """Cancel and join launched workers without re-raising recorded failures."""
    if not pipeline._launched:
        return

    pipeline.abort()

    for worker in pipeline._launched:
        worker._wait(timeout=2.0)


def test_constructor_rejects_worker_owned_by_another_pipeline() -> None:
    """A second pipeline cannot claim a worker before the first has even started.

    Rejecting the second constructor must neither prepare the worker nor
    interfere with its original pipeline's ability to execute it normally.
    """
    worker = RecordingWorker()
    first = Pipeline(worker)

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)

    assert worker.prepare_count == 0

    try:
        first.start()
        first.wait(timeout=2.0)

        assert worker.prepare_count == worker.close_count == 1
    finally:
        _abort_and_join(first)


def test_add_rejects_owned_worker_without_changing_destination() -> None:
    """Adding another pipeline's worker leaves both pipelines unchanged."""
    owned = RecordingWorker()
    first = Pipeline(owned)
    independent = RecordingWorker()
    second = Pipeline(independent)

    with pytest.raises(RuntimeError, match="another pipeline"):
        second.add(owned)

    assert first.workers == (owned,)
    assert second.workers == (independent,)


def test_shared_upstream_is_rejected_without_claiming_new_sink() -> None:
    """An ownership conflict upstream must not strand a newly supplied sink.

    The sink itself is unclaimed, but its source already belongs to the first
    pipeline. Adding the sink to a second pipeline must fail without claiming it.

    Adding that same sink to the source's original pipeline must then succeed.
    """
    source = RecordingWorker()
    first = Pipeline(source)
    sink = RecordingWorker(upstream=source)
    second = Pipeline()

    with pytest.raises(RuntimeError, match="another pipeline"):
        second.add(sink)

    assert second.workers == ()

    first.add(sink)

    assert first.workers == (source, sink)


def test_duplicate_additions_and_shared_branches_are_allowed() -> None:
    """Repeated inclusion within one pipeline is valid and does not duplicate workers.

    Include two branches sharing a source, repeat a constructor argument, then
    explicitly add workers already present. Membership and ordering must remain
    unchanged.
    """
    source = RecordingWorker()
    left = RecordingWorker(upstream=source)
    right = RecordingWorker(upstream=source)

    pipeline = Pipeline(left, right, left)
    pipeline.add(source)
    pipeline.add(right)

    assert pipeline.workers == (source, left, right)


def test_failed_constructor_releases_earlier_claims() -> None:
    """An ownership conflict rolls back claims acquired earlier by the constructor.

    Accept a fresh worker before encountering one owned by another pipeline.
    Construction must release the fresh worker while preserving the existing
    owner's exclusive claim.
    """
    owned = RecordingWorker()
    first = Pipeline(owned)
    fresh = RecordingWorker()

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(fresh, owned)

    assert Pipeline(fresh).workers == (fresh,)

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(owned)

    assert first.workers == (owned,)


def test_failed_construction_releases_previously_accepted_worker() -> None:
    """An invalid constructor argument releases ownership of an earlier valid worker.

    The constructor accepts the worker before encountering a non-worker.
    Reusing that worker in a new pipeline proves that failed construction did
    not leave it permanently reserved.
    """
    worker = StubWorker()

    with pytest.raises(TypeError, match="worker must be a Worker"):
        Pipeline(worker, object())  # ty: ignore[invalid-argument-type]

    assert Pipeline(worker).workers == (worker,)


def test_completion_does_not_release_ownership() -> None:
    """Normal completion does not make a single-use worker available to another
    pipeline.
    """
    worker = RecordingWorker()
    pipeline = Pipeline(worker)

    try:
        pipeline.start()
        pipeline.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="another pipeline"):
            Pipeline(worker)

        assert worker.prepare_count == worker.close_count == 1
    finally:
        _abort_and_join(pipeline)


def test_preparation_failure_does_not_release_ownership() -> None:
    """Failed preparation leaves the worker reserved for its original pipeline.

    This differs from constructor rollback: the pipeline was constructed
    successfully and has attempted execution. Cleanup must run, but another
    pipeline must not be permitted to reuse the worker.
    """
    worker = RecordingWorker(fail_prepare=True)
    pipeline = Pipeline(worker)

    with pytest.raises(RuntimeError, match="Preparation failed"):
        pipeline.start()

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)

    assert worker.prepare_count == worker.close_count == 1


def test_discarding_pipeline_does_not_make_worker_reusable() -> None:
    """Collecting a pipeline does not release its reservation of a surviving worker.

    The weak reference observes the pipeline without keeping it alive. Delete
    the strong reference and explicitly collect garbage to establish that the
    pipeline is gone.

    Even though execution never started, its previously accepted worker must
    remain unavailable to another pipeline.
    """
    worker = RecordingWorker()
    pipeline = Pipeline(worker)
    reference = weakref.ref(pipeline)

    del pipeline
    gc.collect()

    assert reference() is None

    with pytest.raises(RuntimeError, match="another pipeline"):
        Pipeline(worker)
