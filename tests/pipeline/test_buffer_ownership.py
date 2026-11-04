"""Test the internal ownership record used to recycle pooled buffers.

Each record starts with one producer claim. Retaining adds claims, and releasing
the final claim must recycle the original payload exactly once.

Invalid operations must preserve claim accounting. An exhausted record cannot
be revived, even if its payload storage is later reused by the pool.

Lease behavior, publication, and delivery integration are tested separately.
"""

from threading import Condition

import pytest

from dirigo.pipeline._ownership import _Ownership


def test_ownership_recycles_only_after_final_release() -> None:
    """Retained claims prevent recycling until every claim has been released.

    The record starts with one producer claim. Retaining two additional claims
    therefore requires three releases before recycling may occur.

    The callback must receive the original payload exactly once.
    """
    payload = bytearray(4)
    recycled: list[bytearray] = []
    ownership = _Ownership[bytearray](
        payload=payload,
        condition=Condition(),
        recycle=recycled.append,
    )

    ownership.retain(2)

    assert recycled == []

    for _ in range(2):
        ownership.release()
        assert recycled == []

    ownership.release()

    assert len(recycled) == 1
    assert recycled[0] is payload


def test_ownership_rejects_negative_retain_count() -> None:
    """Rejecting a negative retain count leaves the original claim intact.

    The invalid operation must neither recycle the payload nor alter the number
    of releases required. Releasing the original producer claim must still
    recycle the payload exactly once.
    """
    payload = bytearray(4)
    recycled: list[bytearray] = []
    ownership = _Ownership[bytearray](
        payload=payload,
        condition=Condition(),
        recycle=recycled.append,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        ownership.retain(-1)

    assert recycled == []

    ownership.release()

    assert len(recycled) == 1
    assert recycled[0] is payload


def test_ownership_rejects_release_after_recycling() -> None:
    """An extra release raises without recycling the same payload again.

    Once the original claim is released, the record has no remaining owner.
    Another release indicates incorrect accounting and must not invoke the
    recycling callback a second time.
    """
    payload = bytearray(4)
    recycled: list[bytearray] = []
    ownership = _Ownership[bytearray](
        payload=payload,
        condition=Condition(),
        recycle=recycled.append,
    )

    ownership.release()

    with pytest.raises(RuntimeError, match="already been released"):
        ownership.release()

    assert len(recycled) == 1
    assert recycled[0] is payload


def test_ownership_rejects_retain_after_recycling() -> None:
    """An exhausted ownership record cannot acquire new claims.

    Recycling ends this record's lifetime. Retaining afterward must raise,
    preventing a stale record from claiming storage that the pool may already
    have assigned to another acquisition.
    """
    payload = bytearray(4)
    recycled: list[bytearray] = []
    ownership = _Ownership[bytearray](
        payload=payload,
        condition=Condition(),
        recycle=recycled.append,
    )

    ownership.release()

    with pytest.raises(RuntimeError, match="ownership has been released"):
        ownership.retain(1)

    assert len(recycled) == 1
    assert recycled[0] is payload
