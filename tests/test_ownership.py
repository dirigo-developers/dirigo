from threading import Condition

import pytest

from dirigo._ownership import _Ownership


def test_ownership_rejects_negative_retain_count() -> None:
    payload = bytearray(4)
    recycled: list[bytearray] = []
    ownership = _Ownership[bytearray](
        payload=payload,
        condition=Condition(),
        recycle=recycled.append,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        ownership.retain(-1)

    # Rejected retention must not change the original ownership count.
    ownership.release()

    assert len(recycled) == 1
    assert recycled[0] is payload


def test_ownership_rejects_release_after_recycling() -> None:
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
