from threading import Event, Thread

import pytest

from dirigo.pipeline import OperationCancelled
from dirigo.pipeline.pool import Pool


def test_pool_preallocates_and_acquires_distinct_buffers() -> None:
    created: list[bytearray] = []

    def factory() -> bytearray:
        buffer = bytearray(4)
        created.append(buffer)
        return buffer

    pool = Pool[bytearray](capacity=2, factory=factory)

    assert len(created) == 2

    with pool.acquire(timeout=0.0) as first: # ruff: ignore[multiple-with-statements]
        with pool.acquire(timeout=0.0) as second:
            assert first.payload is not second.payload
            assert {id(first.payload), id(second.payload)} == {
                id(buffer) for buffer in created
            }

            with pytest.raises(TimeoutError), pool.acquire(timeout=0.0):
                pass

    assert len(created) == 2


def test_acquisition_context_returns_buffer() -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with pool.acquire(timeout=0.0) as first:
        original = first.payload

    with pool.acquire(timeout=0.0) as second:
        assert second.payload is original


def test_lease_release_is_idempotent_and_invalidates_handle() -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with pool.acquire(timeout=0.0) as first:
        original = first.payload
        first.release()

        with pytest.raises(RuntimeError, match="released"):
            _ = first.payload

        with pool.acquire(timeout=0.0) as second:
            assert second.payload is original

            # The old handle must not release the new owner's claim.
            first.release()

            with pytest.raises(TimeoutError): # ruff: ignore[multiple-with-statements]
                with pool.acquire(timeout=0.0):
                    pass

    with pool.acquire(timeout=0.0): #  ruff: ignore[multiple-with-statements]
        with pytest.raises(TimeoutError):
            with pool.acquire(timeout=0.0):
                pass


@pytest.mark.parametrize("capacity", [0, -1])
def test_pool_rejects_nonpositive_capacity(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity"):
        Pool[bytearray](capacity=capacity, factory=lambda: bytearray(4))


def test_pool_abort_preserves_borrowed_buffer() -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        pool.abort()

        assert lease.payload is original

        with pytest.raises(OperationCancelled), pool.acquire(timeout=0.0):
            pass

    pool.abort()

    with pytest.raises(OperationCancelled), pool.acquire(timeout=0.0):
        pass


@pytest.mark.parametrize("abort_pool", [False, True])
def test_blocked_acquisition_wakes(
    monkeypatch: pytest.MonkeyPatch,
    abort_pool: bool,
) -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    waiting = Event()
    acquired: list[bytearray] = []
    errors: list[BaseException] = []

    original_wait = pool._condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(pool._condition, "wait", observed_wait)

    def acquire() -> None:
        try:
            with pool.acquire(timeout=2.0) as lease:
                acquired.append(lease.payload)
        except BaseException as exc:
            errors.append(exc)

    with pool.acquire(timeout=0.0) as held:
        original = held.payload
        thread = Thread(target=acquire)
        thread.start()

        try:
            assert waiting.wait(timeout=2.0)

            if abort_pool:
                pool.abort()
            else:
                held.release()

            thread.join(timeout=3.0)

            assert not thread.is_alive()

            if abort_pool:
                assert acquired == []
                assert len(errors) == 1
                assert isinstance(errors[0], OperationCancelled)
            else:
                assert errors == []
                assert len(acquired) == 1
                assert acquired[0] is original
        finally:
            pool.abort()
            held.release()
            thread.join(timeout=3.0)


@pytest.mark.parametrize("timeout", [-1.0, -0.1])
def test_pool_rejects_negative_timeout(timeout: float) -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with (
        pytest.raises(ValueError, match="timeout must be nonnegative"),
        pool.acquire(timeout=timeout),
    ):
        pass


def test_pool_abort_is_permanent_and_idempotent() -> None:
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with pool.acquire(timeout=0.0) as lease:
        original = lease.payload

        pool.abort()
        pool.abort()

        assert lease.payload is original

        with (
            pytest.raises(OperationCancelled, match="Pool was aborted"),
            pool.acquire(timeout=0.0),
        ):
            pass

    # Returning the borrowed buffer must not reopen the pool.
    with pytest.raises(OperationCancelled), pool.acquire(timeout=0.0):
        pass



