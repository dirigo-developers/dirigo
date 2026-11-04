"""Test fixed-capacity buffer pools and producer leases.

Verify preallocation, exclusive acquisition, buffer reuse, timeout validation,
and lease invalidation after release. Acquisition contexts must return their
buffers on both normal exit and exceptions.

Pool cancellation must wake blocked acquisitions and permanently reject new
ones while preserving existing leases. Publication and delivery ownership are
tested separately.
"""

from threading import Condition, Event, Thread

import pytest

from dirigo.pipeline import OperationCancelled
from dirigo.pipeline.pool import Pool


def _observe_wait(
    condition: Condition,
    monkeypatch: pytest.MonkeyPatch,
) -> Event:
    """Signal when an operation reaches the condition's wait method."""
    waiting = Event()
    original_wait = condition.wait

    def observed_wait(timeout: float | None = None) -> bool:
        waiting.set()
        return original_wait(timeout)

    monkeypatch.setattr(condition, "wait", observed_wait)
    return waiting


def test_pool_preallocates_and_acquires_distinct_buffers() -> None:
    """Construction allocates every buffer, and simultaneous leases are distinct.

    Record each factory call to verify that construction fills the pool.
    Borrowing every buffer must return exactly those objects without further
    allocation. Another acquisition must time out while all buffers are held.
    """
    created: list[bytearray] = []

    def factory() -> bytearray:
        buffer = bytearray(4)
        created.append(buffer)
        return buffer

    pool = Pool[bytearray](capacity=2, factory=factory)

    assert len(created) == 2

    with (
        pool.acquire(timeout=0.0) as first,
        pool.acquire(timeout=0.0) as second,
    ):
        assert first.payload is not second.payload
        assert {id(first.payload), id(second.payload)} == {
            id(buffer) for buffer in created
        }

        with pytest.raises(TimeoutError), pool.acquire(timeout=0.0):
            pass

    assert len(created) == 2


def test_acquisition_context_returns_buffer() -> None:
    """Leaving an acquisition context makes the same buffer available again."""
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with pool.acquire(timeout=0.0) as first:
        original = first.payload

    with pool.acquire(timeout=0.0) as second:
        assert second.payload is original


def test_acquisition_context_returns_buffer_after_exception() -> None:
    """An exception propagates while context cleanup returns the borrowed buffer.

    A pool with one buffer would remain exhausted if exceptional exit failed
    to release its lease. Successful reacquisition proves that cleanup ran.
    """
    buffer = bytearray(4)
    pool = Pool[bytearray](capacity=1, factory=lambda: buffer)
    failure = RuntimeError("producer failed")

    with (
        pytest.raises(RuntimeError, match="producer failed") as caught,
        pool.acquire(timeout=0.0),
    ):
        raise failure

    assert caught.value is failure

    with pool.acquire(timeout=0.0) as lease:
        assert lease.payload is buffer


def test_lease_release_is_idempotent_and_invalidates_handle() -> None:
    """Release invalidates a lease and returns its buffer exactly once.

    Reacquire the released buffer, then release the old handle again. The stale
    handle must not return a buffer that now belongs to the new lease.

    Subsequent context exits must also leave exactly one available buffer,
    rather than introducing duplicate entries into the pool.
    """
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

            with pytest.raises(TimeoutError), pool.acquire(timeout=0.0):
                pass

    with pool.acquire(timeout=0.0) as third:
        assert third.payload is original

        with pytest.raises(TimeoutError), pool.acquire(timeout=0.0):
            pass


@pytest.mark.parametrize("capacity", [0, -2])
def test_pool_rejects_nonpositive_capacity(capacity: int) -> None:
    """A pool must have capacity for at least one buffer."""
    with pytest.raises(ValueError, match="capacity"):
        Pool[bytearray](capacity=capacity, factory=lambda: bytearray(4))


@pytest.mark.parametrize("timeout", [-3.14159265, -1.0, -0.1])
def test_pool_rejects_negative_timeout(timeout: float) -> None:
    """Entering an acquisition context rejects a negative timeout."""
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))

    with (
        pytest.raises(ValueError, match="timeout must be nonnegative"),
        pool.acquire(timeout=timeout),
    ):
        pass


@pytest.mark.parametrize("abort_pool", [False, True])
def test_blocked_acquisition_wakes(
    monkeypatch: pytest.MonkeyPatch,
    abort_pool: bool,
) -> None:
    """Returning a buffer or aborting the pool wakes a waiting acquisition.

    Hold the pool's only buffer and observe a second thread waiting to acquire.
    Releasing the held lease must let that thread borrow the same buffer.
    Aborting instead must wake it with OperationCancelled and no acquired buffer.

    The thread must finish before its acquisition timeout expires, proving that
    release or cancellation actively wakes the blocked operation.
    """
    pool = Pool[bytearray](capacity=1, factory=lambda: bytearray(4))
    waiting = _observe_wait(pool._condition, monkeypatch)
    acquired: list[bytearray] = []
    errors: list[BaseException] = []

    def acquire() -> None:
        try:
            with pool.acquire(timeout=5.0) as lease:
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

            thread.join(timeout=2.0)

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
            thread.join(timeout=6.0)


def test_pool_abort_preserves_borrowed_buffer() -> None:
    """Aborting rejects new acquisitions without invalidating an active lease.

    The current borrower must retain access to its buffer until release.
    Returning that buffer must not permit acquisition from the aborted pool.
    """
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


def test_pool_abort_is_permanent_and_idempotent() -> None:
    """Repeated aborts preserve cancellation, including after a buffer is returned.

    An existing lease remains usable across repeated abort calls. New
    acquisitions must fail both while that lease is held and after its context
    returns the buffer, proving that returning storage cannot reopen the pool.
    """
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

    with pytest.raises(OperationCancelled), pool.acquire(timeout=0.0):
        pass
