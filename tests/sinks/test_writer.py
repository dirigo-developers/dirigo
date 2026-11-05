"""Test writer destination configuration and exclusive file creation."""

from pathlib import Path

import pytest

from dirigo.sinks import Writer
from tests.pipeline.stubs import StubWorker


class StubWriter(Writer[int]):
    """Implement lifecycle hooks without opening an output automatically."""

    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        pass

    def _close(self) -> None:
        pass


@pytest.mark.parametrize("string_path", [False, True])
def test_configuration_preserves_destination_and_upstream(
    tmp_path: Path,
    string_path: bool,
) -> None:
    """Accept Path and string destinations without creating filesystem entries.

    Use a destination beneath a missing directory to verify that construction
    creates neither the output nor its parent.
    """
    destination = tmp_path / "recording" / "frames.bin"
    path = str(destination) if string_path else destination
    upstream = StubWorker[object, int]()

    writer = StubWriter(
        upstream=upstream,
        path=path,
        name="recording",
    )

    assert writer.path == destination
    assert writer.upstream is upstream
    assert writer.name == "recording"
    assert not destination.parent.exists()


def test_open_file_creates_binary_output_and_missing_parents(
    tmp_path: Path,
) -> None:
    """Opening a new output creates its parent directories and accepts bytes."""
    destination = tmp_path / "experiment" / "run" / "frames.bin"
    writer = StubWriter(upstream=StubWorker(), path=destination)

    with writer._open_file() as output:
        output.write(b"\x00\x80\xff")

    assert destination.read_bytes() == b"\x00\x80\xff"


def test_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    """An existing output raises FileExistsError and retains its contents."""
    destination = tmp_path / "frames.bin"
    destination.write_bytes(b"existing recording")
    writer = StubWriter(upstream=StubWorker(), path=destination)

    with pytest.raises(FileExistsError), writer._open_file():
        pytest.fail("An existing output must not be opened.")

    assert destination.read_bytes() == b"existing recording"


def test_two_writers_cannot_create_the_same_output(tmp_path: Path) -> None:
    """Exclusive creation protects an output claimed after writer construction.

    Configure both writers before the destination exists. Keep the first handle
    open while the second writer attempts creation, proving that construction
    does not reserve the path and opening enforces exclusivity.
    """
    destination = tmp_path / "frames.bin"
    first = StubWriter(upstream=StubWorker(), path=destination)
    second = StubWriter(upstream=StubWorker(), path=destination)

    with first._open_file() as output:
        output.write(b"first recording")

        with pytest.raises(FileExistsError), second._open_file():
            pytest.fail("A second writer must not open the same output.")

    assert destination.read_bytes() == b"first recording"
