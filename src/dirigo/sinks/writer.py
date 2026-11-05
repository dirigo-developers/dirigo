from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO

from dirigo.pipeline.sink import Sink
from dirigo.pipeline.worker import Worker


class Writer[InputT = object](Sink[InputT]):
    """A sink that saves incoming data to a configured destination.

    Construction does not create filesystem entries. Concrete writers own
    their output resources and implement preparation, execution, and cleanup.

    The destination may represent a file or directory, depending on the
    concrete writer.
    """

    def __init__(
        self,
        *,
        upstream: Worker[Any, InputT],
        path: str | PathLike[str],
        name: str | None = None,
    ) -> None:
        # Validate the path before creating an upstream subscription.
        self._path = Path(path)

        super().__init__(upstream=upstream, name=name)

    @property
    def path(self) -> Path:
        """The configured output destination."""
        return self._path

    def _open_file(self) -> BinaryIO:
        """Create a new binary output file and any missing parent directories.

        Existing destinations raise FileExistsError. Exclusive creation
        prevents competing writers from opening the same output successfully.

        The caller owns the returned handle and must close it, including
        after preparation or execution fails.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return self._path.open("xb") # x=exclusive
