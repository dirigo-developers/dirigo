from typing import Never

from .worker import Worker


class Source[OutputT = object](Worker[Never, OutputT]):
    """A worker that originates products and has no upstream."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
