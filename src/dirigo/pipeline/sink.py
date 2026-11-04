from typing import Any, Never

from .worker import Worker


class Sink[InputT = object](Worker[InputT, Never]):
    """A worker that consumes upstream products without publishing output."""

    def __init__(
        self,
        *,
        upstream: Worker[Any, InputT],
        name: str | None = None,
    ) -> None:
        if upstream is None:
            raise TypeError("Sink requires an upstream worker.")

        super().__init__(upstream=upstream, name=name)
