from typing import Never

from .hardware import Hardware
from .worker import Worker


class Source[OutputT = object](Worker[Never, OutputT]):
    """A worker that originates products and has no upstream."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)


class HardwareSource[OutputT = object](Source[OutputT]):
    """A source that borrows hardware supplied by its pipeline."""

    def __init__(self, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self._bound_hardware: Hardware | None = None

    @property
    def _hardware(self) -> Hardware:
        """The borrowed collection, available from preparation onward."""
        hardware = self._bound_hardware

        if hardware is None:
            raise RuntimeError(
                f"Hardware has not been bound to source {self.name!r}."
            )

        return hardware

    def _bind_hardware(self, hardware: Hardware) -> None:
        """Bind once before preparation; ownership stays with the caller."""
        if self._bound_hardware is not None:
            raise RuntimeError(
                f"Hardware is already bound to source {self.name!r}."
            )

        self._bound_hardware = hardware