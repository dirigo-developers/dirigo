from typing import Any

from dirigo.hardware.hardware import Hardware

from .worker import Worker


class HardwareWorker[InputT = object, OutputT = object](Worker[InputT, OutputT]):
    """A worker that borrows hardware supplied by its pipeline.

    May originate products or consume upstream products. The pipeline binds
    hardware before any worker prepares.

    Ownership of the collection and its devices remains with the caller.
    """

    def __init__(
        self,
        *,
        upstream: Worker[Any, InputT] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(upstream=upstream, name=name)
        self._bound_hardware: Hardware | None = None

    @property
    def _hardware(self) -> Hardware:
        """The borrowed collection, available from preparation onward."""
        hardware = self._bound_hardware

        if hardware is None:
            raise RuntimeError(f"Hardware has not been bound to worker {self.name!r}.")

        return hardware

    def _bind_hardware(self, hardware: Hardware) -> None:
        """Bind once before preparation; ownership stays with the caller."""
        if self._bound_hardware is not None:
            raise RuntimeError(f"Hardware is already bound to worker {self.name!r}.")

        self._bound_hardware = hardware
