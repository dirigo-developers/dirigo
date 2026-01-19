from abc import abstractmethod
from typing import ClassVar
from dataclasses import dataclass

from pydantic import Field

from dirigo.hw_interfaces.hw_interface import DeviceConfig, Device



@dataclass(frozen=True, slots=True)
class IlluminatorCapabilities:
    adjustable_intensity: bool = False


class IlluminatorConfig(DeviceConfig):
    pass


class Illuminator(Device):
    """
    Dirigo illuminator interface
    """
    config_model: ClassVar[type[DeviceConfig]] = IlluminatorConfig

    def __init__(self, cfg: IlluminatorConfig, **kwargs):
        super().__init__(cfg, **kwargs)

    @property
    def capabilities(self) -> IlluminatorCapabilities:
        return IlluminatorCapabilities()

    @property
    @abstractmethod
    def enabled(self) -> bool:
        ...

    @enabled.setter
    @abstractmethod
    def enabled(self, value: bool) -> None:
        ...


class DimmableIlluminator(Illuminator):
    @property
    @abstractmethod
    def intensity(self) -> float: 
        """Normalized intensity in [0, 1]."""
        ...

    @intensity.setter
    @abstractmethod
    def intensity(self, new_value: float) -> None: ...

    @property
    def capabilities(self) -> IlluminatorCapabilities:
        return IlluminatorCapabilities(adjustable_intensity=True)