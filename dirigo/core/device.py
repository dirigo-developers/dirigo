from abc import ABC
from typing import TypeVar, Generic

from pydantic import BaseModel, Field


TDeviceConfig = TypeVar("TDeviceConfig", bound='BaseDeviceConfig')


class BaseDeviceConfig(BaseModel):
    """Base class for device-specific configuration schemas."""
    pass


class BaseDevice(Generic[TDeviceConfig], ABC):
    """All devices have a pydantic config of type C."""
    ConfigModel: type[TDeviceConfig]            # must be set by subclasses
    config: TDeviceConfig

    def __init__(self, config: TDeviceConfig) -> None:
        self.config = config

    @classmethod
    def validate_config(cls, data: dict | BaseModel) -> TDeviceConfig:
        """Centralized pre-init validation used by the registry/factory."""
        return cls.ConfigModel.model_validate(data)


class DeviceContainer:
    pass