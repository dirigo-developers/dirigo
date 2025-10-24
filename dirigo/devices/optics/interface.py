from typing import TypeVar, Generic
from dirigo.core.device import BaseDevice, BaseDeviceConfig


TOpticsConfig = TypeVar("TOpticsConfig", bound='OpticsDeviceConfig')


class OpticsDeviceConfig(BaseDeviceConfig):
    pass


class OpticsDevice(BaseDevice[TOpticsConfig], Generic[TOpticsConfig]):
    pass