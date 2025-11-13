from typing import TypeVar, Generic
from dirigo.core.device import BaseDevice, BaseDeviceConfig


TScannerConfig = TypeVar("TScannerConfig", bound='ScannerDeviceConfig')


class ScannerDeviceConfig(BaseDeviceConfig):
    pass


class ScannerDevice(BaseDevice[TScannerConfig], Generic[TScannerConfig]):
    pass