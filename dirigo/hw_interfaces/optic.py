
from dirigo.core.device import BaseDevice, BaseDeviceConfig


class OpticsDeviceConfig(BaseDeviceConfig):
    """Common fields for any optics device could live here later."""
    pass


class OpticsDevice(BaseDevice):
    config: OpticsDeviceConfig
