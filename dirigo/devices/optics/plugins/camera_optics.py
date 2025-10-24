from pydantic import Field

from dirigo.devices.optics.interface import OpticsDevice, OpticsDeviceConfig



class CameraOpticsConfig(OpticsDeviceConfig):
    magnification: float = Field(
        ..., description="Object → sensor lateral magnification."
    )


class CameraOpticsDevice(OpticsDevice[CameraOpticsConfig]):
    ConfigModel = CameraOpticsConfig 

    def __init__(self, config: CameraOpticsConfig) -> None:
        super().__init__(config)
        self._magnification = config.magnification

    @property
    def magnification(self) -> float:
        """Returns the object → sensor lateral magnification."""
        return self._magnification