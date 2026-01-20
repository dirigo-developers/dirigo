from enum import StrEnum
from abc import abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dirigo.components import units
from dirigo.hw_interfaces.geometry import GlobalAxes
from dirigo.hw_interfaces.hw_interface import DeviceConfig, Device




class ImageOrientation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flip_fast: bool = Field(
        default     = False,
        description = "Reverse the fast (in-row) image axis."
    )
    flip_slow: bool = Field(
        default     = False,
        description = "Reverse the slow (row-to-row) image axis."
    )
    rotate: units.Angle = Field(
        default     = units.Angle("0 deg"), 
        description = "Rotation applied after flips (must be multiple of 90 deg)"
    )

    @field_validator("rotate")
    @classmethod
    def _validate_right_angle(cls, ang: units.Angle) -> units.Angle:
        # Allow small numerical noise if user constructed from floats
        q = float(ang / units.Angle("90 deg"))
        if abs(q - round(q)) > 1e-6:
            raise ValueError(
                f"Image rotation must be a multiple of 90°, got {ang.with_unit('deg')}°"
            )

        return ang


class CameraConfig(DeviceConfig):
    pixel_size: units.Position = Field(
        ...,
        description="Physical pixel size at sensor"
    )
    orientation: ImageOrientation = Field(
        default_factory=ImageOrientation
    )


class TriggerModes(StrEnum):
    FREE_RUN            = "free_run"
    EXTERNAL_TRIGGER    = "external_trigger"


class Camera(Device):
    config_model: ClassVar[type[DeviceConfig]] = CameraConfig

    def __init__(self, cfg: CameraConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg: CameraConfig

    @property
    def pixel_size(self) -> units.Position:
        """Physical pixel size at sensor."""
        return self.cfg.pixel_size
    
    # ---- Sensor geometry ----
    @property
    @abstractmethod
    def sensor_width_px(self) -> int: ...

    @property
    @abstractmethod
    def sensor_height_px(self) -> int: ...

    # ---- Controls ----
    @property
    @abstractmethod
    def integration_time(self) -> units.Time:
        pass
    
    @integration_time.setter
    @abstractmethod
    def integration_time(self, new_value: units.Time):
        pass

    @property
    @abstractmethod
    def gain(self):
        pass
    
    @gain.setter
    @abstractmethod
    def gain(self, new_value):
        pass

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        pass
    
    @bit_depth.setter
    @abstractmethod
    def bit_depth(self, new_value: int):
        pass

    @property
    @abstractmethod
    def data_range(self) -> units.IntRange:
        """
        Returns the range of values returned by the camera. 
        
        The returned data range may exceed the bit depth, which can be useful
        for in-place averaging.
        """
        pass

    # IO
    @property
    @abstractmethod
    def trigger_mode(self) -> TriggerModes:
        pass
    
    @trigger_mode.setter
    @abstractmethod
    def trigger_mode(self, mode: TriggerModes):
        pass

    @abstractmethod
    def load_profile(self):
        pass



class LineCameraConfig(CameraConfig):
    axis: GlobalAxes = Field(
        ...,
        description="Global axis aligned with sensor line"
    )


class LineCamera(Camera):
    config_model: ClassVar[type[DeviceConfig]] = LineCameraConfig

    def __init__(self, cfg: LineCameraConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg: LineCameraConfig

    @property
    def axis(self):
        return self.cfg.axis
    