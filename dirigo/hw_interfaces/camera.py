from enum import StrEnum
from abc import abstractmethod
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dirigo.components import units
from dirigo.hw_interfaces.geometry import GlobalAxes
from dirigo.hw_interfaces.hw_interface import DeviceConfig, DeviceSettings, Device



# ---- Camera Enumerations ----
class TriggerMode(StrEnum):
    FREE_RUN            = "free_run"
    EXTERNAL_TRIGGER    = "external_trigger"
    # TODO, add trigger+integration time, but this would invalidate integration_time property


class PixelFormat(StrEnum):
    MONO8   = "mono8"
    MONO10  = "mono10"
    MONO12  = "mono12"
    MONO16  = "mono16"
    RGB24   = "rgb24"
    # TODO: should we also defined packed formats? More relevant for ImageTransport


_BIT_DEPTH_MAPPING = {
    PixelFormat.MONO8: 8,
    PixelFormat.MONO10: 10,
    PixelFormat.MONO12: 12,
    PixelFormat.MONO16: 16,
    PixelFormat.RGB24: 8
}


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


class CameraSettings(DeviceSettings):
    """
    Adjustable camera settings
    """
    integration_time: units.Time | None = None
    gain: float | None = None
    trigger_mode: TriggerMode | None = None
    pixel_format: PixelFormat | None = None


class Camera(Device):
    config_model: ClassVar[type[DeviceConfig]] = CameraConfig
    settings_model: ClassVar[type[DeviceSettings]] = CameraSettings

    def __init__(self, cfg: CameraConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg: CameraConfig

    @property
    def pixel_size(self) -> units.Position:
        """
        Effective pixel sampling pitch. 

        Override if introspectable and/or pixel size changes with binning.
        """
        return self.cfg.pixel_size
    
    # ---- Sensor characteristics ----
    @property
    @abstractmethod
    def image_width_px(self) -> int: 
        """Width of delivered frames in pixels."""
        ...

    @property
    @abstractmethod
    def image_height_px(self) -> int:
        """Height of delivered frames in pixels."""
        ...

    # ---- Controls ----
    @property
    @abstractmethod
    def pixel_format(self) -> PixelFormat:
        ...

    @pixel_format.setter
    @abstractmethod
    def pixel_format(self, f: PixelFormat) -> None:
        """Should raise SettingNotSettableError if this is not settable."""
        ...

    @property
    @abstractmethod
    def supported_pixel_formats(self) -> tuple[PixelFormat, ...]:
        ...

    @property
    @abstractmethod
    def integration_time(self) -> units.Time:
        ...
    
    @integration_time.setter
    @abstractmethod
    def integration_time(self, t: units.Time):
        ...

    @property
    @abstractmethod
    def integration_time_range(self) -> units.TimeRange:
        ...

    @property
    @abstractmethod
    def gain(self) -> float:
        ...
    
    @gain.setter
    @abstractmethod
    def gain(self, g: float):
        ...

    @property
    @abstractmethod
    def supported_gains(self) -> units.FloatRange | tuple[float, ...]:
        """
        Returns either a FloatRange for quasi-continuously adjustable gain or
        a tuple of floats for discrete supported gain values.
        """

    @property
    @abstractmethod
    def trigger_mode(self) -> TriggerMode:
        ...
    
    @trigger_mode.setter
    @abstractmethod
    def trigger_mode(self, mode: TriggerMode):
        ...

    @property
    @abstractmethod
    def supported_trigger_modes(self) -> tuple[TriggerMode, ...]:
        ...

    # ---- Helpers ----
    @property
    def bit_depth(self) -> int:
        """The binary bits per pixel, per channel"""
        return _BIT_DEPTH_MAPPING[self.pixel_format]
    
    @property
    def is_color(self) -> bool:
        """Whether or not the camera outputs color pixels"""
        if self.pixel_format in (PixelFormat.RGB24,):
            return True
        else:
            return False


class ScanDirection(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"


class LineCameraConfig(CameraConfig):
    axis: GlobalAxes = Field(
        ...,
        description="Global axis aligned with sensor line"
    )
    # TODO, modify orientation to remove height and rotation


class LineCameraSettings(CameraSettings):
    scan_direction: ScanDirection | None = None


class LineCamera(Camera):
    """Adds line axis semantics and hardcodes image height to 1 pixel."""
    config_model: ClassVar[type[DeviceConfig]] = LineCameraConfig
    settings_model: ClassVar[type[DeviceSettings]] = LineCameraSettings

    def __init__(self, cfg: LineCameraConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg: LineCameraConfig

    @property
    def axis(self) -> GlobalAxes:
        return self.cfg.axis
    
    @property
    @abstractmethod
    def scan_direction(self) -> ScanDirection:
        """Direction convention used by the sensor/readout for line acquisition."""
        ...

    @scan_direction.setter
    @abstractmethod
    def scan_direction(self, d: ScanDirection) -> None:
        """Raise SettingNotSettableError if fixed."""
        ...
    
    # ---- Sensor geometry ----
    @property
    def image_height_px(self) -> int:
        return 1 # Line sensor
    