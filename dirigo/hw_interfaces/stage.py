from abc import abstractmethod
import time
from typing import Literal, ClassVar

from pydantic import ConfigDict, Field, field_validator

from dirigo import units
from dirigo.hw_interfaces.hw_interface import (
    Device,
    DeviceConfig,
    DeviceSettings,
)



# ---- Types & helpers ----

LinearAxisLabel = Literal["x", "y", "z"] # TODO, make global axes?
RotationAxisLabel = Literal["theta", "phi"]


# ---- Config models ----

class _StageAxisConfig(DeviceConfig):
    axis: str = Field(
        description = "Logical axis label in the system, e.g. 'x', 'y', 'z', 'theta'."
    )
    limits: units.PositionRange | units.AngleRange | None = Field(
        default     = None,
        description = "Optional override for travel limits if hardware cannot report them.",
    )

    @field_validator("axis")
    @classmethod
    def _strip_axis(cls, v: str) -> str:
        v2 = v.strip()
        if not v2:
            raise ValueError("axis must be a non-empty string")
        return v2


class LinearStageAxisConfig(_StageAxisConfig):
    axis: LinearAxisLabel
    limits: units.PositionRange | None = None


class RotationStageAxisConfig(_StageAxisConfig):
    axis: RotationAxisLabel
    limits: units.AngleRange | None = None


# ---- Settings models ----

class _StageAxisSettings(DeviceSettings): # TODO, not sure this 'base' setting is really useful
    """
    Persistable adjustable state for a generic stage axis.

    Convention enforced by Device._check_settings():
      StageAxisSettings.max_velocity <-> StageAxis.max_velocity property
      etc.
    """
    # Keep these Optional so they can be partially applied.
    position: units.UnitQuantity | None = None
    max_velocity: units.UnitQuantity | None = None
    acceleration: units.UnitQuantity | None = None


class LinearStageAxisSettings(_StageAxisSettings):
    position: units.Position | None = None
    max_velocity: units.Velocity | None = None
    acceleration: units.Acceleration | None = None


class RotationStageAxisSettings(_StageAxisSettings):
    position: units.Angle | None = None
    max_velocity: units.AngularVelocity | None = None
    acceleration: units.AngularAcceleration | None = None


# ---- Axis base class ----

class _StageAxis(Device):
    """
    Base class for a single motion axis (linear or rotary).

    Concrete subclasses typically correspond to a real hardware axis.
    """
    config_model: ClassVar[type[DeviceConfig]] = _StageAxisConfig
    settings_model: ClassVar[type[DeviceSettings]] = _StageAxisSettings

    SLEEP_INTERVAL: ClassVar[units.Time] = units.Time("1 ms")

    def __init__(self, cfg: _StageAxisConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self._axis: str = cfg.axis
        
        self.cfg: _StageAxisConfig # for type hints

    @property
    def axis(self) -> str:
        return self._axis

    # --- Static / capability metadata ---
    @property
    def position_limits(self) -> units.PositionRange | units.AngleRange | None:
        """
        Return hard travel limits if known, else None.

        Default: config override if provided, otherwise use introspection.
        """
        if self.cfg.limits is not None:
            return self.cfg.limits
        return self._introspect_position_limits()
    
    def _introspect_position_limits(self) -> units.PositionRange | units.AngleRange | None:
        """Override if hardware can report limits."""
        return None
    
    # --- Live state ---
    @property
    @abstractmethod
    def moving(self) -> bool:
        """True while in motion."""
        ...

    @property
    @abstractmethod
    def homed(self) -> bool:
        """True if the axis has been homed (if supported)."""
        ...

    # --- Motion commands ---
    @abstractmethod
    def stop(self) -> None:
        """Immediate stop / halt."""
        ...

    @abstractmethod
    def home(self, blocking: bool = False) -> None:
        """Start a homing procedure (if supported)."""
        ...

    def wait_until_move_finished(self, timeout: units.Time | None = None) -> None:
        """
        Busy-wait until motion completes.

        You can later replace this with event/callback driven waiting.
        """
        if not self.is_connected:
            raise RuntimeError("Device must connect before waiting on motion.")

        # Optional timeout
        t0 = time.perf_counter()
        while self.moving:
            time.sleep(self.SLEEP_INTERVAL) 
            
            if timeout is not None:
                if time.perf_counter() - t0 >= timeout:
                    raise TimeoutError(f"{type(self).__name__}: motion did not finish within {timeout}")

    # --- Settings-backed properties (must have getter+setter) ---
    @property
    @abstractmethod
    def max_velocity(self) -> units.UnitQuantity:
        ...

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value: units.UnitQuantity) -> None:
        ...

    @property
    @abstractmethod
    def max_velocity_range(self) -> units.RangeWithUnits:
        ...

    @property
    @abstractmethod
    def acceleration(self) -> units.UnitQuantity:
        ...

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.UnitQuantity) -> None:
        ...

    @property
    @abstractmethod
    def acceleration_range(self) -> units.RangeWithUnits:
        ...


class LinearStageAxis(_StageAxis):
    """
    Single linear axis in position units.
    """
    config_model: ClassVar[type[DeviceConfig]] = LinearStageAxisConfig
    settings_model: ClassVar[type[DeviceSettings]] = LinearStageAxisSettings

    @property
    @abstractmethod
    def position(self) -> units.Position:
        ...

    # Should this be in _StageAxis?
    @abstractmethod
    def move_to(self, position: units.Position, blocking: bool = False) -> None:
        ...

    @abstractmethod
    def move_velocity(self, velocity: units.Velocity) -> None:
        ...

    @property
    @abstractmethod
    def max_velocity(self) -> units.Velocity:
        ...

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value: units.Velocity) -> None:
        ...

    @property
    @abstractmethod
    def max_velocity_range(self) -> units.VelocityRange:
        ...

    @property
    @abstractmethod
    def acceleration(self) -> units.Acceleration:
        ...

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.Acceleration) -> None:
        ...

    @property
    @abstractmethod
    def acceleration_range(self) -> units.AccelerationRange:
        ...


class RotationStageAxis(_StageAxis):
    """
    Single rotary axis in angle units.
    """
    config_model: ClassVar[type[DeviceConfig]] = RotationStageAxisConfig
    settings_model: ClassVar[type[DeviceSettings]] = RotationStageAxisSettings

    @property
    @abstractmethod
    def position(self) -> units.Angle:
        ...

    @abstractmethod
    def move_to(self, angle: units.Angle, blocking: bool = False) -> None:
        ...

    @abstractmethod
    def move_velocity(self, velocity: units.AngularVelocity) -> None:
        ...

    @property
    @abstractmethod
    def max_velocity(self) -> units.AngularVelocity:
        ...

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value: units.AngularVelocity) -> None:
        ...

    @property
    @abstractmethod
    def max_velocity_range(self) -> units.AngularVelocityRange:
        ...

    @property
    @abstractmethod
    def acceleration(self) -> units.AngularAcceleration:
        ...

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.AngularAcceleration) -> None:
        ...

    @property
    @abstractmethod
    def acceleration_range(self) -> units.AngularAccelerationRange:
        ...


# ---- Multi-axis composition ----

class XYStageConfig(DeviceConfig):
    x: LinearStageAxisConfig
    y: LinearStageAxisConfig


class XYZStageConfig(XYStageConfig):
    z: LinearStageAxisConfig


class XYStageSettings(DeviceSettings):
    x: LinearStageAxisSettings
    y: LinearStageAxisSettings


class XYZStageSettings(XYStageSettings):
    z: LinearStageAxisSettings


class XYStage(Device):
    """
    An XY stage composed of LinearStageAxis objects.
    """
    config_model: ClassVar[type[DeviceConfig]] = XYStageConfig
    settings_model: ClassVar[type[DeviceSettings]] = XYStageSettings

    def __init__(self, cfg: DeviceConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        self._x: LinearStageAxis
        self._y: LinearStageAxis

    # --- Axes getters ---
    @property
    def x(self) -> LinearStageAxis:
        return self._x
    
    @property
    def y(self) -> LinearStageAxis:
        return self._y

    # --- Vector convenience helpers ---
    @property
    def position(self) -> tuple[units.Position, units.Position]:
        """
        Return a vector of current positions (x,y).
        """
        return (self._x.position, self._y.position)

    def move_to(self, vect_pos: tuple[units.Position, units.Position]) -> None:
        """
        Move to vector position (x,y).
        """
        self._x.move_to(vect_pos[0])
        self._y.move_to(vect_pos[1])


class XYZStage(XYStage):
    """
    An XYZ stage composed of LinearStageAxis objects.
    """
    config_model: ClassVar[type[DeviceConfig]] = XYZStageConfig
    settings_model: ClassVar[type[DeviceSettings]] = XYZStageSettings

    def __init__(self, cfg: DeviceConfig, **kwargs):
        super().__init__(cfg, **kwargs)

        self._z: LinearStageAxis

    # --- Axes getters ---
    # (add to the existing XY getters)
    @property
    def z(self) -> LinearStageAxis:
        return self._z

    # --- Vector convenience helpers ---
    # (override existing XY helpers)
    @property
    def position(self) -> tuple[units.Position, units.Position, units.Position]:
        """
        Return a vector of current positions (x,y).
        """
        return (self._x.position, self._y.position, self._z.position)

    def move_to(self, vect_pos: tuple[units.Position, units.Position, units.Position]) -> None:
        """
        Move to vector position (x,y).
        """
        self._x.move_to(vect_pos[0])
        self._y.move_to(vect_pos[1])
        self._z.move_to(vect_pos[2])