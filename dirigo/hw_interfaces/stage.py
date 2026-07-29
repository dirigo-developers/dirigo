from dataclasses import dataclass
from functools import cached_property
from abc import abstractmethod
import time

from dirigo.components import units
from dirigo.hw_interfaces.hw_interface import HardwareInterface



@dataclass
class StageInfo:
    """
    Object describing permanent information (manufacturer, model, etc.).
    
    Does not describe characteristics that could potentially be customized by 
    the user (velocity, position limits, axis orientation, etc).

    Subclass this base class to include more information fields.
    """
    manufacturer: str
    model: str


class Stage(HardwareInterface):
    """
    Abstract interface for a single stage. Can be linear or rotational.
    """
    attr_name = "stage"
    VALID_AXES = {} # subclasses must overwrite with allowed axes labels e.g. 'x'
    SLEEP_INTERVAL = units.Time('1 ms')

    @staticmethod
    def _validate_limits_dict(limits_dict):
        if not isinstance(limits_dict, dict):
            raise ValueError(
                "limits must be a dictionary."
            )
        missing_keys = {'min', 'max'} - limits_dict.keys()
        if missing_keys:
            raise ValueError(
                f"limits must be a dictionary with 'min' and 'max' keys."
            )
        # if no error raised, then limits_dict is OK

    def __init__(self, axis: str,  **kwargs):
        # Validate axis label
        if axis not in self.VALID_AXES:
            raise ValueError(f"axis must be one of {self.VALID_AXES}")
        self._axis = axis

    @property
    @abstractmethod
    def device_info(self) -> StageInfo:
        """Returns an object describing permanent properties of the stage."""
        pass

    @property
    def axis(self) -> str:
        """
        The axis along which the stage operates.

        VALID_AXES class attribute should contain a set of valid axes labels. 
        """
        return self._axis

    @property
    @abstractmethod
    def backlash(self) -> units.UnitQuantity:
        """
        Amount of backlash for this axis.
        
        Acquisition Workers can use this to take up backlash when required.
        """
        pass

    @property
    @abstractmethod
    def position_limits(self) -> units.RangeWithUnits:
        """Returns an object describing the stage movement limits."""
        pass

    @property
    @abstractmethod
    def position(self) -> units.UnitQuantity:
        """The current position."""
        pass

    @abstractmethod 
    def move_to(self, position: units.UnitQuantity, blocking: bool = False):
        """
        Initiate move to specified position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        pass

    @property
    @abstractmethod
    def moving(self) -> bool:   
        """Return True if the stage axis is currently moving."""
        pass

    @abstractmethod
    def move_velocity(self, velocity: units.Velocity):
        """"
        Initiate movement at velocity until stopped.
        """
        pass

    def wait_until_move_finished(self): # TODO, timeout
        """
        Blocks until finished moving.
        
        Useful when two axes need to be moved simultaneously and both need to be
        checked for completion before moving on. Move both axes (non-blocking)
        and call this method on each axis."""
        while self.moving:
            time.sleep(self.SLEEP_INTERVAL)

    @abstractmethod
    def stop(self):
        """Halts motion."""
        pass

    @abstractmethod
    def home(self, blocking: bool = False):
        """
        Initiate homing. 
        
        Choose whether to return immediately (blocking=False, default) or to
        wait until finished homing (blocking=True).
        """
        pass

    @property
    @abstractmethod
    def homed(self) -> bool:
        """Return whether the stage has been home."""
        pass
        
    @property
    @abstractmethod
    def max_velocity(self) -> units.UnitQuantity:
        """
        Return the current maximum velocity setting.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:units.UnitQuantity):
        pass

    @property
    @abstractmethod
    def acceleration(self) -> units.UnitQuantity:
        """
        Return the current acceleration used during ramp up/down phase of move.
        """
        pass

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.UnitQuantity):
        pass


class LinearStage(Stage):
    """
    Abstract linear stage with a configurable Dirigo coordinate frame.
    """
    VALID_AXES = {'x', 'y', 'z'}

    def __init__(
        self,
        axis,
        backlash: str = "0 um",
        invert_direction: bool = False,   # does +stage == away from sample?
        position_offset: str = "0 mm",
        **kwargs
    ):
        super().__init__(axis, **kwargs)

        backlash = units.Position(backlash)
        if backlash < 0:
            raise ValueError(f"Backlash cannot be less than 0, got {backlash}")
        self._backlash = backlash

        self._sign = -1 if invert_direction else 1
        self._position_offset = units.Position(position_offset)

    # --- Coordinate transforms ---
    def _device_to_dirigo(self, device_pos: units.Position) -> units.Position:
        return self._sign * device_pos + self._position_offset

    def _dirigo_to_device(self, dirigo_pos: units.Position) -> units.Position:
        return self._sign * (dirigo_pos - self._position_offset)

    # --- Public API ---
    @property
    def backlash(self) -> units.Position:
        """Amount of backlash declared in the system configuration, or zero."""
        return self._backlash
    
    @property
    def position(self) -> units.Position:
        """Current position in Dirigo coordinates."""
        return self._device_to_dirigo(self._get_device_position())

    @cached_property
    def position_limits(self) -> units.PositionRange:
        """Movement limits in Dirigo coordinates."""
        dev_lims = self._get_device_position_limits()
        lo = self._device_to_dirigo(dev_lims.min)
        hi = self._device_to_dirigo(dev_lims.max)
        lo, hi = sorted((lo, hi))
        return units.PositionRange(min=lo, max=hi)
    
    def move_to(self, position: units.Position, blocking: bool = False):
        """
        Initiate move to a position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        self._move_to_device(self._dirigo_to_device(position), blocking)

    def move_velocity(self, velocity: units.Velocity):
        """Initiate movement at velocity until stopped."""
        self._move_velocity_device(self._sign * velocity)

    # --- Device-space abstract methods (plugins must implement) ----------------------
    @abstractmethod
    def _get_device_position_limits(self) -> units.PositionRange: 
        ...

    @abstractmethod
    def _get_device_position(self) -> units.Position: 
        ...

    @abstractmethod
    def _move_to_device(self, position: units.Position, blocking: bool = False): 
        ...

    @abstractmethod
    def _move_velocity_device(self, velocity: units.Velocity): 
        ...
    
    @property
    @abstractmethod
    def max_velocity(self) -> units.Velocity:
        """
        Maximum velocity used in move operations.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:units.Velocity):
        ...

    @property
    @abstractmethod
    def acceleration(self) -> units.Acceleration:
        """Acceleration used during ramp up/down phase of move."""

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.Acceleration):
        ...


class MultiAxisStage(HardwareInterface):
    """
    Dirigo interface for an X, Y, and/or Z sample translation stage.
    """
    attr_name = "stages"
    # TODO, device info?

    @property
    @abstractmethod
    def x(self) -> LinearStage:
        """If available, returns reference to the X stage axis"""
        pass
    
    @property
    @abstractmethod
    def y(self) -> LinearStage:
        """If available, returns reference to the Y stage axis"""
        pass

    @property
    @abstractmethod
    def z(self) -> LinearStage:
        """If available, returns reference to the Z stage axis"""
        pass


class RotationStage(Stage):
    VALID_AXES = {'theta'} # other global angles?

    def __init__(self, limits: dict, backlash: str = "0 deg", **kwargs):
        super().__init__(**kwargs)

        # Validate limits
        self._validate_limits_dict(limits)
        self._limits = units.AngleRange(**limits)

        # validate backlash
        backlash = units.Angle(backlash)
        if backlash < 0:
            raise ValueError(f"Backlash cannot be less than 0, got {backlash}")
        self._backlash = backlash
    
    @property
    def backlash(self) -> units.Angle:
        return self._backlash

    @property
    def position_limits(self) -> units.AngleRange:
        """Returns an object describing the stage angular position limits."""
        # these stages may have no limits, how to handle this?
        return self._limits
    
    @property
    @abstractmethod
    def position(self) -> units.Angle:
        ...
    
    @abstractmethod 
    def move_to(self, angle: units.Angle, blocking: bool = False):
        """
        Initiate move to specified angular position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        pass

    @property
    @abstractmethod
    def max_velocity(self) -> units.AngularVelocity:
        """
        Return the current maximum angular velocity setting.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:units.AngularVelocity):
        pass

    @property
    @abstractmethod
    def acceleration(self) -> units.AngularAcceleration:
        """
        Return the angular acceleration used during ramp up/down phase of move.
        """
        pass

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.AngularAcceleration):
        pass



