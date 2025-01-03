from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

from dirigo import units



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


class Stage(ABC):
    """
    Abstract interface for a single stage.
    """
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

    def __init__(self, axis: str):
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
    VALID_AXES = {'x', 'y', 'z'}
    
    # Thoughts on scrapping this?
    # def __init__(self, limits: dict, **kwargs):
    #     super().__init__(**kwargs)

    #     # Validate limits
    #     self._validate_limits_dict(limits)
    #     self._limits = dirigo.PositionRange(**limits)

    # @property
    # def position_limits(self) -> dirigo.PositionRange:
    #     """Returns an object describing the stage spatial position limits."""
    #     return self._limits
    
    @abstractmethod 
    def move_to(self, position: units.Position, blocking: bool = False):
        """
        Initiate move to specified spatial position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        pass
    
    @property
    @abstractmethod
    def max_velocity(self) -> units.Velocity:
        """
        Return the maximum velocity used in move operations.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:units.Velocity):
        """Sets the maximum velocity."""
        pass

    @property
    @abstractmethod
    def acceleration(self) -> units.Acceleration:
        """
        Return the acceleration used during ramp up/down phase of move.
        """
        pass

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value: units.Acceleration):
        pass


class MultiAxisStage(ABC):
    """
    Dirigo interface for an X, Y, and/or Z sample translation stage.
    """
    # TODO, device info?

    @property
    @abstractmethod
    def x(self) -> None | LinearStage:
        """If available, returns reference to the X stage axis"""
        pass
    
    @property
    @abstractmethod
    def y(self) -> None | LinearStage:
        """If available, returns reference to the Y stage axis"""
        pass

    @property
    @abstractmethod
    def z(self) -> None | LinearStage:
        """If available, returns reference to the Z stage axis"""
        pass


class RotationStage(Stage):
    VALID_AXES = {'theta'} # other global angles?

    def __init__(self, limits: dict, **kwargs):
        super().__init__(**kwargs)

        # Validate limits
        self._validate_limits_dict(limits)
        self._limits = units.AngleRange(**limits)

    @property
    def position_limits(self) -> units.AngleRange:
        """Returns an object describing the stage angular position limits."""
        # these stages may have no limits, how to handle this?
        return self._limits
    
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


# Not sure whether this distinction is necessary
class LinearMotorStageAxis(LinearStage):
    """
    Represents ...
    """
    pass


class StepperMotorStageAxis(Stage):
    """
    Represents ...
    """
    pass


