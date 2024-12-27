from dataclasses import dataclass
from abc import ABC, abstractmethod

import dirigo



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
    def position_limits(self) -> dirigo.RangeWithUnits:
        """Returns an object describing the stage movement limits."""
        pass

    @property
    @abstractmethod
    def position(self) -> dirigo.UnitQuantity:
        """The current position."""
        pass

    @abstractmethod 
    def move_to(self, position: dirigo.UnitQuantity, blocking: bool = False):
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
    def max_velocity(self) -> dirigo.UnitQuantity:
        """
        Return the current maximum velocity setting.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:dirigo.UnitQuantity):
        """Sets the maximum velocity."""
        pass


class LinearStage(Stage):
    VALID_AXES = {'x', 'y', 'z'}
    
    def __init__(self, limits: dict, **kwargs):
        super().__init__(**kwargs)

        # Validate limits
        self._validate_limits_dict(limits)
        self._limits = dirigo.PositionRange(**limits)

    @property
    def position_limits(self) -> dirigo.PositionRange:
        """Returns an object describing the stage spatial position limits."""
        return self._limits
    
    @abstractmethod 
    def move_to(self, position: dirigo.Position, blocking: bool = False):
        """
        Initiate move to specified spatial position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        pass
    
    @property
    @abstractmethod
    def max_velocity(self) -> dirigo.Velocity:
        """
        Return the current maximum velocity setting.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:dirigo.Velocity):
        """Sets the maximum velocity."""
        pass


import time

import clr
kinesis_location = "C:\\Program Files\\Thorlabs\\Kinesis\\"
clr.AddReference(kinesis_location + "Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(kinesis_location + "Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference(kinesis_location + "Thorlabs.MotionControl.Benchtop.BrushlessMotorCLI.dll")
from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection, DeviceUnitConverter
from Thorlabs.MotionControl.Benchtop.BrushlessMotorCLI import *
from Thorlabs.MotionControl.Benchtop.BrushlessMotorCLI import BenchtopBrushlessMotor
from System import Decimal


class ThorlabsLinearMotor(LinearStage): # alt name: Linear brushless?
    MOVE_TIMEOUT = dirigo.Time('10 s')

    def __init__(self, position_limits: dict = None, **kwargs):
        super().__init__(**kwargs)
        # position limits may be set manually in system_config.toml, or omitted to use 
        if position_limits is None:
            self._position_limits = None # position limit property will read limits from API
        else:
            self._position_limits = dirigo.PositionRange(**position_limits)
        # more TODO

    @property # could be cached property -- but don't expect big increase in speed
    def position_limits(self) -> dirigo.PositionRange:
        if self._position_limits is None:
            min_position = Decimal.ToDouble(
                self._channel.AdvancedMotorLimits.LengthMinimum) / 1000
            max_position = Decimal.ToDouble(
                self._channel.AdvancedMotorLimits.LengthMaximum) / 1000
            return dirigo.PositionRange(min_position, max_position)
        else:
            return self._position_limits

    def move_to(self, position: dirigo.Position, blocking: bool = False):
        """
        Initiate move to specified spatial position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        #validate move
        if not self.position_limits.within_range(position):
            raise ValueError(
                f"Requested move, ({position}) beyond limits, "
                f"min: {self.position_limits.min}, max: min: {self.position_limits.max}"
            )
        
        timeout_or_blocking = 1000 * self.MOVE_TIMEOUT if blocking else 0
        self._channel.MoveTo(position * 1000, timeout_or_blocking) # 2nd arg=0 means non-blocking
            
        self._last_move_timestamp = time.perf_counter() # to fix bug where position goes to 0.0 immediately after issuing command

    def stop(self):
        """Halts motion."""
        self._channel.StopImmediate()

    @property
    def max_velocity(self) -> dirigo.Velocity:
        """
        Return the current maximum velocity setting.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        vel_params = self._channel.GetVelocityParams()
        v_max = Decimal.ToDouble(vel_params.MaxVelocity) 
        return dirigo.Velocity(v_max / 1000)  # API uses velocity in mm/s

    @max_velocity.setter
    def max_velocity(self, new_velocity: dirigo.Velocity):
        # todo, validate
        vel_params = self._channel.GetVelocityParams()
        vel_params.MaxVelocity = Decimal(new_velocity * 1000) # API uses velocity in mm/s
        self._channel.SetVelocityParams(vel_params)




class RotationStage(Stage):
    VALID_AXES = {'theta'} # other global angles?

    def __init__(self, limits: dict, **kwargs):
        super().__init__(**kwargs)

        # Validate limits
        self._validate_limits_dict(limits)
        self._limits = dirigo.AngleRange(**limits)

    @property
    def position_limits(self) -> dirigo.AngleRange:
        """Returns an object describing the stage angular position limits."""
        # these stages may have no limits, how to handle this?
        return self._limits
    
    @abstractmethod 
    def move_to(self, angle: dirigo.Angle, blocking: bool = False):
        """
        Initiate move to specified angular position.

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished moving (blocking=True).
        """
        pass

    @property
    @abstractmethod
    def max_velocity(self) -> dirigo.AngularVelocity:
        """
        Return the current maximum angular velocity setting.

        Note that this is the imposed velocity limit for moves. It is not
        necessarily the maximum attainable velocity for this stage.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:dirigo.AngularVelocity):
        """Sets the maximum angular velocity."""
        pass



class OLD:
    @property
    @abstractmethod
    def acceleration(self) -> float:
        """Return the current acceleration setting (in meters/second^2.)"""
        pass

    @acceleration.setter
    @abstractmethod
    def acceleration(self, value:float):
        """Sets the acceleration (in meters/second^2)."""
        pass


    # TODO, not sure whether to keep this
    # @abstractmethod
    # def move_at_velocity(self, direction:str, velocity:float): 
    #     pass


    # # TODO, why do we need this?
    # @property
    # @abstractmethod
    # def home_position(self):
    #     pass
    #     #return (self.max_position + self.min_position)/2
    
    # def wait_until_finished(self, sleep_period = 0.1e-3):
    #     """ Waits until not busy (ie blocking) """
    #     while self.is_busy:
    #         time.sleep(sleep_period)


class LinearMotorStageAxis(Stage):
    """
    Represents ...
    """


class StepperMotorStageAxis(Stage):
    """
    Represents ...
    """
    pass


class MultiAxisStage(ABC):
    """
    Dirogo interface for a sample translation stage. Comprises one or more axes. 
    """

    @property
    @abstractmethod
    def x(self) -> None | Stage:
        """If available, returns reference to the X stage axis"""
        pass
    
    @property
    @abstractmethod
    def y(self) -> None | Stage:
        """If available, returns reference to the Y stage axis"""
        pass

    @property
    @abstractmethod
    def z(self) -> None | Stage:
        """If available, returns reference to the Z stage axis"""
        pass



# for testing
if __name__ == "__main__":
    
    config = {
        "axis": "x",
        "limits": {"min": "0 mm", "max": "55 mm"} 
    }

    stage1 = LinearStage(**config)

    None