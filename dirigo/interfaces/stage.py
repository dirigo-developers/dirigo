from abc import ABC, abstractmethod




class StageAxis(ABC): # TODO, split into base classes for linear & stepper motors?
    """
    Dirigo interface for a single translation stage.
    """

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def busy(self) -> bool:
        """Return whether the stage is busy (i.e. in motion)."""
        pass
    
    @property
    @abstractmethod
    def homed(self) -> bool:
        """Return whether the stage has been home."""
        pass

    @abstractmethod
    def home(self, blocking:bool=False):
        """
        Initiate homing. 
        
        Choose whether to return immediately (blocking=False, default) or to
        wait until finished homing (blocking=True).
        """
        pass
    
    @property
    @abstractmethod
    def max_velocity(self) -> float:
        """
        Return the current maximum velocity setting (in meters/second).

        Note that this does not represent the stage's maximum attainable 
        velocity. It represents a velocity limit.
        """
        pass

    @max_velocity.setter
    @abstractmethod
    def max_velocity(self, value:float):
        """Sets the maximum velocity (in meters/second)."""
        pass

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

    @property
    @abstractmethod
    def position(self) -> float:
        """
        Returns the current position (in meters). 
        
        Note this property has no setter. Use move_to_position() method to 
        initiate a move.
        """
        pass

    @abstractmethod
    def move_to_position(self, position:float, blocking:bool=False):
        """
        Initiate move to position (in meters).

        Choose whether to return immediately (blocking=False, default) or to
        wait until finished homing (blocking=True).
        """
        pass

    # TODO, not sure whether to keep this
    # @abstractmethod
    # def move_at_velocity(self, direction:str, velocity:float): 
    #     pass

    @abstractmethod
    def stop(self):
        """Try to stop the stage axis movement immediately."""
        pass

    @property
    @abstractmethod
    def min_position(self) -> float:
        """Get the minimum absolute position value (in meters)."""
        pass

    @property
    @abstractmethod
    def max_position(self) -> float:
        """Get the maximum absolute position value (in meters)."""
        pass

    # TODO, why do we need this?
    @property
    @abstractmethod
    def home_position(self):
        pass
        #return (self.max_position + self.min_position)/2
    
    # def wait_until_finished(self, sleep_period = 0.1e-3):
    #     """ Waits until not busy (ie blocking) """
    #     while self.is_busy:
    #         time.sleep(sleep_period)


class Stage(ABC):
    """
    Dirogo interface for a sample translation stage. Comprises one or more axes. 
    """

    @property
    @abstractmethod
    def x(self) -> None | StageAxis:
        """If available, returns reference to the X stage axis"""
        pass
    
    @property
    @abstractmethod
    def y(self) -> None | StageAxis:
        """If available, returns reference to the Y stage axis"""
        pass

    @property
    @abstractmethod
    def z(self) -> None | StageAxis:
        """If available, returns reference to the Z stage axis"""
        pass

