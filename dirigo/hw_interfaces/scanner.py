from abc import ABC, abstractmethod

import dirigo

"""
Dirigo scanner interface.
"""
# TODO, add objective motor (or should that be grouped with stages?)



class RasterScanner(ABC):
    """Abstraction of a single raster scanner axis."""
    def __init__(self, **kwargs):
        """
        Initialize the raster scanner with parameters from a dictionary.

        Args:
            kwargs: A dictionary containing initialization parameters.
                Required keys:
                - axis (str): The axis label ('x' or 'y').
                - angle_limits (dict): A dictionary with 'min' and 'max' keys
                  defining the scan angle range in degrees.
        """
        axis = kwargs.get('axis')
        if axis not in {'x', 'y'}:
            raise ValueError("axis must be 'x' or 'y'.")
        self._axis = axis

        angle_limits = kwargs.get('angle_limits')
        if not isinstance(angle_limits, dict):
            raise ValueError(
                "angle_limits must be a dictionary."
            )
        missing_keys = {'min', 'max'} - angle_limits.keys()
        if missing_keys:
            raise ValueError(
                f"angle_limits must be a dictionary with 'min' and 'max' keys."
            )
        self._angle_limits = dirigo.AngleRange(**angle_limits)

    @property
    def axis(self) -> str:
        """
        The axis along which the scanner operates.

        Valid values: 'x' or 'y'
        """
        return self._axis

    @property
    def angle_limits(self) -> dirigo.AngleRange:
        """Returns an object describing the scan angle limits."""
        return self._angle_limits

    @property
    @abstractmethod
    def amplitude(self) -> dirigo.Angle:
        """
        The peak-to-peak scan amplitude.

        Setting this property updates the scan amplitude. Implementations should
        document whether changes have effect immediately, at the beginning of
        the next period, or neither.
        """
        pass

    @amplitude.setter
    @abstractmethod
    def amplitude(self, value: dirigo.Angle | float):
        pass

    @property
    @abstractmethod
    def frequency(self) -> dirigo.Frequency:
        """The scan frequency."""
        pass

    @frequency.setter
    @abstractmethod
    def frequency(self, value: dirigo.Frequency | float):
        pass

    @property
    @abstractmethod
    def waveform(self) -> str:
        """
        Describes the scan angle waveform.

        Valid options: 'sinusoid', 'sawtooth', 'triangle'
        """
        pass

    @waveform.setter
    @abstractmethod
    def waveform(self, new_waveform: str):
        pass


class FastRasterScanner(RasterScanner):
    """Abstraction for fast raster scanning axis."""

    @property
    @abstractmethod
    def enabled(self) -> bool: # Think on: should this be in RasterScanner?
        """
        Indicates whether the scanner is currently enabled.
        
        When True, the scanner is active. Setting this property to False 
        disables the scanner.
        """
        pass

    @enabled.setter
    @abstractmethod
    def enabled(self, new_state: bool):
        pass


class ResonantScanner(FastRasterScanner):
    """
    Abstraction for resonant scanner.
    
    A resonant scanner oscillates with a fixed frequency and adjustable 
    amplitude. Frequency may drift slightly depending on external factors such 
    as temperature.
    """
    def __init__(self, frequency: str, **kwargs):
        super().__init__(**kwargs)
        
        frequency_obj = dirigo.Frequency(frequency)
        if frequency_obj <= 0:
            raise ValueError(f"Value for frequency must be positive, "
                             f"got {frequency_obj}")

        self._frequency = frequency_obj
    
    @property
    def frequency(self):
        """
        Returns the nominal scanner frequency. 
        
        Not a measurement of the actual frequency.
        """
        return self._frequency
    
    @frequency.setter
    def frequency(self, _):
        raise NotImplementedError("Frequency is fixed for resonant scanners.")
    
    @property
    def waveform(self):
        return 'sinusoid'
    
    @waveform.setter
    def waveform(self, _):
        raise NotImplementedError("Waveform is sinusoidal for resonant scanners.")


class PolygonScanner(FastRasterScanner):
    # WIP

    @property
    def waveform(self):
        return 'sawtooth'
    
    @waveform.setter
    def waveform(self, value):
        NotImplementedError("Waveform is sawtooth for polygonal scanners.")



class SlowRasterScanner(RasterScanner):
    """Abstraction for slow raster scanning axis."""

    @property
    @abstractmethod
    def offset(self) -> dirigo.Angle:
        """
        Returns the scan angle offset.
        
        Setting this value shifts the scan range.
        """
        pass

    @offset.setter
    @abstractmethod
    def offset(self, value: float):
        pass
   
    
