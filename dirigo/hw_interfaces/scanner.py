from abc import ABC, abstractmethod
import re

from dirigo.components.utilities import AngleRange, Frequency

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
        self._angle_limits = AngleRange(**angle_limits)

    @property
    def axis_label(self) -> str:
        """
        The axis along which the scanner operates.

        Valid values: 'x' or 'y'
        """
        return self._axis

    @property
    def angle_limits(self) -> AngleRange:
        """Returns an object describing the scan angle limits."""
        return self._angle_limits

    @property
    @abstractmethod
    def amplitude(self) -> float:
        """
        The peak-to-peak scan amplitude, in degrees optical.

        Setting this property updates the scan amplitude. Implementations should
        document whether changes have effect immediately, at the beginning of
        the next period, or neither.
        
        Requirements:

        - Must be a positive float.
        - Must not exceed the maximum angle defined by `scan_angle_range.max`.

        Attempting to set a value outside these bounds should raise a ValueError.
        """
        pass

    @property
    @abstractmethod
    def frequency(self) -> float:
        """
        The scan frequency, in hertz.
        
        TODO
        """
        pass

    @property
    @abstractmethod
    def waveform(self) -> str:
        """
        Describes the scan angle waveform.

        Valid options: 'sinusoid', 'sawtooth', 'triangle'
        """
        pass


class FastRasterScanner(RasterScanner):
    """Abstraction for fast raster scanning axis."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
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
        
        frequency = Frequency(frequency)
        if frequency <= 0:
            raise ValueError(f"Value for frequency must be positive, "
                             f"got {frequency}")

        self._frequency = frequency

    @property
    def waveform(self):
        return 'sinusoid'
    
    @property
    def frequency(self):
        return self._frequency

    @RasterScanner.amplitude.setter
    @abstractmethod
    def amplitude(self, value: float):
        """Set the amplitude."""
        pass


class PolygonScanner(FastRasterScanner):
    @property
    def waveform(self):
        return 'sawtooth'
    
    @RasterScanner.frequency.setter
    @abstractmethod
    def frequency(self, value: float):
        """Sets the frequency, in hertz."""
        pass

    
    # @property
    # @abstractmethod
    # def nominal_scanner_frequency(self) -> float:
    #     """
    #     Returns the nominal scanner frequency in hertz. 

    #     The frequency defines the rate at which the scanner oscillates and may 
    #     or may not be equivalent to the line rate. This property represents the
    #     'specification' or an a priori measurement. It is not the instantaneous 
    #     (actual) frequency.
        
    #     Must be a positive float.
    #     """
    #     pass


class SlowRasterScanner(RasterScanner):
    """Abstraction for slow raster scanning axis."""
    @RasterScanner.amplitude.setter
    @abstractmethod
    def amplitude(self, value: float):
        """
        Set the scan amplitude, in degrees optical.
        """
        pass

    @property
    @abstractmethod
    def offset(self) -> float:
        """Returns the scan angle offset, in degrees optical."""
        pass

    @offset.setter
    @abstractmethod
    def offset(self, value: float):
        pass

    @RasterScanner.frequency.setter
    @abstractmethod
    def frequency(self, value: float):
        """
        Set the scan frequency, in hertz.
        """
        pass

    @RasterScanner.waveform.setter
    @abstractmethod
    def waveform(self, value: str):
        """
        Sets the scan angle waveform.

        Valid options: 'sinusoid', 'sawtooth', 'triangle'
        """
        pass
    
    

