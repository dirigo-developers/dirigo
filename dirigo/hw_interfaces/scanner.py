from abc import ABC, abstractmethod
from typing import Literal, Any, get_args

from dirigo.components import units
from dirigo.hw_interfaces.stage import LinearStage # for z-scanner
from dirigo.hw_interfaces.hw_interface import HardwareInterface


"""
Dirigo scanner interface.
"""

WaveformNames = Literal['sinusoid', 'asymmetric triangle', 'sawtooth']
                # e.g.   resonant    galvanometer           polygon


class RasterScanner(HardwareInterface):
    """Abstraction of a single raster scanner axis."""
    attr_name = "raster_scanner"
    VALID_AXES = {'x', 'y'}

    def __init__(self, axis: str, angle_limits: dict, **kwargs):
        """
        Initialize the raster scanner with parameters from a dictionary.

        Args:
            - axis (str): The axis label. See `VALID_AXES` for options.
            - angle_limits (dict): A dictionary with 'min' and 'max' keys defining the scan angle range.
        """
        # Validate axis label and set in private attr
        if axis not in self.VALID_AXES: 
            raise ValueError(f"axis must be one of {', '.join(self.VALID_AXES)}.")
        self._axis = axis

        # validate angle limits and set in private attr
        if not isinstance(angle_limits, dict):
            raise ValueError("`angle_limits` must be a dictionary.")
        missing_keys = {'min', 'max'} - angle_limits.keys()
        if missing_keys:
            raise ValueError(
                f"`angle_limits` must be a dictionary with 'min' and 'max' keys."
            )
        self._angle_limits = units.AngleRange(**angle_limits)

    @property
    def axis(self) -> str:
        """The axis along which the scanner operates."""
        return self._axis

    @property
    def angle_limits(self) -> units.AngleRange:
        """Returns an object describing the scan angle limits."""
        return self._angle_limits

    @property
    @abstractmethod
    def amplitude(self) -> units.Angle:
        """
        The peak-to-peak scan amplitude.

        Setting this property updates the scan amplitude. Implementations should
        document whether changes have effect immediately, at the beginning of
        the next period, or neither.

        If amplitude is not adjustable, implementations should raise 
        NotImplementedError in setter.
        """
        pass

    @amplitude.setter
    @abstractmethod
    def amplitude(self, value: units.Angle):
        pass

    @property
    @abstractmethod
    def frequency(self) -> units.Frequency:
        """The scan frequency.
        
        If frequency is not adjustable, implementations should raise 
        NotImplementedError in setter.
        """
        pass

    @frequency.setter
    @abstractmethod
    def frequency(self, value: units.Frequency):
        pass

    @property
    @abstractmethod
    def waveform(self) -> str:
        """
        Describes the scan angle waveform.

        Valid options: 'sinusoid', 'sawtooth', 'triangle', 'asymmetric triangle'

        If waveform is not adjustable, implementations should raise 
        NotImplementedError in setter.
        """
        pass

    @waveform.setter
    @abstractmethod
    def waveform(self, new_waveform: str):
        pass

    @property
    @abstractmethod
    def duty_cycle(self) -> float:
        """For asymmetric waveforms, fraction of the scan period spent rising.

        For example, to raster scan with 450 image lines and 50 flyback lines,
        the duty cycle should be set to 0.9.
        
        If asymmetric waveforms are not available, implementations should raise
        NotImplementedError in setter.
        """
        pass

    @duty_cycle.setter
    @abstractmethod
    def duty_cycle(self, duty: float):
        pass

    @abstractmethod
    def park(self):
        """Positions the scanner at minimum angle.
        
        Scanners that can not be positioned arbitrarily should raise a 
        NotImplementedError.
        """

    @abstractmethod
    def center(self) -> None:
        """
        Positions the scanner at zero angle.

        Scanners that can not be positioned arbitrarily should raise a 
        NotImplementedError.
        """

    @abstractmethod
    def start(self, **kw: Any):
        pass

    @abstractmethod
    def stop(self):
        pass


class FastRasterScanner(RasterScanner):
    """Marker class indicating that this scanner is operated as the fast axis in
    a raster scanning system.
    """
    attr_name = "fast_raster_scanner"


class SlowRasterScanner(RasterScanner):
    """Marker class indicating that this scanner is operated as the slow axis in
    a raster scanning system.
    """
    attr_name = "slow_raster_scanner"


class ResonantScanner(RasterScanner):
    """
    Abstraction for resonant scanner.
    
    A resonant scanner oscillates with a fixed frequency and adjustable 
    amplitude. Frequency may drift slightly depending on external factors such 
    as temperature.

    A resonant scanner is almost always the fast axis when in a raster scanning 
    system, however implementations should explicitly inhert FastRasterScanner 
    to mark it as such. 

    Implementations must include methods:
        __init__ (which should include a call to super().__init__ method)
        amplitude (getter & setter)

    """
    def __init__(self, 
                 frequency: str, 
                 frequency_error: float,
                 response_time: str = "0 s",
                 **kwargs):
        super().__init__(**kwargs)
        
        frequency_obj = units.Frequency(frequency)
        if frequency_obj <= 0:
            raise ValueError(f"Value for frequency must be positive, "
                             f"got {frequency_obj}")

        self._frequency = frequency_obj
        self.frequency_error = frequency_error

        r_time = units.Time(response_time)
        if not (0 <= r_time < units.Time('1 s')):
            raise ValueError(
                "Response time outside of valid range 0-1 seconds, got {r_time}."
            )
        self.response_time = r_time
    
    @property
    def frequency(self) -> units.Frequency:
        """
        Returns the nominal scanner frequency. 
        
        Not a measurement of the actual frequency.
        """
        return self._frequency
    
    @frequency.setter
    def frequency(self, _):
        raise NotImplementedError("Frequency is fixed for resonant scanners.")
    
    @property
    def waveform(self) -> str:
        return 'sinusoid'
    
    @waveform.setter
    def waveform(self, _):
        raise NotImplementedError("Waveform is sinusoidal for resonant scanners.")
    
    @property
    def duty_cycle(self) -> float:
        return 0.5 # DC = 0.5 here just means the waveform is symmetric
    
    @duty_cycle.setter
    def duty_cycle(self, _):
        raise NotImplementedError(
            "Waveform duty cycle is not adjustable with resonant scanners."
        )
    
    def park(self):
        raise NotImplementedError("Resonant scanners can not be parked.")
    
    def center(self):
        """Sets resonant amplitude to 0."""
        self.amplitude = units.Angle(0) # or turn off?


class PolygonScanner(RasterScanner):
    """
    Abstraction for motor polygon assembly scanner.

    A polygon scanner is almost always the fast axis when in a raster scanning 
    system, however implementations should explicitly inhert FastRasterScanner 
    to mark it as such. 
    """
    def __init__(self, facet_count: int, **kwargs):
        super().__init__(**kwargs)
        self._facet_count = facet_count

    @property
    def facet_count(self) -> int:
        return self._facet_count

    @property
    def amplitude(self) -> units.AngleRange:
        theta = 360 / self.facet_count
        return units.AngleRange(min=f"{-theta} deg", max=f"{theta} deg")

    @amplitude.setter
    def amplitude(self, _):
        raise NotImplementedError("Amplitude is not adjustable for polygonal scanners.")

    @property
    def waveform(self) -> str:
        return 'sawtooth'
    
    @waveform.setter
    def waveform(self, _):
        raise NotImplementedError("Waveform is sawtooth for polygonal scanners.")
    
    @property
    def duty_cycle(self) -> float:
        # Duty cycle of the scan path is 100%, but vignetting effectively limits 
        # collection on edges (not the scanner's concern, it is acquisition's 
        # responsibility to manage vignetting)
        return 1.0
    
    @duty_cycle.setter
    def duty_cycle(self, _):
        raise NotImplementedError(
            "Waveform duty cycle is not adjustable with polygon scanners."
        )
    
    def park(self):
        raise NotImplementedError("Polygon scanners can not be parked.")
    
    def center(self):
        raise NotImplementedError("Polygon scanners can not be centered.")


class GalvoScanner(RasterScanner):
    """Abstraction for galvanometer mirror servo scanner."""
    INPUT_DELAY_RANGE = units.TimeRange(min=0, max=units.Time('0.5 ms'))

    def __init__(self, 
                 ao_sample_rate: str | None = None, # This implies analog output (chance some users might want digital galvo control)
                 input_delay: str = "0 s",
                 **kwargs):
        super().__init__(**kwargs)

        self._amplitude = units.Angle(0.0)
        self._offset = units.Angle(0.0)
        
        self._frequency = units.Frequency(0)
        self._waveform: WaveformNames = "asymmetric triangle"
        self._duty_cycle: float = 0.0

        delay = units.Time(input_delay)
        if not self.INPUT_DELAY_RANGE.within_range(delay):
            raise ValueError(f"input_delay out of valid range ({self.INPUT_DELAY_RANGE})")
        self.input_delay = delay

        # Validate sample rate
        if ao_sample_rate is not None:
            rate = units.SampleRate(ao_sample_rate)
        else:
            rate = None
        self._ao_sample_rate = rate

    @property
    def amplitude(self) -> units.Angle:
        """
        The peak-to-peak scan amplitude.
        """
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, new_amplitude: units.Angle | float):
        ampl = units.Angle(new_amplitude)
        
        # Check that proposed waveform will not exceed scanner limits
        upper = units.Angle(self.offset + ampl/2)
        if not self.angle_limits.within_range(upper):
            raise ValueError(
                f"Error setting amplitude. Scan waveform would exceed scanner "
                f"upper limit ({self.angle_limits.max_degrees})."
            )
        lower = units.Angle(self.offset - ampl/2)
        if not self.angle_limits.within_range(lower):
            raise ValueError(
                f"Error setting amplitude. Scan waveform would exceed scanner "
                f"lower limit ({self.angle_limits.min_degrees})."
            )

        self._amplitude = ampl

    @property
    def offset(self) -> units.Angle:
        return self._offset
    
    @offset.setter
    def offset(self, new_offset: units.Angle | float):
        offset = units.Angle(new_offset)

        # Check that proposed waveform will not exceed scanner limits
        upper = units.Angle(offset + self.amplitude/2)
        if not self.angle_limits.within_range(upper):
            raise ValueError(
                f"Error setting offset. Scan waveform would exceed scanner "
                f"upper limit ({self.angle_limits.max_degrees})."
            )
        lower = units.Angle(offset - self.amplitude/2)
        if not self.angle_limits.within_range(lower):
            raise ValueError(
                f"Error setting offset. Scan waveform would exceed scanner "
                f"lower limit ({self.angle_limits.min_degrees})."
            )
        
        self._offset = offset

    @property
    def frequency(self) -> float:
        """
        The scanner frequency.
        """
        return self._frequency
    
    @frequency.setter
    def frequency(self, new_frequency: units.Frequency | float):
        freq = units.Frequency(new_frequency)

        # Check positive 
        if freq <= 0:
            raise ValueError(
                f"Error setting frequency. Must be positve, got {freq}"
            )
        
        self._frequency = freq

    @property
    def waveform(self) -> WaveformNames: 
        return self._waveform
    
    @waveform.setter
    def waveform(self, new_waveform: WaveformNames):
        if new_waveform not in get_args(WaveformNames):
            raise ValueError(
                f"Error setting waveform type. Valid options 'sinusoid', "
                f"'sawtooth', 'triangle'. Recieved {new_waveform}"
            )
        self._waveform = new_waveform

    @property
    def duty_cycle(self) -> float:
        """
        Fraction of scan period spent rising.

        For example, to raster scan with 450 image lines and 50 flyback lines,
        using an 'asymmetric triangle' waveform, duty cycle should be set to 0.9.

        If the current waveform has a fixed duty cycle, setter will raise a 
        ValueError.
        """
        if self.waveform in {'sinusoid', 'triangle'}:
            return 0.5
        elif self.waveform in {'sawtooth'}:
            return 1.0
        else:
            return self._duty_cycle
    
    @duty_cycle.setter
    def duty_cycle(self, new_duty_cycle: float):
        if self.waveform in {'sinusoid', 'triangle', 'sawtooth'}:
            raise ValueError(
                f"Duty cycle for the current waveform ({self.waveform}) is not adjustable."
            )

        # Validate
        if not isinstance(new_duty_cycle, float):
            raise ValueError("`duty_cycle` must be a float value.")
        if not (0.0 < new_duty_cycle <= 1.0):
            raise ValueError("`duty_cycle` must be between 0 and 1 (upper limit inclusive).")
        
        self._duty_cycle = new_duty_cycle

    

class ObjectiveZScanner(LinearStage):
    """Abstraction for a objective lens motorized Z-axis stage.
    
    Despite inheriting functionality from LinearStage, the function of an 
    objective scanner is to move the beam through the sample, which is the 
    Dirigo definition of a scanner.
    """
    attr_name = "objective_z_scanner"
    VALID_AXES = {'z'}
