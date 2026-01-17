from abc import abstractmethod
from typing import Any
from enum import StrEnum

from pydantic import BaseModel, Field

from dirigo.components import units
from dirigo.hw_interfaces.stage import LinearStage # for z-scanner
from dirigo.hw_interfaces.hw_interface import HardwareInterface


"""
Dirigo scanner interface.
"""


class Waveforms(StrEnum):
    SINUSOID                = "sinusoid"
    LINEAR_UNIDIRECTIONAL   = "linear unidirectional"   # like an asymmetric triangle wave or sawtooth
    LINEAR_BIDIRECTIONAL    = "linear bidirectional"    # like a triangle wave
    STEP_UNIDIRECTIONAL     = "step unidirectional"     # To be implemented...

class Axes(StrEnum):
    X   = "x"
    Y   = "y"
    Z   = "z"


class RasterScannerConfig(BaseModel):
    axis: Axes = Field(
        ..., 
        description = "Raster scan axis label (e.g. x, y, or z)")
    angle_limits: units.AngleRange = Field(
        ..., 
        description = "Scan angle range"
    )


class RasterScanner(HardwareInterface):
    """Abstraction of a single raster scanner axis."""
    attr_name = "raster_scanner"
    VALID_AXES = {Axes.X, Axes.Y} # not responsible for Z axis

    def __init__(self, 
                 axis: str, 
                 angle_limits: dict, 
                 **kwargs):
        """
        Initialize the raster scanner with parameters from a dictionary.

        Args:
            - axis: (str) The axis label. See `VALID_AXES` for options.
            - angle_limits: (dict) Dict with 'min' and 'max' keys defining the scan angle range (optical).
        """
        # Validate axis label and store in private attr
        axis = Axes(axis)
        if axis not in self.VALID_AXES: 
            raise ValueError(f"axis must be one of {', '.join(self.VALID_AXES)}.")
        self._axis = axis

        # validate angle limits and store in private attr
        if not isinstance(angle_limits, dict):
            raise ValueError("`angle_limits` must be a dictionary.")
        missing_keys = {'min', 'max'} - angle_limits.keys()
        if missing_keys:
            raise ValueError(
                f"`angle_limits` must be a dictionary with 'min' and 'max' keys."
            )
        self._angle_limits = units.AngleRange(**angle_limits)

    @property
    def axis(self) -> Axes:
        """The axis along which the scanner operates."""
        return self._axis

    @property
    def angle_limits(self) -> units.AngleRange:
        """Returns an object describing the scan angle limits (optical)."""
        return self._angle_limits

    @property
    @abstractmethod
    def amplitude(self) -> units.Angle:
        """
        The peak-to-peak scan amplitude (optical).

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
    def waveform(self) -> Waveforms:
        """
        Describes the scan angle waveform.

        If waveform is not adjustable, implementations should raise 
        NotImplementedError in setter.
        """
        pass

    @waveform.setter
    @abstractmethod
    def waveform(self, new_waveform: Waveforms):
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
    flyback_time: units.Time | None


class ResonantScannerConfig(RasterScannerConfig):
    frequency: units.Frequency = Field(
        ..., 
        description = "Nominal scanner frequency (Hz)"
    )
    frequency_error: float = Field(
        ..., 
        ge=0.0, 
        description = "Normalized frequency error, e.g. 0.01 = 1%."
    )
    response_time: units.Time = Field(
        units.Time("0 s"), 
        description = "Settling time for amplitude changes."
    )


class ResonantScanner(RasterScanner):
    """
    Abstraction for resonant scanner.
    
    A resonant scanner oscillates with a fixed frequency and adjustable 
    amplitude.

    A resonant scanner is almost always the fast axis when in a raster scanning 
    system, however implementations should explicitly inhert FastRasterScanner 
    to 'mark' it as such. 

    Implementations must include methods:
        __init__ (which should include a call to super().__init__ method)
        amplitude (getter & setter)

    """
    def __init__(self, 
                 frequency: str, 
                 frequency_error: float,
                 response_time: str = "0 s",
                 **kwargs):
        """
        Args:
            - frequency: (str) Nominal frequency with units (e.g. "7.91 kHz")
            - frequency_error: (float) Normalized frequency error (e.g 1% = 0.01)
            - response_time: (str, optional) Time needed to settle at new amplitude with units (e.g. "100 ms")
        """
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
    def waveform(self):
        return Waveforms.SINUSOID
    
    @waveform.setter
    def waveform(self, _):
        raise NotImplementedError("Waveform is sinusoidal for resonant scanners.")
    
    def park(self):
        raise NotImplementedError("Resonant scanners can not be parked.")
    
    def center(self):
        """Sets resonant amplitude to 0."""
        self.amplitude = units.Angle(0) # or turn off?


class PolygonScannerConfig(RasterScannerConfig):
    facet_count: int = Field(
        ..., 
        description = "Number of mirror facets on polygon"
    )


class PolygonScanner(RasterScanner):
    """
    Abstraction for motor polygon assembly scanner.

    A polygon scanner is almost always the fast axis when in a raster scanning 
    system, however implementations should explicitly inhert FastRasterScanner 
    to 'mark' it as such. 
    """
    def __init__(self, 
                 facet_count: int, 
                 **kwargs):
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
    def waveform(self):
        return Waveforms.LINEAR_UNIDIRECTIONAL
    
    @waveform.setter
    def waveform(self, _):
        raise NotImplementedError("Waveform is sawtooth for polygonal scanners.")
    
    def park(self):
        raise NotImplementedError("Polygon scanners can not be parked.")
    
    def center(self):
        raise NotImplementedError("Polygon scanners can not be centered.")


class GalvoScannerConfig(RasterScannerConfig):
    ao_sample_rate: units.SampleRate | None = Field(
        ..., 
        description = "Analog control update rate"
    )
    input_delay: units.Time = Field(
        ...,
        description = "Voltage command to scanner movement time (latency)"
    )
    flyback_time: units.Time = Field(
        ...,
        description = "Time to allow full range flyback"
    )

class GalvoScanner(RasterScanner):
    """Abstraction for galvanometer mirror servo scanner."""
    INPUT_DELAY_RANGE = units.TimeRange(min=0, max=units.Time('0.5 ms')) # max is a bit arbitrary

    def __init__(self, 
                 ao_sample_rate: str | None = None, # This implies analog output (chance some users might want digital galvo control)
                 input_delay: str = "0 s",
                 flyback_time: str | None = None, # intended for Y axis flyback time, leave none for X axis
                 **kwargs):
        super().__init__(**kwargs)

        self._amplitude = units.Angle(0.0)
        self._offset = units.Angle(0.0)
        
        self._frequency = units.Frequency(0)
        self._waveform: Waveforms | None = None # require that uni or bidi be chosen before operation
        self._ramp_time_fraction: float | None = None

        delay = units.Time(input_delay)
        if not self.INPUT_DELAY_RANGE.within_range(delay):
            raise ValueError(f"input_delay out of valid range ({self.INPUT_DELAY_RANGE})")
        self.input_delay = delay

        if flyback_time is not None:
            self.flyback_time = units.Time(flyback_time)
        else:
            self.flyback_time = None

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

        # TODO check an extended amplitude range to account for some overswing in rounded linear waveforms
        
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
    def waveform(self) -> Waveforms: 
        if self._waveform is None:
            raise RuntimeError("Waveform not initialized.")
        return self._waveform
    
    @waveform.setter
    def waveform(self, new_waveform: Waveforms):
        if not isinstance(new_waveform, Waveforms):
            raise ValueError(
                f"Invalid  waveform type {new_waveform}. "
                f"Valid options: {[w.value for w in Waveforms]}."
            )
        self._waveform = new_waveform

    @property
    def ramp_time_fraction(self) -> float:
        """
        Fraction of scan period spent rising.

        For example, to raster scan with 450 image lines and 50 flyback lines,
        using an 'asymmetric triangle' waveform, duty cycle should be set to 0.9.

        If the current waveform has a fixed duty cycle, setter will raise a 
        ValueError.
        """
        if self.waveform == Waveforms.SINUSOID:
            return 0.0
        else:
            if self._ramp_time_fraction is None:
                raise RuntimeError("Ramp time fraction not initialized.")
            return self._ramp_time_fraction
    
    @ramp_time_fraction.setter
    def ramp_time_fraction(self, new_ramp_time_fraction: float):
        if self.waveform == Waveforms.SINUSOID:
            raise ValueError(
                f"Duty cycle for the current waveform ({self.waveform}) is not adjustable."
            )

        # Validate
        if not isinstance(new_ramp_time_fraction, float):
            raise ValueError("`duty_cycle` must be a float value.")
        if not (0.0 < new_ramp_time_fraction <= 1.0):
            raise ValueError("`duty_cycle` must be between 0 and 1 (upper limit inclusive).")
        
        self._ramp_time_fraction = new_ramp_time_fraction


class ObjectiveZScanner(LinearStage):
    """Abstraction for a objective lens motorized Z-axis stage.
    
    Despite inheriting functionality from LinearStage, the function of an 
    objective scanner is to move the beam through the sample, which is the 
    Dirigo definition of a scanner.
    """
    attr_name = "objective_z_scanner"
    VALID_AXES = {Axes.Z}
