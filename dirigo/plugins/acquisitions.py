from pathlib import Path
import math
from typing import Optional

from platformdirs import user_config_dir

from dirigo import units
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import (
    FastRasterScanner, SlowRasterScanner, GalvoScanner, ResonantScanner
)
from dirigo.sw_interfaces.acquisition import AcquisitionSpec, Acquisition


TWO_PI = 2 * math.pi 

class LineAcquisitionSpec(AcquisitionSpec): 
    """Specification for a point-scanned line acquisition"""
    MAX_PIXEL_SIZE_ADJUSTMENT = 0.01
    def __init__(
            self,
            line_width: str,
            pixel_size: str,
            buffers_per_acquisition: int | float, # float('inf')
            bidirectional_scanning: bool = False,
            pixel_rate: str = None, # e.g. "100 kHz"
            fill_fraction: float = 1.0,
            digitizer_profile: Optional[str] = None,
            buffers_allocated: Optional[int] = None,
            lines_per_buffer: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.bidirectional_scanning = bidirectional_scanning 
        if pixel_rate:
            self.pixel_rate = units.Frequency(pixel_rate)
        else:
            self.pixel_rate = None # nonlinear scan path (ie resonant scanning)
        self.line_width = units.Position(line_width)
        pixel_size = units.Position(pixel_size)

        # Adjust pixel size such that line_width/pixel_size is an integer
        self.pixel_size = self.line_width / round(self.line_width / pixel_size)
        if abs(self.pixel_size - pixel_size) / pixel_size > self.MAX_PIXEL_SIZE_ADJUSTMENT:
            raise ValueError(
                f"To maintain integer number of pixels in line, required adjusting " \
                f"the pixel size by more than pre-specified limit: {100*self.MAX_PIXEL_SIZE_ADJUSTMENT}%"
            )

        if not (0 < fill_fraction <= 1):
            raise ValueError(f"Invalid fill fraction, got {fill_fraction}. "
                             "Must be between 0.0 and 1.0 (upper bound incl.)")
        self.fill_fraction = fill_fraction
        # TODO validate lines per buffer
        self.lines_per_buffer = lines_per_buffer

        # Validate buffers per acquisition
        if isinstance(buffers_per_acquisition, str):
            if buffers_per_acquisition.lower() == "inf":
                buffers_per_acquisition = float('inf')
            else:
                raise ValueError(f"`buffers_per_acquisition` must be a finite int or a string, 'inf'.")
        elif isinstance(buffers_per_acquisition, float):
            if not buffers_per_acquisition == float('inf'):
                raise ValueError(f"`buffers_per_acquisition` must be integer or string 'inf'.")
        elif not isinstance(buffers_per_acquisition, int):
            raise ValueError(f"`buffers_per_acquisition` must be integer or string, 'inf'.")
        elif buffers_per_acquisition < 1:
            raise ValueError(f"`buffers_per_acquisition` must be > 0.")
        self.buffers_per_acquisition = buffers_per_acquisition

        self.buffers_allocated = buffers_allocated
        self.digitizer_profile = digitizer_profile    

    # Convenience properties
    @property
    def extended_scan_width(self) -> units.Position:
        """
        Returns the desired line width divided by the fill fraction. For sinusoidal
        scanpaths this is the full scan amplitude required to cover the line
        width given a certain fill fraction.
        """
        return units.Position(self.line_width / self.fill_fraction)
    
    @property
    def records_per_buffer(self) -> int:
        """
        Returns the number of records (i.e. triggered recordings) per buffer.

        A value < lines_per_buffer indicates that data for multiple lines is 
        contained in each record (e.g. bi-directional scanning).
        """
        if self.bidirectional_scanning:
            return self.lines_per_buffer // 2
        else:
            return self.lines_per_buffer
        
    @property
    def pixels_per_line(self) -> int:
        """
        Returns the number of pixels per line.

        If the line width is not divisible by pixel size, rounds to nearest 
        integer pixel.
        """
        return round(self.line_width / self.pixel_size)


class LineAcquisition(Acquisition):
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/linescan"
    SPEC_OBJECT = LineAcquisitionSpec
    
    def __init__(self, hw, spec: LineAcquisitionSpec):
        """Initialize a line acquisition worker."""
        # Tip: since this method runs on main thread, limit to HW init tasks that will return fast, prepend slower tasks to run method
        super().__init__(hw, spec) # sets up thread, inbox, stores hw, checks resources
        self.spec: LineAcquisitionSpec # to refine type hints

        # If using galvo scanner, then set it up based on acquisition spec parameters
        if isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            self.hw.fast_raster_scanner.amplitude = \
                self.hw.laser_scanning_optics.object_position_to_scan_angle(spec.line_width)
            self.hw.fast_raster_scanner.frequency = \
                self.spec.pixel_rate / round(self.spec.pixels_per_line / self.spec.fill_fraction) # TODO, pixel periods per line OK?
            self.hw.fast_raster_scanner.waveform = "asymmetric triangle"
            self.hw.fast_raster_scanner.duty_cycle = self.spec.fill_fraction
        
        else:
            self.hw.fast_raster_scanner.amplitude = \
                self.hw.laser_scanning_optics.object_position_to_scan_angle(self.spec.extended_scan_width)
            # for res scanner: frequency fixed, waveform fixed, duty cycle fixed
        
        self.configure_digitizer(profile_name=self.spec.digitizer_profile)

    def configure_digitizer(self, profile_name: str):
        """
        Loads digitizer profile and sets record and buffer settings.

        Digitizer profile is a set of user-modifiable settings, like channels
        enabled, voltage ranges, impedances, trigger source, sample rate, etc.

        Record and buffer settings, such as record length, number of records per
        buffer, etc. are automatically calculated for the profile, acquisition
        specificiaton, and other system properties.
        """
        digi = self.hw.digitizer # for brevity

        digi.load_profile(profile_name)
        if self.spec.pixel_rate:
            digi.sample_clock.rate = self.spec.pixel_rate

        # Configure acquisition timing and sizes
        digi.acquire.pre_trigger_samples = 0 # TODO, maybe allow this to be adjustable?
        digi.acquire.timestamps_enabled = True #testing
        digi.acquire.trigger_delay_samples = self._calculate_trigger_delay()
        digi.acquire.record_length = self._calculate_record_length()
        digi.acquire.records_per_buffer = self.spec.records_per_buffer
        digi.acquire.buffers_per_acquisition = self.spec.buffers_per_acquisition
        digi.acquire.buffers_allocated = self.spec.buffers_allocated

    def run(self):
        digi = self.hw.digitizer # for brevity

        # Start scanner & digitizer
        if isinstance(self.hw.fast_raster_scanner, ResonantScanner):
            self.hw.fast_raster_scanner.start()        
            digi.acquire.start() # This includes the buffer allocation
        elif isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            digi.acquire.start()
            self.hw.fast_raster_scanner.start(
                pixel_frequency=self.spec.pixel_rate,
                pixels_per_period=self.spec.pixels_per_line,
                periods_per_write=self.spec.records_per_buffer
            )

        try:
            while not self._stop_event.is_set() and \
                digi.acquire.buffers_acquired < self.spec.buffers_per_acquisition:
                print(f"Acquired {digi.acquire.buffers_acquired} of {self.spec.buffers_per_acquisition}")
                buffer = digi.acquire.get_next_completed_buffer()
                if self.hw.stage or self.hw.objective_scanner:
                    buffer.positions = self.read_positions()
                self.publish(buffer)

        finally:
            self.cleanup()

    def cleanup(self):
        """Closes resources started during the acquisition."""
        self.hw.digitizer.acquire.stop()
        self.hw.fast_raster_scanner.stop()

        # Put None into queue to signal to subscribers that we are finished
        self.publish(None)

    def read_positions(self):
        """Subclasses can override this method to provide position readout from
        stages or linear position encoders."""
        positions = []
        if self.hw.stage:
            positions.append(self.hw.stage.x.position)
            positions.append(self.hw.stage.y.position)

        if self.hw.objective_scanner:
            positions.append(self.hw.objective_scanner.position)
        
        return tuple(positions) if len(positions) else None

    def _calculate_trigger_delay(self, round_down: bool = True) -> int | float:
        """Compute the number of samples to delay
        
        Set round_down to True (default value), to automatically round to
        digitizer-compatible increment.
        """
        scan_period = units.Time(1.0 / self.hw.fast_raster_scanner.frequency)

        if isinstance(self.hw.fast_raster_scanner, ResonantScanner):
            start_time = scan_period * math.acos(self.spec.fill_fraction) / (2 * math.pi)
        else:
            start_time = 0

        # For tolerance to sync signal phase error and/or initial scanner 
        # frequency error, make sure to start earlier than absolutely required
        start_time *= 0.995

        start_index = start_time * self.hw.digitizer.sample_clock.rate

        if round_down:
            tdr = self.hw.digitizer.acquire.trigger_delay_sample_resolution
            start_index = tdr * int(start_index / tdr)

        return start_index

    def _calculate_record_length(self, round_up: bool = True) -> int | float:
        ff = self.spec.fill_fraction

        if isinstance(self.hw.fast_raster_scanner, ResonantScanner): #TODO consider scanner types and logic here
            start = math.acos(ff) / TWO_PI

            if self.spec.bidirectional_scanning:
                end = 1.0 - start
            else:
                end = math.acos(-ff) / TWO_PI

            record_len = (end - start) * self.hw.digitizer.sample_clock.rate \
                / self.hw.fast_raster_scanner.frequency
            
            # For tolerance to error in initial frequency, extend record length
            record_len *= 1.01

        else: #TODO make elif and refine logic with above
            record_len = round(self.spec.pixels_per_line / ff)
        
        if round_up:
            # Round record length up to the next allowable size (or the min)
            rlr = self.hw.digitizer.acquire.record_length_resolution
            record_len = rlr * math.ceil(record_len / rlr) 

            # Also set enforce the min record length requirement
            if record_len < self.hw.digitizer.acquire.record_length_minimum:
                record_len = self.hw.digitizer.acquire.record_length_minimum
        
        print("record length", record_len)
        return record_len



class FrameAcquisitionSpec(LineAcquisitionSpec):
    MAX_PIXEL_HEIGHT_ADJUSTMENT = 0.01
    def __init__(self, frame_height: str, flyback_periods: int, pixel_height: str = None, **kwargs):
        super().__init__(**kwargs)

        self.frame_height = units.Position(frame_height)

        if pixel_height is not None:
            pixel_height = units.Position(pixel_height)
        else:
            # If no pixel height is specified, assume square pixel shape
            pixel_height = self.pixel_size

        self.pixel_height = self.frame_height / round(self.frame_height / pixel_height)
        if abs(self.pixel_height - pixel_height) / pixel_height > self.MAX_PIXEL_HEIGHT_ADJUSTMENT:
            raise ValueError(
                f"To maintain integer number of pixels in frame height, required adjusting " \
                f"the pixel height by more than pre-specified limit: {100*self.MAX_PIXEL_HEIGHT_ADJUSTMENT}%"
            )

        self.flyback_periods = flyback_periods

    @property
    def lines_per_frame(self) -> int:
        """Returns the number of lines per frame.
        
        Rounds to nearest integer line number or multiple of 2 if bidirectional scanning.
        """
        if self.bidirectional_scanning:
            return 2 * round(self.frame_height / self.pixel_height / 2)
        else:
            return round(self.frame_height / self.pixel_height)

    @property
    def records_per_buffer(self) -> int:
        """Returns the number of digitizer records per buffer.

        Includes records that may be part of the slow raster axis flyback.        
        """
        if self.bidirectional_scanning:
            return (self.lines_per_frame // 2) + self.flyback_periods
        else:
            return self.lines_per_frame + self.flyback_periods


class FrameAcquisition(LineAcquisition):
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner, SlowRasterScanner]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/frame"
    SPEC_OBJECT = FrameAcquisitionSpec

    def __init__(self, hw, spec: FrameAcquisitionSpec):
        super().__init__(hw, spec)
        self.spec: FrameAcquisitionSpec

        # Set up slow scanner, fast scanner is already set up in super().__init__()
        self.hw.slow_raster_scanner.amplitude = \
            self.hw.laser_scanning_optics.object_position_to_scan_angle(spec.frame_height)
        self.hw.slow_raster_scanner.frequency = (
            self.hw.fast_raster_scanner.frequency / spec.records_per_buffer
        )
        self.hw.slow_raster_scanner.waveform = 'asymmetric triangle'
        self.hw.slow_raster_scanner.duty_cycle = (
            1 - spec.flyback_periods / spec.records_per_buffer
        )

    def run(self):
        self.hw.slow_raster_scanner.start(
            periods_per_frame=self.spec.records_per_buffer
        )

        super().run() # The hard work is done by super's run method

    def cleanup(self):
        """Extends LineAcquisition's cleanup method to stop both slow axis and fast"""
        super().cleanup()
        # LineAcquisition's cleanup (ie super().cleanup()):
        # self.hw.digitizer.acquire.stop()
        # self.hw.fast_raster_scanner.stop()

        # # Put None into queue to signal to subscribers that we are finished
        # self.publish(None)

        self.hw.slow_raster_scanner.stop()

        self.hw.slow_raster_scanner.park()
        try:
            self.hw.fast_raster_scanner.park()
        except NotImplemented:
            pass # Scanners like resonant scanners can't be parked.
        
        


