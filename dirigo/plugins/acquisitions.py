from pathlib import Path
import math
from typing import Optional

from platformdirs import user_config_dir

from dirigo import units
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner
from dirigo.sw_interfaces.acquisition import AcquisitionSpec, Acquisition


TWO_PI = 2 * math.pi 

class LineAcquisitionSpec(AcquisitionSpec): 
    def __init__(
            self,
            line_width: str,
            pixel_size: str,
            buffers_per_acquisition: int | float, # float('inf')
            bidirectional_scanning: bool = False,
            fill_fraction: float = 1.0,
            digitizer_profile: Optional[str] = None,
            buffers_allocated: Optional[int] = None,
            lines_per_buffer: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.bidirectional_scanning = bidirectional_scanning 
        self.line_width = units.Position(line_width)
        self.pixel_size = units.Position(pixel_size)
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
    def scan_width(self) -> units.Position:
        """
        Returns the scan width required to reach the line width with the given
        the fill fraction (scan width > line width).
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

        self.configure_digitizer(profile_name=self.spec.digitizer_profile)
        
        # Setup scanner
        self.hw.fast_raster_scanner.amplitude = \
            self.hw.optics.object_position_to_scan_angle(self.spec.scan_width)
        # for res scanner: frequency is set (fixed), waveform is set (fixed), duty cycle is set (fixed)
        # for other scanners--TBD

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
        self.hw.fast_raster_scanner.start()        
        digi.acquire.start() # This includes the buffer allocation

        try:
            while not self._stop_event.is_set() and \
                digi.acquire.buffers_acquired < self.spec.buffers_per_acquisition:

                buffer = digi.acquire.get_next_completed_buffer()
                if hasattr(self.hw, 'stage'):
                    buffer.positions = self.read_positions()
                self.publish(buffer)

        finally:
            self.cleanup()

    def cleanup(self):
        """Closes resources started during the acquisition."""
        self.hw.digitizer.acquire.stop()
        self.hw.fast_raster_scanner.stop()

        # Put None into queue to signal finished, stop scanning
        self.publish(None)

    def read_positions(self):
        """Subclasses can override this method to provide position readout from
        stages or linear position encoders."""
        if self.hw.objective_scanner:
            return (
                self.hw.stage.x.position, 
                self.hw.stage.y.position,
                self.hw.objective_scanner.position
            )
        else:
            return (
                self.hw.stage.x.position, 
                self.hw.stage.y.position,
            )

    def _calculate_trigger_delay(self, round_down: bool = True) -> int | float:
        """Compute the number of samples to delay
        
        Set round_down to True (default value), to automatically round to
        digitizer-compatible increment.
        """
        scan_period = 1.0 / self.hw.fast_raster_scanner.frequency

        start_time = scan_period * math.acos(self.spec.fill_fraction) / (2 * math.pi)
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
        start = math.acos(ff) / TWO_PI

        if self.spec.bidirectional_scanning:
            end = 1.0 - start
        else:
            end = math.acos(-ff) / TWO_PI

        record_len = (end - start) * self.hw.digitizer.sample_clock.rate \
            / self.hw.fast_raster_scanner.frequency
        
        # For tolerance to error in initial frequency, extend record length
        record_len *= 1.01
        
        if round_up:
            # Round record length up to the next allowable size (or the min)
            rlr = self.hw.digitizer.acquire.record_length_resolution
            record_len = rlr * int(record_len / rlr + 1) 

            # Also set enforce the min record length requirement
            if record_len < self.hw.digitizer.acquire.record_length_minimum:
                record_len = self.hw.digitizer.acquire.record_length_minimum
        print("record length", record_len)
        return record_len



class FrameAcquisitionSpec(LineAcquisitionSpec):
    def __init__(self, frame_height: str, flyback_periods: int, pixel_height: str = None, **kwargs):
        super().__init__(**kwargs)

        self.frame_height = units.Position(frame_height)

        if pixel_height is not None:
            self.pixel_height = units.Position(pixel_height)
        else:
            # If no pixel height is specified, assume square pixel shape
            self.pixel_height = self.pixel_size

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
            self.hw.optics.object_position_to_scan_angle(spec.frame_height)
        self.hw.slow_raster_scanner.frequency = (
            self.hw.fast_raster_scanner.frequency / spec.records_per_buffer
        )
        self.hw.slow_raster_scanner.waveform = 'asymmetric triangle'
        self.hw.slow_raster_scanner.duty_cycle = (
            1 - spec.flyback_periods / spec.records_per_buffer
        )

        self.hw.slow_raster_scanner.prepare_frame_clock(self.hw.fast_raster_scanner, spec)

    def run(self):
        self.hw.slow_raster_scanner.start()

        super().run() # The hard work is done by super's run method

    def cleanup(self):
        """Over-ride LineAcquisition's finally method to alter the HW stop order"""
        self.hw.slow_raster_scanner.stop()
        super().cleanup()


