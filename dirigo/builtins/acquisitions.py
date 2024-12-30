from pathlib import Path
import math
import queue
from typing import Union

from platformdirs import user_config_dir

import dirigo
from dirigo.components.io import load_toml
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import FastRasterScanner, SlowRasterScanner
from dirigo.sw_interfaces import Acquisition



class LineAcquisitionSpec(): # TODO, this could inherit from an overall AcquisitionSpec
    def __init__(
            self,
            bidirectional_scanning: bool,
            line_width: str,
            pixel_size: str,
            fill_fraction: float,
            buffers_per_acquisition: int | float, # float('inf')
            buffers_allocated: int,
            digitizer_profile: str,
            lines_per_buffer: int = None,
    ):
        self.bidirectional_scanning = bidirectional_scanning 
        self.line_width = dirigo.Position(line_width)
        self.pixel_size = dirigo.Position(pixel_size)
        if not (0 < fill_fraction <= 1):
            raise ValueError(f"Invalid fill fraction, got {fill_fraction}. "
                             "Must be between 0.0 and 1.0 (upper bound incl.)")
        self.fill_fraction = fill_fraction
        # TODO validate lines per buffer
        self.lines_per_buffer = lines_per_buffer
        if isinstance(buffers_per_acquisition, float) and not buffers_per_acquisition == float('inf'):
            raise ValueError(f"`buffers_per_acquisition` cannot be a non-infinite float.")
        self.buffers_per_acquisition = buffers_per_acquisition
        self.buffers_allocated = buffers_allocated
        self.digitizer_profile = digitizer_profile    

    # Convenience properties
    @property
    def scan_width(self) -> dirigo.Position:
        """
        Returns the scan width required to reach the line width with the given
        the fill fraction (scan width > line width).
        """
        return dirigo.Position(self.line_width / self.fill_fraction)
    
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
    def pixels_per_line(self) -> float:
        """
        Returns the number of pixels per line.
        
        Note: The caller must decide how to handle a fractional pixel count 
        (for instance, by either rounding up or down).
        """
        return self.line_width / self.pixel_size


class LineAcquisition(Acquisition):
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/linescan"
    SPEC_OBJECT = LineAcquisitionSpec
    
    def __init__(self, hw, data_queue: queue.Queue, spec: LineAcquisitionSpec):
        """Initialize a line acquisition worker."""
        # Tip: since this method runs on main thread, limit to HW init tasks that will return fast, prepend slower tasks to run method
        super().__init__(hw, data_queue, spec) # stores hw and queue, checks resources
        self.spec: LineAcquisitionSpec # to refine type hints

        # Get digitizer default profile, set it
        self.hw.digitizer.load_profile("default")

        # Setup digitizer buffers
        self.hw.digitizer.acquire.pre_trigger_samples = 0 # TODO, maybe allow this to be adjustable?
        self.hw.digitizer.acquire.record_length = self._calculate_record_length()
        self.hw.digitizer.acquire.records_per_buffer = spec.records_per_buffer
        self.hw.digitizer.acquire.buffers_per_acquisition = spec.buffers_per_acquisition
        self.hw.digitizer.acquire.buffers_allocated = spec.buffers_allocated

        # Setup scanner
        self.hw.fast_raster_scanner.amplitude = \
            self.hw.optics.object_position_to_scan_angle(self.spec.scan_width)
        # for res scanner: frequency is set (fixed), waveform is set (fixed), duty cycle is set (fixed)
        # for other scanners--TBD

    def run(self):
        digitizer = self.hw.digitizer # for brevity in this method

        # Start scanner & digitizer
        self.hw.fast_raster_scanner.start()        
        digitizer.acquire.start() # This includes the buffer allocation

        try:
            while not self._stop_event.is_set() and \
                digitizer.acquire.buffers_acquired < self.spec.buffers_per_acquisition:
                
                buffer_data = digitizer.acquire.get_next_completed_buffer()
                
                self.data_queue.put(buffer_data)
                print(f"Got buffer {digitizer.acquire.buffers_acquired}")

        finally:
            # Stop scanner
            self.hw.fast_raster_scanner.stop()

    def _calculate_record_length(self, round_up: bool = True) -> int | float:
        # Calculate record length
        scan_per = 1 / self.hw.fast_raster_scanner.frequency
        
        if self.spec.bidirectional_scanning:
            record_per = scan_per * math.asin((self.spec.fill_fraction+1) / 2) * 2/math.pi
        else:
            record_per = scan_per/2  * math.asin(self.spec.fill_fraction) * 2/math.pi

        record_len = record_per * self.hw.digitizer.sample_clock.rate
       
        if round_up:
            # Round record length up to the next allowable size (or the min)
            rlr = self.hw.digitizer.acquire.record_length_resolution
            record_len = rlr * int(record_len / rlr + 1) 

            if record_len < self.hw.digitizer.acquire.record_length_minimum:
                record_len = self.hw.digitizer.acquire.record_length_minimum
        
        return record_len



class FrameAcquisitionSpec(LineAcquisitionSpec):
    def __init__(self, frame_height: str, flyback_periods: int, pixel_height: str = None, **kwargs):
        super().__init__(**kwargs)

        self.frame_height = dirigo.Position(frame_height)

        if pixel_height is not None:
            self.pixel_height = dirigo.Position(pixel_height)
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

    def __init__(self, hw, data_queue: queue.Queue, spec: FrameAcquisitionSpec):
        super().__init__(hw, data_queue, spec)
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

        try:
            super().run()

        finally:
            self.hw.slow_raster_scanner.stop()




