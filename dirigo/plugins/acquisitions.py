from pathlib import Path
import math
import time
from typing import Optional

from platformdirs import user_config_dir
import numpy as np

from dirigo import units
from dirigo.hw_interfaces.detector import DetectorSet, Detector
from dirigo.hw_interfaces.digitizer import Digitizer
from dirigo.hw_interfaces.scanner import (
    FastRasterScanner, SlowRasterScanner, GalvoScanner, ResonantScanner,
    ObjectiveZScanner
)
from dirigo.sw_interfaces.acquisition import (
    AcquisitionSpec, Acquisition, AcquisitionProduct
)


TWO_PI = 2 * math.pi 



class SampleAcquisitionSpec(AcquisitionSpec):
    pass


class SampleAcquisition(Acquisition):
    """
    Fundamental Acquisition type for digitizer. Acquires a number of digitizer 
    samples at a rate. Should be independent of any spatial semantics.
    """
    REQUIRED_RESOURCES = [Digitizer,]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/sample"
    SPEC_OBJECT = SampleAcquisitionSpec

    def __init__(self):
        pass # TODO
    # This is essential for adding support for such things as SS-OCT, direct-
    # sampling fluorescence lifetime, pump-probe microscopy, and others...


class LineAcquisitionSpec(AcquisitionSpec): 
    """Specification for a point-scanned line acquisition"""
    MAX_PIXEL_SIZE_ADJUSTMENT = 0.01
    def __init__(
            self,
            line_width: str,
            pixel_size: str,
            buffers_per_acquisition: int | float = float('inf'),
            bidirectional_scanning: bool = False,
            pixel_time: str = None, # e.g. "1 μs"
            fill_fraction: float = 1.0,
            digitizer_profile: Optional[str] = None,
            buffers_allocated: Optional[int] = None,
            lines_per_buffer: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.bidirectional_scanning = bidirectional_scanning 
        if pixel_time:
            self.pixel_time = units.Time(pixel_time)
        else:
            self.pixel_time = None # None codes for non-constant pixel time (ie resonant scanning)
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
        # TODO validate lines per buffer [What is this for?]
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
    def extended_scan_width(self) -> units.Position: # TODO remove this b/c this calculation should be done explicitly where its needed for transparency
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
    REQUIRED_RESOURCES = [Digitizer, DetectorSet, FastRasterScanner]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/linescan"
    SPEC_OBJECT = LineAcquisitionSpec
    
    def __init__(self, hw, spec: LineAcquisitionSpec):
        """Initialize a line acquisition worker."""
        # Tip: since this method runs on main thread, limit to HW init tasks that will return fast, prepend slower tasks to run method
        super().__init__(hw, spec) # sets up thread, inbox, stores hw, checks resources
        self.spec: LineAcquisitionSpec # to refine type hints

        self.hw.digitizer.load_profile(profile_name=self.spec.digitizer_profile)
        if self.hw.digitizer.sample_clock.rate is None:
            self.hw.digitizer.sample_clock.rate = units.SampleRate(1 / self.spec.pixel_time)

        # If using galvo scanner, then set it up based on acquisition spec parameters
        if isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            self.hw.fast_raster_scanner.amplitude = \
                self.hw.laser_scanning_optics.object_position_to_scan_angle(spec.line_width)

            # Fast axis period should be multiple of digitizer sample resolution
            T_exact = self.spec.pixel_time * self.spec.pixels_per_line / self.spec.fill_fraction
            dt = units.Time(
                self.hw.digitizer.acquire.record_length_resolution / self.hw.digitizer.sample_clock.rate
            )
            T_rounded = round(T_exact / dt) * dt
            self.hw.fast_raster_scanner.frequency = 1 / T_rounded 

            self.hw.fast_raster_scanner.waveform = "asymmetric triangle"
            self.hw.fast_raster_scanner.duty_cycle = self.spec.fill_fraction
        
        else:
            self.hw.fast_raster_scanner.amplitude = \
                self.hw.laser_scanning_optics.object_position_to_scan_angle(self.spec.extended_scan_width)
            # for res scanner: frequency fixed, waveform fixed, duty cycle fixed
        
        self.configure_digitizer()

    def configure_digitizer(self):
        """
        Sets record and buffer settings.

        Record and buffer settings, such as record length, number of records per
        buffer, etc. are automatically calculated for the profile, acquisition
        specificiaton, and other system properties.
        """
        acq = self.hw.digitizer.acquire # for brevity

        # Configure acquisition timing and sizes
        acq.pre_trigger_samples = 0 # TODO, maybe allow this to be adjustable?
        acq.timestamps_enabled = True # testing
        acq.trigger_delay_samples = self._calculate_trigger_delay()
        acq.record_length = self._calculate_record_length()
        acq.records_per_buffer = self.spec.records_per_buffer
        acq.buffers_per_acquisition = self.spec.buffers_per_acquisition
        acq.buffers_allocated = self.spec.buffers_allocated

    def run(self):
        digi = self.hw.digitizer # for brevity

        # Enable detectors
        for detector in self.hw.detectors:
            try:
                detector.enabled = True
            except NotImplementedError:
                pass

        # Set up acquisition buffer pool
        shape = (
            self.spec.records_per_buffer, 
            self.hw.digitizer.acquire.record_length,
            self.hw.digitizer.acquire.n_channels_enabled
        )
        self.init_product_pool(n=3, shape=shape, dtype=np.int16)

        # Start scanner & digitizer
        if isinstance(self.hw.fast_raster_scanner, ResonantScanner):
            self.hw.fast_raster_scanner.start()
            # pause for a little while to allow res scanner to reach steady state
            if hasattr(self.hw.fast_raster_scanner, 'response_time'):
                time.sleep(self.hw.fast_raster_scanner.response_time)
            digi.acquire.start() # This includes the buffer allocation

        elif isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            if digi._mode == "analog":
                self.hw.fast_raster_scanner.start(
                    input_sample_rate=digi.sample_clock.rate,
                    input_sample_clock_channel=digi.sample_clock.source,
                    pixels_per_period=self.spec.pixels_per_line,
                    periods_per_write=self.spec.records_per_buffer
                )
                digi.acquire.start()

            elif digi._mode == "edge counting":
                digi.acquire.start()
                self.hw.fast_raster_scanner.start(
                    input_sample_rate=digi.sample_clock.rate,
                    input_sample_clock_channel=digi.sample_clock.source,
                    pixels_per_period=self.spec.pixels_per_line,
                    periods_per_write=self.spec.records_per_buffer
                )

        try:
            while not self._stop_event.is_set() and \
                digi.acquire.buffers_acquired < self.spec.buffers_per_acquisition:

                acq_buf = self.get_free_product()
                digi.acquire.get_next_completed_buffer(acq_buf)

                if self.hw.stage or self.hw.objective_scanner:
                    acq_buf.positions = self.read_positions()
                self.publish(acq_buf)

                print(f"Acquired {digi.acquire.buffers_acquired} of {self.spec.buffers_per_acquisition} ")

        finally:
            self.cleanup()

    def cleanup(self):
        """Closes resources started during the acquisition."""
        # Disable detectors
        for detector in self.hw.detectors:
            detector: Detector
            try:
                detector.enabled = False
            except NotImplementedError:
                pass

        try:
            self.hw.digitizer.acquire.stop()
        except:
            pass  # TODO, remove these try except blocks
        try:
            self.hw.fast_raster_scanner.stop()
        except:
            pass

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
        rate = self.hw.digitizer.sample_clock.rate
        f = self.hw.fast_raster_scanner.frequency

        if isinstance(self.hw.fast_raster_scanner, ResonantScanner): #TODO consider scanner types and logic here
            start = math.acos(ff) / TWO_PI

            if self.spec.bidirectional_scanning:
                end = 1.0 - start
            else:
                end = math.acos(-ff) / TWO_PI

            record_len = (end - start) * rate / f
            
            # For tolerance to error in initial frequency, extend record length
            record_len *= 1.01

        elif isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            # For galvo-galvo scanning, we record 100% of time including flyback
            record_len = round(rate / f)
        # OLD:
        # else: #TODO make elif and refine logic with above
        #     record_len = round(self.spec.pixels_per_line / ff)
        
        if round_up:
            # Round record length up to the next allowable size (or the min)
            rlr = self.hw.digitizer.acquire.record_length_resolution
            record_len = rlr * math.ceil(record_len / rlr) 

            # Also set enforce the min record length requirement
            if record_len < self.hw.digitizer.acquire.record_length_minimum:
                record_len = self.hw.digitizer.acquire.record_length_minimum
        
        return record_len


class FrameAcquisitionSpec(LineAcquisitionSpec):
    MAX_PIXEL_HEIGHT_ADJUSTMENT = 0.01
    def __init__(self, 
                 frame_height: str, 
                 flyback_periods: int, 
                 pixel_height: str = None, 
                 **kwargs):
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
    REQUIRED_RESOURCES = [Digitizer, DetectorSet, FastRasterScanner, SlowRasterScanner]
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
        except NotImplementedError:
            pass # Scanners like resonant scanners can't be parked.
        
        
class StackAcquisitionSpec(FrameAcquisitionSpec):
    def __init__(self, 
                 lower_limit: str | units.Position, 
                 upper_limit: str | units.Position, 
                 depth_spacing: str | units.Position,
                 saved_frames_per_step: int = 2, 
                 sacrificial_frames_per_step: int = 2,
                 **kwargs):
        super().__init__(**kwargs)

        self.lower_limit = units.Position(lower_limit)
        self.upper_limit = units.Position(upper_limit)
        if self.depth_range < 0:
            raise ValueError("Stack upper limit must be greater than lower limit.")

        self.depth_spacing = units.Position(depth_spacing)
        if self.depth_spacing <= 0:
            raise ValueError("Stack depth spacing must be greater than 0.")
        
        self._saved_frames_per_step = int(saved_frames_per_step)
        if not (0 <= self._saved_frames_per_step < 10):
            raise ValueError("Saved frames out of range [0,10)")
        
        self._sacrificial_frames_per_step = int(sacrificial_frames_per_step)
        if not (0 <= self._sacrificial_frames_per_step < 10):
            raise ValueError("Sacrificial frames out of range [0,10)")

    @property
    def depth_range(self) -> units.Position:
        return self.upper_limit - self.lower_limit

    @property
    def depths_per_acquisition(self) -> int:
        return int(self.depth_range / self.depth_spacing)
    

class StackAcquisition(Acquisition):
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner, SlowRasterScanner, ObjectiveZScanner]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/stack"
    SPEC_OBJECT = StackAcquisitionSpec

    def __init__(self, hw, spec: StackAcquisitionSpec):
        super().__init__(hw, spec)
        self.spec: StackAcquisitionSpec

        self._depths = np.arange(
            start=spec.lower_limit,
            stop=spec.upper_limit,
            step=spec.depth_spacing
        )

        # Set up child FrameAcquisition & subscribe to it
        spec.buffers_per_acquisition = float('inf')
        self._frame_acquisition = FrameAcquisition(hw, spec)
        self._frame_acquisition.add_subscriber(self)

        # Initialize object scanner
        self.hw.objective_scanner.max_velocity = units.Velocity("300 um/s")
        self.hw.objective_scanner.acceleration = units.Acceleration("1 mm/s^2")


    def run(self):
        """
        For video-rate frame scanning (resonant or polygon scanners), there are
        3 ways to manage axial movement during a Z stack:
            1) Continuous actuation--constant slow z axis movement, but end up 
                with shearing in data
            2) Very fast step (likely w/ piezo) during frame slow axis flyback 
            3) Step during a sacrificial frame period

        For galvo-galvo scanning, 
        """
        # Move to lower limit
        z_scanner = self.hw.objective_scanner
        z = z_scanner.position
        self.hw.objective_scanner.move_relative(self._depths[0])
        
        # wait until reached start position
        while (z_scanner.position - z) - self._depths[0] > units.Position('1 um'):
            time.sleep(0.01)

        try:
            # Start child FrameAcquisition
            self._frame_acquisition.start()

            # Get sacrificial frames
            for _ in range(self.spec._sacrificial_frames_per_step):
                # pocket the sacrificial frames (don't pass them on to subscribers)
                if self.inbox.get(block=True) is None:
                    return # runs finally: block on its way out

            for i in range(1, self.spec.depths_per_acquisition+1):
                for _ in range(self.spec._saved_frames_per_step):
                    # Wait for frame data, and pass along
                    buf: AcquisitionProduct = self.inbox.get(block=True)
                    if buf is None:
                        return # runs finally: block on its way out
                    self.publish(buf)

                    # TOOD, blank laser beam for step time (sacrificial frames)

                if i < self.spec.depths_per_acquisition:
                    # Move Z scanner to next depth
                    z_scanner.move_relative(self._depths[i]-self._depths[i-1])

                    # Wait for sacrificial frames
                    for _ in range(self.spec._sacrificial_frames_per_step):
                        # pocket the sacrificial frames (don't pass them on)
                        if self.inbox.get(block=True) is None:
                            return # runs finally: block on its way out
        finally:
            self._frame_acquisition.stop()
            self.publish(None) # publish the sentinel
            z_scanner.move_relative(-self._depths[-1])
    


class BidiFrameCalibrationSpec(FrameAcquisitionSpec):
    pass


class BidiFrameCalibration(Acquisition):
    REQUIRED_RESOURCES = [Digitizer, FastRasterScanner, SlowRasterScanner]
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/frame"
    SPEC_OBJECT = BidiFrameCalibrationSpec


    def __init__(self, hw, spec: BidiFrameCalibrationSpec):
        super().__init__(hw, spec)
        self.spec: BidiFrameCalibrationSpec

        # self._amplitudes = np.arange( # TODO
        #     start=spec.lower_limit,
        #     stop=spec.upper_limit,
        #     step=spec.spacing
        # )

        # Need # blanks, # waits

        # Set up child FrameAcquisition & subscribe to it
        spec.buffers_per_acquisition = float('inf')
        self._frame_acquisition = FrameAcquisition(hw, spec)
        self._frame_acquisition.add_subscriber(self)

        # Initialize hw

    def run(self):
        """
        """
        # Preliminary movements

        try:
            # Start child FrameAcquisition
            self._frame_acquisition.start()

            # Get sacrificial frames
            for _ in range(self.spec._sacrificial_frames_per_step):
                # pocket the sacrificial frames (don't pass them on to subscribers)
                if self.inbox.get(block=True) is None:
                    return # runs finally: block on its way out

            for i in range(1, self.spec.amplitudes_per_acquisition+1):
                for _ in range(self.spec._saved_frames_per_step):
                    # Wait for frame data, and pass along
                    buf: AcquisitionProduct = self.inbox.get(block=True)
                    if buf is None:
                        return # runs finally: block on its way out
                    self.publish(buf)

                    # TOOD, blank laser beam for step time (sacrificial frames)

                if i < self.spec.amplitudes_per_acquisition:
                    # Adjust amplitude
                    #z_scanner.move_relative(self._depths[i]-self._depths[i-1])

                    # Wait for sacrificial frames
                    for _ in range(self.spec._sacrificial_frames_per_step):
                        # pocket the sacrificial frames (don't pass them on)
                        if self.inbox.get(block=True) is None:
                            return # runs finally: block on its way out
        finally:
            self._frame_acquisition.stop()
            self.publish(None) # publish the sentinel
