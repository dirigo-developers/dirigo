from pathlib import Path
import threading
import math
import time
from typing import Optional, Type
from functools import cached_property
from dataclasses import dataclass, asdict

from platformdirs import user_config_dir
import numpy as np

from dirigo.components import units
from dirigo.hw_interfaces.detector import DetectorSet, Detector
from dirigo.hw_interfaces.digitizer import Digitizer, DigitizerProfile
from dirigo.hw_interfaces.scanner import (
    FastRasterScanner, SlowRasterScanner, GalvoScanner, ResonantScanner,
    ObjectiveZScanner
)
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
    

TWO_PI = 2 * math.pi 


class SampleAcquisitionSpec(Acquisition.Spec):
    def __init__(
            self,
            digitizer_profile: str = "default",
            timestamps_enabled: bool = True,
            pre_trigger_samples: int = 0,
            trigger_delay_samples: Optional[int] = None,
            record_length: Optional[int] = None,
            records_per_buffer: int = 8,
            buffers_per_acquisition: int | float = float('inf'),
            buffers_allocated: int = 4,
            **kwargs
            ) -> None:
        
        self.digitizer_profile = digitizer_profile
        self.pre_trigger_samples = pre_trigger_samples
        self.timestamps_enabled = timestamps_enabled
        self.trigger_delay_samples = trigger_delay_samples
        self.record_length = record_length
        try:
            self.records_per_buffer = records_per_buffer
        except AttributeError:
            pass # if subclass implements a property for `records_per_buffer`

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

    
class SampleAcquisition(Acquisition):
    """
    Fundamental Acquisition type for Digitizer. Acquires a number of digitizer 
    samples at some rate. Should be independent of any spatial semantics.
    """
    required_resources = (Digitizer,)
    Spec: Type[SampleAcquisitionSpec] = SampleAcquisitionSpec

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec) # sets up thread, inbox, stores hw, checks resources
        self.spec: SampleAcquisitionSpec # to refine type hints    
        self.active = threading.Event()  # to indicate data acquisition occuring

        self.hw.digitizer.load_profile(profile_name=self.spec.digitizer_profile)
    
    def configure_digitizer(self):
        """
        Sets record and buffer settings.

        Record and buffer settings, such as record length, number of records per
        buffer, etc. are automatically calculated for the profile, acquisition
        specificiaton, and other system properties.
        """
        acq = self.hw.digitizer.acquire # for brevity

        # Configure acquisition timing and sizes
        acq.pre_trigger_samples = self.spec.pre_trigger_samples
        acq.timestamps_enabled = self.spec.timestamps_enabled
        acq.trigger_delay_samples = self.trigger_delay
        acq.record_length = self.record_length
        acq.records_per_buffer = self.spec.records_per_buffer
        acq.buffers_per_acquisition = self.spec.buffers_per_acquisition
        acq.buffers_allocated = self.spec.buffers_allocated

    def run(self):
        # TODO, should start and stop digitizer, set `active` event
        raise NotImplementedError 

    @property
    def trigger_delay(self) -> int:
        """
        Acquisition start delay, in sample periods.
        
        Subclassses can override this to synchronize with other hardware.
        """
        return self.spec.trigger_delay_samples
    
    @property
    def record_length(self) -> int:
        """
        Acquisition record length, in sample periods.
        
        Subclassses can override this to synchronize with other hardware.
        """
        return self.spec.record_length
    
    @property
    def digitizer_profile(self) -> DigitizerProfile:
        return self.hw.digitizer.profile


class LineAcquisitionSpec(SampleAcquisitionSpec): 
    """Specification for a point-scanned line acquisition"""
    MAX_PIXEL_SIZE_ADJUSTMENT = 0.01
    def __init__(
        self,
        line_width: str,
        pixel_size: str,
        bidirectional_scanning: bool = False,
        pixel_time: str = None, # e.g. "1 μs"
        fill_fraction: float = 1.0,
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
        self.lines_per_buffer = lines_per_buffer

    # Convenience properties
    @property
    def extended_scan_width(self) -> units.Position: # TODO remove this b/c this calculation should be done explicitly where its needed for transparency
        """
        Returns the desired line width divided by the fill fraction. For sinusoidal
        scan paths this is the full scan amplitude required to cover the line
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


@dataclass
class LineAcquisitionRuntimeInfo:
    """
    Makes runtime LineAcquisition info and parameters available.
    (Experimental) 
    """
    scanner_amplitude: units.Angle
    digitizer_bit_depth: int
    digitizer_trigger_delay: int

    @classmethod
    def from_acquisition(cls, acquisition: "LineAcquisition"):
        return cls(
            scanner_amplitude=acquisition.hw.fast_raster_scanner.amplitude,
            digitizer_bit_depth=acquisition.hw.digitizer.bit_depth,
            digitizer_trigger_delay=acquisition.hw.digitizer.acquire.trigger_delay_samples
        )
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            scanner_amplitude=units.Angle(d['scanner_amplitude']),
            digitizer_bit_depth=int(d['digitizer_bit_depth']),
            digitizer_trigger_delay=int(d['digitizer_trigger_delay'])
        )
    
    def to_dict(self) -> dict:
        """Make dictionary for serialization."""
        return asdict(self)


class LineAcquisition(SampleAcquisition):
    required_resources = (Digitizer, DetectorSet, FastRasterScanner)
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/line"
    Spec: Type[LineAcquisitionSpec] = LineAcquisitionSpec
    
    def __init__(self, hw, system_config, spec):
        """Initialize a line acquisition worker."""
        # Tip: since this method runs on main thread, limit to HW init tasks that will return fast, prepend slower tasks to run method
        super().__init__(hw, system_config, spec) # sets up thread, inbox, stores hw, checks resources
        self.spec: LineAcquisitionSpec # to refine type hints  

        # for brevity
        scanner = self.hw.fast_raster_scanner
        optics = self.hw.laser_scanning_optics
        digi = self.hw.digitizer
       
        if digi.sample_clock.rate is None:
            # if sample rate is not set, set it based on pixel dwell time
            digi.sample_clock.rate = units.SampleRate(1 / spec.pixel_time)

        # If using galvo scanner, then set it up based on acquisition spec parameters
        if isinstance(scanner, GalvoScanner):
            scanner.amplitude = optics.object_position_to_scan_angle(
                spec.line_width,
                axis="fast"
            )

            # Fast axis period should be multiple of digitizer sample resolution
            T_exact = spec.pixel_time * spec.pixels_per_line / spec.fill_fraction
            dt = units.Time(
                digi.acquire.record_length_resolution / digi.sample_clock.rate
            )
            T_rounded = round(T_exact / dt) * dt
            scanner.frequency = 1 / T_rounded 
            scanner.waveform = "asymmetric triangle"
            scanner.duty_cycle = spec.fill_fraction # TODO set duty cycle to 50% if doing bidi
        
        elif isinstance(scanner, ResonantScanner):
            # for res scanner: fixed: frequency, waveform, duty cycle; 
            # adjustable: amplitude
            scanner.amplitude = optics.object_position_to_scan_angle(
                spec.extended_scan_width,
                #axis="fast"
            )
            
        # Scanner settings implemented, configure digitizer acquisition params
        self.configure_digitizer()

        # Capture runtime info
        self.runtime_info = LineAcquisitionRuntimeInfo.from_acquisition(self)

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
            digi.acquire.record_length,
            digi.acquire.n_channels_enabled
        )
        self.init_product_pool(n=3, shape=shape, dtype=np.int16)

        # Start scanner & digitizer
        if isinstance(self.hw.fast_raster_scanner, ResonantScanner):
            self.hw.fast_raster_scanner.start()
            # pause for a little while to allow res scanner to reach steady state
            if hasattr(self.hw.fast_raster_scanner, 'response_time'):
                time.sleep(self.hw.fast_raster_scanner.response_time)
            digi.acquire.start() # This includes the buffer allocation
            self.active.set()

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
                #print("Buffers acquired", digi.acquire.buffers_acquired)
                acq_product = self.get_free_product()
                digi.acquire.get_next_completed_buffer(acq_product)

                if self.hw.stage or self.hw.objective_scanner:
                    t0 = time.perf_counter()
                    acq_product.positions = self.read_positions()
                    t1 = time.perf_counter()

                self.publish(acq_product)

                print(f"Acquired {digi.acquire.buffers_acquired} of {self.spec.buffers_per_acquisition} "
                      f"Reading stage positions took: {1000*(t1-t0):.3f} ms")
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

    @cached_property
    def trigger_delay(self) -> int:
        """
        Acquisition start delay, in sample periods. Rounds to the nearest
        digitizer-compatible increment.
        """

        if self.spec.bidirectional_scanning:
            start_index = 128 # should be half the trigger re-arm time TODO, get actual rearm time
        else:
            start_index = 0  

        # Round 
        tdr = self.hw.digitizer.acquire.trigger_delay_sample_resolution
        start_index = tdr * round(start_index / tdr)

        return start_index

    @property
    def record_length(self) -> int:
        """
        Acquisition record length, in sample periods. Rounds up to the nearest
        digitizer-compatible increment.
        """
        scanner_info = self.system_config.fast_raster_scanner
        digitizer_rate = self.digitizer_profile.sample_clock.rate

        nominal_period = units.Time(1 / units.Frequency(scanner_info['frequency']))
        if 'resonant' in scanner_info['type']:
            shortest_period = nominal_period / 1.005 # TODO set max frequency error

            if self.spec.bidirectional_scanning:
                record_duration = shortest_period
                record_length = record_duration * digitizer_rate - 256 # TODO, get actual rearm time
            else:
                record_duration = shortest_period / 2
                record_length = record_duration * digitizer_rate

        elif isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            # For galvo-galvo scanning, we record 100% of time including flyback
            record_length = nominal_period * digitizer_rate
        
        # Round record length up to the nearest allowable size (or the min)
        rlr = self.hw.digitizer.acquire.record_length_resolution
        record_length = rlr * round(record_length / rlr) 

        # Also set enforce the min record length requirement
        if record_length < self.hw.digitizer.acquire.record_length_minimum:
            record_length = self.hw.digitizer.acquire.record_length_minimum
        
        return record_length


class FrameAcquisitionSpec(LineAcquisitionSpec):
    """Specifications for frame series acquisition"""
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
    required_resources = (Digitizer, DetectorSet, FastRasterScanner, SlowRasterScanner)
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/frame"
    Spec: Type[FrameAcquisitionSpec] = FrameAcquisitionSpec

    def __init__(self, hw, system_config, spec: FrameAcquisitionSpec):
        super().__init__(hw, system_config, spec)
        self.spec: FrameAcquisitionSpec # to refine type hints

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
    required_resources = (Digitizer, FastRasterScanner, SlowRasterScanner, ObjectiveZScanner)
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/stack"
    Spec: Type[StackAcquisitionSpec] = StackAcquisitionSpec

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec)
        self.spec: StackAcquisitionSpec # to refine type hints

        self._depths = np.arange(
            start=self.spec.lower_limit,
            stop=self.spec.upper_limit,
            step=self.spec.depth_spacing
        )

        # Set up child FrameAcquisition & subscribe to it
        self.spec.buffers_per_acquisition = float('inf')
        self._frame_acquisition = FrameAcquisition(hw, self.spec)
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
    


class FrameSizeCalibrationSpec(FrameAcquisitionSpec):
    def __init__(self, 
                 min_ampl_frac: str | float = 0.2, 
                 max_ampl_frac: str | float = 1.0, 
                 n_amplitudes: int = 5,
                 sacrificial_frames: int = 4, 
                 translation_fraction: float = 0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.ampl_frac_range = units.FloatRange(
            min=float(min_ampl_frac), 
            max=float(max_ampl_frac)
        )
        self.n_amplitudes = n_amplitudes
        self.sacrificial_frames = sacrificial_frames
        self.translation_fraction = translation_fraction


class FrameSizeCalibration(Acquisition):
    required_resources = (Digitizer, FastRasterScanner, SlowRasterScanner)
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/frame"
    Spec: Type[FrameSizeCalibrationSpec] = FrameSizeCalibrationSpec
    
    def __init__(self, hw, system_config, spec: FrameSizeCalibrationSpec):
        super().__init__(hw, system_config, spec)
        self.spec: FrameSizeCalibrationSpec # to refine type hints
        
        self._frame_acquisition = FrameAcquisition(self.hw, system_config, self.spec)
        self._frame_acquisition.add_subscriber(self)

        self.digitizer_profile = self._frame_acquisition.digitizer_profile
        self.runtime_info = self._frame_acquisition.runtime_info

        fast_axis = self.system_config.fast_raster_scanner['axis']
        if fast_axis == "x":
            self._fast_stage = self.hw.stage.x
        else:
            self._fast_stage = self.hw.stage.y
        self._original_position = self._fast_stage.position

        ampl = self.hw.fast_raster_scanner.angle_limits.range
        self._amplitudes = np.linspace(
            start=spec.ampl_frac_range.min * ampl,
            stop=spec.ampl_frac_range.max * ampl,
            num=spec.n_amplitudes
        )

    def run(self):
        optics = self.hw.laser_scanning_optics
        try:
            # Get 10x sacrificial start frames to allow warm up
            self._frame_acquisition.start()
            
            for ampl in self._amplitudes:
                # set amplitude
                self.hw.fast_raster_scanner.amplitude = units.Angle(ampl)

                # collect some sacrificial frames
                for _ in range(4 * self.spec.sacrificial_frames):
                    product = self.inbox.get()
                    if product is None: return
                    with product: 
                        pass

                # Gather measurement frame
                product = self.inbox.get()
                if product is None: return
                with product: 
                    self.publish(product)

                # Move 
                line_width = self.spec.fill_fraction \
                    * optics.scan_angle_to_object_position(ampl)
                self._fast_stage.move_to(self._fast_stage.position
                    + line_width * self.spec.translation_fraction
                )

                # Discard frames in motion
                for _ in range(self.spec.sacrificial_frames):
                    product = self.inbox.get()
                    if product is None: return
                    with product: 
                        pass
                
                # Gather measurement frame
                product = self.inbox.get()
                if product is None: return
                with product: 
                    self.publish(product)

                # Move back to original position
                self._fast_stage.move_to(units.Position(self._original_position))

        finally:
            self.publish(None)
            self._frame_acquisition.stop()


class FrameDistortionCalibrationSpec(FrameAcquisitionSpec):
    def __init__(self, 
                 translation: str | units.Position,
                 sacrificial_frames: int = 5, 
                 n_steps: int = 4,
                 **kwargs):
        super().__init__(**kwargs)

        self.sacrificial_frames = sacrificial_frames
        self.translation = units.Position(translation)
        self.n_steps = n_steps


class FrameDistortionCalibration(Acquisition):
    required_resources = (Digitizer, FastRasterScanner, SlowRasterScanner)
    SPEC_LOCATION = Path(user_config_dir("Dirigo")) / "acquisition/frame"
    Spec: Type[FrameDistortionCalibrationSpec] = FrameDistortionCalibrationSpec

    def __init__(self, hw, system_config, spec):
        super().__init__(hw, system_config, spec)
        self.spec: FrameDistortionCalibrationSpec # to refine type hints
        
        self._frame_acquisition = FrameAcquisition(self.hw, system_config, self.spec)
        self._frame_acquisition.add_subscriber(self)

        self.digitizer_profile = self._frame_acquisition.digitizer_profile
        self.runtime_info = self._frame_acquisition.runtime_info

        fast_axis = self.system_config.fast_raster_scanner['axis']
        if fast_axis == "x":
            self._fast_stage = self.hw.stage.x
        else:
            self._fast_stage = self.hw.stage.y
        self._original_position = self._fast_stage.position


    def run(self):
        try:
            self._frame_acquisition.start()

            # collect some sacrificial frames
            for _ in range(10 * self.spec.sacrificial_frames):
                product = self.inbox.get()
                if product is None: return
                with product: 
                    pass
            
            for _ in range(self.spec.n_steps):
                # Gather frame
                product = self.inbox.get()
                if product is None: return
                with product: 
                    self.publish(product)

                # Move 
                self._fast_stage.move_to(self._fast_stage.position + self.spec.translation)

                # Discard frames in motion
                for _ in range(self.spec.sacrificial_frames):
                    product = self.inbox.get()
                    if product is None: return
                    with product: 
                        pass

            # Move back to original position
            self._fast_stage.move_to(self._original_position)

        finally:
            self.publish(None)
            self._frame_acquisition.stop()