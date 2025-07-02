from pathlib import Path
import threading
import math
import time
from typing import Optional, Type
from functools import cached_property
from dataclasses import dataclass, asdict

from platformdirs import user_config_dir
import numpy as np

from dirigo.components import units, io
from dirigo.sw_interfaces.worker import EndOfStream, Product
from dirigo.hw_interfaces.hw_interface import NoBuffers
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionSpec, AcquisitionProduct
from dirigo.hw_interfaces.detector import DetectorSet, Detector
from dirigo.hw_interfaces.digitizer import Digitizer, DigitizerProfile
from dirigo.hw_interfaces.scanner import (
    FastRasterScanner, SlowRasterScanner, GalvoScanner, ResonantScanner,
    ObjectiveZScanner
)
from dirigo.hw_interfaces.camera import LineCamera
from dirigo.hw_interfaces.illuminator import Illuminator
from dirigo.hw_interfaces.encoder import MultiAxisLinearEncoder
from dirigo.hw_interfaces.stage import MultiAxisStage

from dirigo_e2v_line_camera.dirigo_e2v_line_camera import TriggerModes # TODO write Dirigo-specific enum for trigger modes


TWO_PI = 2 * math.pi 


# ---------- Runtime objects ----------
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


@dataclass
class CameraAcquisitionRuntimeInfo:
    camera_bit_depth: int
    frame_grabber_bytes_per_pixel: Optional[int] = None

    @classmethod
    def from_acquisition(cls, acq: "LineCameraAcquisition"):
        fg_bpp: int | None = (
            acq.hw.frame_grabber.bytes_per_pixel
            if hasattr(acq.hw, "frame_grabber") and acq.hw.frame_grabber is not None
            else None
        )

        return cls(
            camera_bit_depth=acq.hw.line_camera.bit_depth,
            frame_grabber_bytes_per_pixel=fg_bpp
        )

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            camera_bit_depth=int(d['camera_bit_depth']),
            frame_grabber_bytes_per_pixel=(
                int(d["frame_grabber_bytes_per_pixel"])
                if d.get("frame_grabber_bytes_per_pixel") is not None
                else None
            )
        )
    
    def to_dict(self) -> dict:
        """Make dictionary for serialization."""
        # keep it lean: drop keys whose value is still None
        obj = asdict(self)
        return {k: v for k, v in obj.items() if v is not None}


# ---------- 0-D acquisitions ----------
class SampleAcquisitionSpec(AcquisitionSpec):
    def __init__(
            self,
            record_length: int,
            digitizer_profile: str = "default",
            timestamps_enabled: bool = True,
            pre_trigger_samples: int = 0,
            trigger_delay_samples: int = 0,
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

        # Validate buffers per acquisition (-1 means infinite)
        if (buffers_per_acquisition == float('inf') 
            or buffers_per_acquisition in ["inf", "infinity"]
            or buffers_per_acquisition == -1):
            self.buffers_per_acquisition = -1
        elif isinstance(buffers_per_acquisition, int) and buffers_per_acquisition > 0:
            self.buffers_per_acquisition = buffers_per_acquisition
        else:
            raise ValueError(f"`buffers_per_acquisition` must be 'inf' or > 0.")

        self.buffers_allocated = buffers_allocated

    
class SampleAcquisition(Acquisition):
    """
    Fundamental Acquisition type for Digitizer. Acquires a number of digitizer 
    samples at some rate. Should be independent of any spatial semantics.
    """
    required_resources = [Digitizer,]
    spec_location = Path() # TODO
    Spec: Type[SampleAcquisitionSpec] = SampleAcquisitionSpec

    def __init__(self, hw, system_config, spec, 
                 thread_name: str = "Sample acquisition"):
        super().__init__(hw, system_config, spec, thread_name) # sets up thread, inbox, stores hw, checks resources
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
    
    @classmethod
    def get_specification(cls, spec_name = "default") -> SampleAcquisitionSpec:
        return super().get_specification(spec_name) # type: ignore


class LineCameraAcquisitionSpec(AcquisitionSpec):
    """Specification for a line scan camera acquisition.
    
    Note: not meant to convey spatial/spectral semantics. Use subclasses
    LineCameraLineAcquisition for one or more physical lines or 
    LineCameraSpectrumAcquisition for one or more spectral measurements
    """
    def __init__(self,
                 integration_time: units.Time | str, # e.g. "1 ms"
                 line_period: units.Time | str, # TODO allow either line_rate or line_period
                 pixels_per_line: int,
                 lines_per_buffer: int,
                 buffers_per_acquisition: int = 1, # -1 codes for unlimited
                 roi_start_pixel: int = 0,
                 **kwargs
                 ):
        super().__init__()

        self.integration_time = units.Time(integration_time)
        self.line_period = units.Time(line_period)
        if not isinstance(roi_start_pixel, int) or roi_start_pixel < 0:
            raise ValueError("ROI start pixel must be an integer > 0")
        self.roi_start_pixel = roi_start_pixel
        if not isinstance(pixels_per_line, int):
            raise ValueError("Pixel per line must be an integer")
        self.pixels_per_line = pixels_per_line
        if not isinstance(lines_per_buffer, int):
            raise ValueError("Lines per buffer must be an integer")
        self.lines_per_buffer = lines_per_buffer
        if not isinstance(buffers_per_acquisition, int):
            raise ValueError("Buffers per acquisition must be an integer")
        self.buffers_per_acquisition = buffers_per_acquisition


class LineCameraAcquisition(Acquisition):
    """
    Base acquisition class for a line-scan camera (i.e. linear array sensor)
    
    Use subclasses to provide spatial/spectral semantics.
    """
    required_resources = [LineCamera,]
    spec_location = io.config_path() / "acquisitions/line_camera"
    Spec = LineCameraAcquisitionSpec

    def __init__(self, hw, system_config, spec,
                 thread_name: str = "Line-scan camera acquisition"):
        super().__init__(hw, system_config, spec, thread_name) # sets up thread, inbox, stores hw, checks resources
        self.spec: LineCameraAcquisitionSpec

        # Set line camera properties (related to spec)
        self.configure_camera() # TODO, should it be configure frame grabber?
        self.hw.frame_grabber.prepare_buffers(nbuffers=4)

        # Load camera profile (device-specific)
        self.hw.line_camera.load_profile()

        self.runtime_info = CameraAcquisitionRuntimeInfo.from_acquisition(self)
        self.camera_profile = [] # TODO need to fix this

        self.active = threading.Event()  # to indicate data acquisition occuring

    def configure_camera(self, trigger_mode: str = "free run"):
        """Configure camera and framegrabber."""
        grabber = self.hw.frame_grabber
        cam = self.hw.line_camera

        cam.integration_time = self.spec.integration_time

        if trigger_mode == "free run":
            cam.trigger_mode = TriggerModes.FREE_RUN
            cam.line_period = self.spec.line_period
        elif trigger_mode == "external trigger":
            cam.trigger_mode = TriggerModes.EXTERNAL_TRIGGER
            # there's no definite line period for external triggering
        else:
            raise ValueError(f"Unsupported trigger mode: {trigger_mode}")
        # TODO add profile settings: gain, offsets, etc.

        # set ROI size based on Spec
        grabber.roi_left = self.spec.roi_start_pixel
        grabber.roi_width = self.spec.pixels_per_line
        self.hw.frame_grabber.lines_per_buffer = self.spec.lines_per_buffer

    @property
    def line_rate(self) -> units.Frequency:
        """Inverse line period (if provided)"""
        return units.Frequency(1 / self.spec.line_period)
    
    def _get_free_product(self) -> AcquisitionProduct:
        return super()._get_free_product() # type: ignore

    def run(self):
        # Set up acquisition buffer pool
        shape = self.hw.frame_grabber._buffers[0].buffer.shape # TODO add some sort of shape/dtype to API for framegrabber
        dtype = self.hw.frame_grabber._buffers[0].buffer.dtype
        self._init_product_pool(n=4, shape=shape, dtype=dtype) 

        self.hw.frame_grabber.start()
        self.active.set() # signals active recording

        try:
            acq_product = self._get_free_product()

            bpa = self.spec.buffers_per_acquisition
            while not self._stop_event.is_set():
                # for finite acquisition, stop when recorded set number of buffers
                if bpa != -1 and self.hw.frame_grabber.buffers_acquired >= bpa: # bpa != -1 codes for finite
                    break 

                try:               
                    self._get_buffer_data(acq_product=acq_product)
                    self._publish(acq_product)
                    print("Published a buffer")

                    acq_product = self._get_free_product() # Get a fresh buffer

                except NoBuffers:
                    time.sleep(0.001)
            
        finally:
            self.cleanup()

    def _get_buffer_data(self, acq_product):
        # Shadow this in subclasses to add metadata to acquisition product (ie positions, timestamps, etc)
        self.hw.frame_grabber.get_next_completed_buffer(acq_product)

    def cleanup(self):
        self._publish(None) # sentinel that shuts down downstream Workers
        self.hw.frame_grabber.stop()
        
        

# ---------- 1-D acquisitions ----------
class LineAcquisitionSpec(SampleAcquisitionSpec): 
    """Specification for a point-scanned line acquisition"""
    MAX_PIXEL_SIZE_ADJUSTMENT = 0.01
    def __init__(
        self,
        line_width: units.Position | str,
        pixel_size: units.Position | str,
        lines_per_buffer: int,
        bidirectional_scanning: bool = False,
        pixel_time: Optional[str] = None, # e.g. "1 μs"
        fill_fraction: float = 1.0,
        **kwargs
    ):
        super().__init__(record_length=0, # will be overwritten
                         **kwargs)
        self.bidirectional_scanning = bidirectional_scanning 
        if pixel_time:
            self.pixel_time = units.Time(pixel_time)
        else:
            self.pixel_time = None # None codes for non-constant pixel time (ie resonant scanning)
        self.line_width = units.Position(line_width)
        psize = units.Position(pixel_size)

        # Adjust pixel size such that line_width/pixel_size is an integer
        self.pixel_size = self.line_width / round(self.line_width / psize)
        if abs(self.pixel_size - psize) / psize > self.MAX_PIXEL_SIZE_ADJUSTMENT:
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


class LineAcquisition(SampleAcquisition):
    required_resources = [Digitizer, DetectorSet, FastRasterScanner]
    optional_resources = [MultiAxisStage, ObjectiveZScanner]
    spec_location = Path(user_config_dir("Dirigo")) / "acquisition/line"
    Spec: Type[LineAcquisitionSpec] = LineAcquisitionSpec
    
    def __init__(self, hw, system_config, spec,
                 thread_name: str = "Line acquisition"):
        """Initialize a line acquisition worker."""
        # Tip: since this method runs on main thread, limit to HW init tasks that will return fast, prepend slower tasks to run method
        super().__init__(hw, system_config, spec, thread_name) # sets up thread, inbox, stores hw, checks resources
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
                self.spec.line_width,
                axis="fast"
            )

            # Fast axis period should be multiple of digitizer sample resolution
            assert self.spec.pixel_time is not None
            T_exact = self.spec.pixel_time * self.spec.pixels_per_line / self.spec.fill_fraction
            dt = units.Time(
                digi.acquire.record_length_resolution / digi.sample_clock.rate
            )
            T_rounded = round(T_exact / dt) * dt
            scanner.frequency = 1 / T_rounded 
            scanner.waveform = "asymmetric triangle"
            scanner.duty_cycle = self.spec.fill_fraction # TODO set duty cycle to 50% if doing bidi
        
        elif isinstance(scanner, ResonantScanner):
            # for res scanner: fixed: frequency, waveform, duty cycle; 
            # adjustable: amplitude
            scanner.amplitude = optics.object_position_to_scan_angle(
                self.spec.extended_scan_width,
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
        self._init_product_pool(n=4, shape=shape, dtype=np.int16)

        # Start scanner & digitizer
        if isinstance(self.hw.fast_raster_scanner, ResonantScanner):
            self.hw.fast_raster_scanner.start()
            # pause for a little while to allow res scanner to reach steady state
            if hasattr(self.hw.fast_raster_scanner, 'response_time'):
                time.sleep(self.hw.fast_raster_scanner.response_time)
            digi.acquire.start() # This includes the buffer allocation
            self.active.set()

        elif isinstance(self.hw.fast_raster_scanner, GalvoScanner):
            if digi._mode == "analog": # type: ignore #TODO fix this
                self.hw.fast_raster_scanner.start(
                    input_sample_rate=digi.sample_clock.rate,
                    input_sample_clock_channel=digi.sample_clock.source,
                    pixels_per_period=self.spec.pixels_per_line,
                    periods_per_write=self.spec.records_per_buffer
                )
                digi.acquire.start()

            elif digi._mode == "edge counting": # type: ignore #TODO fix this
                digi.acquire.start()
                self.hw.fast_raster_scanner.start(
                    input_sample_rate=digi.sample_clock.rate,
                    input_sample_clock_channel=digi.sample_clock.source,
                    pixels_per_period=self.spec.pixels_per_line,
                    periods_per_write=self.spec.records_per_buffer
                )

        try:
            bpa = self.spec.buffers_per_acquisition
            while not self._stop_event.is_set():
                if bpa != -1 and digi.acquire.buffers_acquired >= bpa: # bpa=-1 codes for infinite
                    break
                #print("Buffers acquired", digi.acquire.buffers_acquired)
                acq_product = self._get_free_product()
                digi.acquire.get_next_completed_buffer(acq_product)

                if self.hw.stages or self.hw.objective_z_scanner:
                    t0 = time.perf_counter()
                    acq_product.positions = self.read_positions()
                    t1 = time.perf_counter()

                self._publish(acq_product)

                print(f"Acquired {digi.acquire.buffers_acquired} {"" if bpa==-1 else f"of {bpa}"} "
                      f"Reading stage positions took: {1000*(float(t1)-t0):.3f} ms")
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
        self._publish(None)

    def read_positions(self):
        """Subclasses can override this method to provide position readout from
        stages or linear position encoders."""
        positions = []
        try:
            positions.append(self.hw.stages.x.position)
            positions.append(self.hw.stages.y.position)
        except (KeyError, AttributeError): pass

        try:
            positions.append(self.hw.objective_z_scanner.position)
        except (KeyError, AttributeError): pass
        
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
    
    @classmethod
    def get_specification(cls, spec_name = "default") -> LineAcquisitionSpec:
        return super().get_specification(spec_name) # type: ignore


class LineCameraLineAcquisitionSpec(LineCameraAcquisitionSpec):
    def __init__(self, 
                 line_width: units.Position | str,
                 pixel_size: units.Position | str,  
                 integration_time: units.Time | str,
                 line_period: units.Time | str,
                 lines_per_buffer: int,               
                 **kwargs):
        """
        Specify a line scan camera 1-D line acquisition.
        Note: `line_width` will be adjusted such that it is divisible by `pixel_size`
        """
        self.pixel_size = units.Position(pixel_size)

        lw = units.Position(line_width)
        pixels_per_line = round(lw / self.pixel_size)
        self.line_width = units.Position(pixels_per_line * self.pixel_size)

        super().__init__(
            integration_time=integration_time,
            line_period=line_period,
            pixels_per_line=pixels_per_line,
            lines_per_buffer=lines_per_buffer,
            **kwargs
        )


class LineCameraLineAcquisition(LineCameraAcquisition):
    """
    """
    required_resources = [LineCamera, Illuminator, MultiAxisLinearEncoder] # encoder may be optional
    spec_location = io.config_path() / "acquisition/line_camera_line"
    Spec = LineCameraLineAcquisitionSpec

    def __init__(self, hw, system_config, spec,
                 thread_name: str = "Line scan camera line acquisition"):
        super().__init__(hw, system_config, spec, thread_name) # sets up thread, inbox, stores hw, checks resources
        self.spec: LineCameraLineAcquisitionSpec

    def configure_camera(self, trigger_mode: str = "external trigger"):
        """Configure camera and framegrabber."""
        super().configure_camera(trigger_mode) # sets integration time, trigger mode, lines per buffer, etc.

        # set ROI size based on line spec, assumed ROI centered in array
        obj_pixel_size = self.hw.line_camera.pixel_size / self.hw.camera_optics.magnification # TODO, switch if object pixel size already set
        roi_width = round(self.spec.line_width / obj_pixel_size)
        self.hw.frame_grabber.roi_width = roi_width
        self.hw.frame_grabber.roi_left = (self.hw.frame_grabber.pixels_width - roi_width) // 2

    @property
    def axis(self) -> str:
        """Axis of the array of pixels"""
        return self.hw.line_camera.axis

    # run() - unchanged from base class, but shadow _get_buffer_data to add functionality

    def _get_buffer_data(self, acq_product: AcquisitionProduct):
        self.hw.frame_grabber.get_next_completed_buffer(acq_product)
        acq_product.positions = self._read_positions()
    
    def _read_positions(self):
        """ Read positions, if stage/z scanner available. """
        positions = []
        try:
            positions.append(self.hw.stages.x.position)
            positions.append(self.hw.stages.y.position)
        except (KeyError, AttributeError): pass

        try:
            positions.append(self.hw.objective_z_scanner.position)
        except (KeyError, AttributeError): pass
        
        return tuple(positions) if len(positions) else None

    def cleanup(self):
        super().cleanup()



# ---------- 2-D acquisitions ----------
# (for 2-D line camera acquisitions, see separate package "dirigo-strip-acquisition")
class FrameAcquisitionSpec(LineAcquisitionSpec):
    """Specifications for frame series acquisition"""
    MAX_PIXEL_HEIGHT_ADJUSTMENT = 0.01
    def __init__(self, 
                 frame_height: str | units.Position, 
                 flyback_periods: int, 
                 pixel_height: str = "", # leave empty to set pxiel height = width
                 **kwargs):
        # set some parameters early so line_per_frame property can work
        self.pixel_size = kwargs["pixel_size"]
        self.bidirectional_scanning = kwargs["bidirectional_scanning"]
        
        self.frame_height = units.Position(frame_height)
        if pixel_height == "":
            p_height = units.Position(self.pixel_size)
        else:
            p_height = units.Position(pixel_height)

        self.pixel_height = self.frame_height / round(self.frame_height / p_height)
        if abs(self.pixel_height - p_height) / p_height > self.MAX_PIXEL_HEIGHT_ADJUSTMENT:
            raise ValueError(
                f"To maintain integer number of pixels in frame height, required adjusting " \
                f"the pixel height by more than pre-specified limit: {100*self.MAX_PIXEL_HEIGHT_ADJUSTMENT}%"
            )

        self.flyback_periods = flyback_periods
        
        super().__init__(lines_per_buffer=self.lines_per_frame, **kwargs)

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
    required_resources = [Digitizer, DetectorSet, FastRasterScanner, SlowRasterScanner]
    optional_resources = [MultiAxisStage, ObjectiveZScanner]
    spec_location = Path(user_config_dir("Dirigo")) / "acquisition/frame"
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

    @classmethod
    def get_specification(cls, spec_name = "default") -> FrameAcquisitionSpec:
        return super().get_specification(spec_name) # type: ignore
        

# ---------- 3-D acquisitions ----------
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
    required_resources = [Digitizer, FastRasterScanner, SlowRasterScanner, ObjectiveZScanner]
    optional_resources = [MultiAxisStage,]
    spec_location = Path(user_config_dir("Dirigo")) / "acquisition/stack"
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
        self.spec.buffers_per_acquisition = -1 # codes for infinite buffers
        self._frame_acquisition = FrameAcquisition(hw, system_config, self.spec)
        self._frame_acquisition.add_subscriber(self)

        # Initialize object scanner
        self.hw.objective_z_scanner.max_velocity = units.Velocity("300 um/s")
        self.hw.objective_z_scanner.acceleration = units.Acceleration("1 mm/s^2")

    def _receive_product(self, 
                         block: bool = True, 
                         timeout: float | None = None) -> AcquisitionProduct:
        return super()._receive_product(block, timeout) # type: ignore

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
        z_scanner = self.hw.objective_z_scanner
        self.hw.objective_z_scanner.move_to(self._depths[0])
        
        # spin until reach start position
        time.sleep(units.Time('10 ms'))
        while self.hw.objective_z_scanner.moving:
            time.sleep(units.Time('10 ms'))

        try:
            # Start child FrameAcquisition
            self._frame_acquisition.start()

            # Get sacrificial frames (don't pass them along)
            for _ in range(self.spec._sacrificial_frames_per_step):
                with self._receive_product(): pass

            for i in range(1, self.spec.depths_per_acquisition+1):
                for _ in range(self.spec._saved_frames_per_step):
                    # Wait for frame data, and pass along
                    with self._receive_product() as product:
                        self._publish(product)

                    # TOOD, blank laser beam for step time (sacrificial frames)

                if i < self.spec.depths_per_acquisition:
                    # Move Z scanner to next depth
                    z_scanner.move_to(self._depths[i])

                    # Wait for sacrificial frames
                    for _ in range(self.spec._sacrificial_frames_per_step):
                        with self._receive_product(): pass
        
        finally:
            self._frame_acquisition.stop()
            self._publish(None) # publish the sentinel
            z_scanner.move_to(self._depths[0]) # move back to start

    @property
    def digitizer_profile(self) -> DigitizerProfile:
        return self.hw.digitizer.profile
    
    @property
    def runtime_info(self) -> LineAcquisitionRuntimeInfo:
        return self._frame_acquisition.runtime_info

            