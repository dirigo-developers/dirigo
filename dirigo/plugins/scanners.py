from functools import cached_property
import threading
import time

import numpy as np

import nidaqmx
import nidaqmx.system
import nidaqmx.errors
from nidaqmx.stream_writers import AnalogMultiChannelWriter
from nidaqmx.constants import AcquisitionType, RegenerationMode, LineGrouping

from dirigo.components import units
from dirigo.hw_interfaces.scanner import (
    GalvoScanner, ResonantScanner, PolygonScanner,
    FastRasterScanner, SlowRasterScanner,
)
from dirigo.hw_interfaces import digitizer



def get_device(device_name: str = "Dev1") -> nidaqmx.system.Device:
    """Returns handle to the NIDAQ system device object."""
    return nidaqmx.system.System.local().devices[device_name]


def validate_ni_channel(channel_name: str) -> str:
    """
    Confirm if a given channel or terminal exists on a device.
    
    Parameters:
    - channel_name: str, the full channel name to check (e.g., 'Dev1/ai0', 
      'Dev1/ao0', '/Dev1/PFI0').
      Note: Terminals (e.g., PFI0) must include a leading '/' (e.g., '/Dev1/PFI0'). 

    Raises:
    - ValueError: If the channel name format is invalid or the channel/terminal
      does not exist.
    - KeyError: If the specified device does not exist.

    Returns:
    - None: If the channel exists (raises no exception).
    """
    # Ensure channel_name includes both device and channel/terminal parts
    if '/' not in channel_name or channel_name.count('/') > 3:
        raise ValueError(
            f"Invalid channel name format, {channel_name}. "
            f"Valid formats: [device name]/[channel name] or /[device name]/[terminal name]. "
            f"Examples: 'Dev1/ao0', '/Dev1/PFI4', '/Dev1/ao/SampleClock'"
        )

    # Split device name from the rest of the channel name
    parts = channel_name.split('/')
    if channel_name.startswith('/'):
        # Handle terminal format: '/Dev1/PFI0'
        device_name = parts[1]
    else:
        # Handle channel format: 'Dev1/ai0'
        device_name = parts[0]

    try:
        # Get the device
        device = get_device(device_name)

    except KeyError:
        raise ValueError(f"Device {device_name} not found in the system.")

    # Collect all channel and terminal names from the device
    all_channels = []
    all_channels.extend(chan.name for chan in device.ai_physical_chans)
    all_channels.extend(chan.name for chan in device.ao_physical_chans)
    all_channels.extend(chan.name for chan in device.di_lines)
    all_channels.extend(chan.name for chan in device.do_lines)
    all_channels.extend(device.terminals)

    # Check if the specified channel exists
    if channel_name not in all_channels:
        raise ValueError(
            f"Channel/terminal, {channel_name} not found on device, {device_name}"
        )
    # If no exception is raised, the channel is valid
    return channel_name


class CounterRegistry:
    _available_counters: list[str] = []
    _in_use: set[str] = set()
    _initialized: bool = False
    _device_name: str | None = None

    @classmethod
    def initialize(cls, device_name="Dev1"):
        device = nidaqmx.system.Device(device_name)
        cls._available_counters = [chan.name for chan in device.ci_physical_chans]
        cls._in_use.clear()
        cls._initialized = True
        cls._device_name = device_name

    @classmethod
    def allocate_counter(cls, device_name="Dev1") -> str:
        """
        Returns a free counter from the registry, marking it in use.
        Lazily initializes the registry if not already done.
        """
        # Lazy initialization
        if not cls._initialized:
            cls.initialize(device_name)

        # If someone tries to allocate from a different device
        # than we previously initialized, decide how you want to handle that:
        if device_name != cls._device_name:
            # Possibly re‐initialize for the new device?
            cls.initialize(device_name)

        for ctr in cls._available_counters:
            if ctr not in cls._in_use:
                cls._in_use.add(ctr)
                return ctr

        raise RuntimeError("No free counters available!")

    @classmethod
    def free_counter(cls, counter_name: str):
        """
        Marks a previously-allocated counter as free again.
        """
        cls._in_use.discard(counter_name)


def get_min_ao_rate(device: nidaqmx.system.Device) -> units.SampleRate:
    return units.SampleRate(device.ao_min_rate)


def get_max_ao_rate(device: nidaqmx.system.Device) -> units.SampleRate:
    return units.SampleRate(device.ao_max_rate)



class ResonantScannerViaNI(ResonantScanner, FastRasterScanner):
    """
    Resonant scanner operated via an NIDAQ MFIO card.
    """
    def __init__(self, amplitude_control_channel: str, analog_control_range: dict, **kwargs):
        super().__init__(**kwargs)

        # Validated and set ampltidue control analog channel
        validate_ni_channel(amplitude_control_channel)
        self._amplitude_control_channel = amplitude_control_channel

        # Validate the analog control range
        if not isinstance(analog_control_range, dict):
            raise ValueError(
                "analog_control_range must be a dictionary."
            )
        missing_keys = {'min', 'max'} - analog_control_range.keys()
        if missing_keys:
            raise ValueError(
                f"analog_control_range must have 'min' and 'max' keys."
            )
        self._analog_control_range = units.VoltageRange(**analog_control_range)

        # Default: set amplitude to 0 upon startup
        self.amplitude = units.Angle(0.0)

    @property
    def amplitude(self) -> units.Angle:
        """Get the peak-to-peak amplitude, in optical angle."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_ampl: units.Angle):
        """Set the peak-to-peak amplitude."""
        if not isinstance(new_ampl, units.Angle):
            raise ValueError(
                f"`amplitude` must be set with Angle object. Got {type(new_ampl)}"
            )

        # Validate that the value is within the acceptable range
        if not self.angle_limits.within_range(units.Angle(new_ampl/2)):
            raise ValueError(
                f"Value for 'amplitude' outside settable range "
                f"{self.angle_limits.min} to {self.angle_limits.max}. "
                f"Got: {new_ampl}"
            )
        
        # Calculate the required analog voltage value, validate within range
        ampl_fraction = new_ampl / self.angle_limits.range
        analog_value =  ampl_fraction * self.analog_control_range.max
        if not self.analog_control_range.within_range(units.Voltage(analog_value)):
            raise ValueError(
                f"Voltage to achieve amplitude={new_ampl} is outside range. "
                f"Attempted to set {analog_value} V. "
                f"Range: {self.analog_control_range.min} to {self.analog_control_range.max} V"
            )
        
        # Set the analog value using nidaqmx
        try:
            with nidaqmx.Task() as task: 
                task.ao_channels.add_ao_voltage_chan(self._amplitude_control_channel)
                task.write(analog_value, auto_start=True)
        except nidaqmx.DaqError as e:
            raise RuntimeError(f"Failed to set analog output: {e}") from e

        self._amplitude = new_ampl

    @property
    def analog_control_range(self) -> units.VoltageRange:
        """Returns an object describing the analog control range."""
        return self._analog_control_range
    
    def start(self):
        pass # TODO, add some sort of enable/disbale channel

    def stop(self):
        pass # TODO, add some sort of enable/disbale channel


class PolygonScannerViaNI(PolygonScanner, FastRasterScanner):
    pass # TODO, write this


class GalvoScannerViaNI(GalvoScanner):

    def __init__(self, 
                 analog_control_channel: str, # e.g. 'Dev1/ao1'
                 analog_control_range: dict, # e.g. {'min': '-10 V', 'max': '10 V'}
                 **kwargs):
        
        super().__init__(**kwargs) # Sets axis & scan angle range
        self._device = get_device()

        self._analog_control_channel = validate_ni_channel(analog_control_channel)

        if not isinstance(analog_control_range, dict):
            raise ValueError("`analog_control_range` must be a dictionary.")
        missing_keys = {'min', 'max'} - analog_control_range.keys()
        if missing_keys:
            raise ValueError(
                f"`analog_control_range` must be a dictionary with 'min' and 'max' keys."
            )
        self._analog_control_range = units.VoltageRange(**analog_control_range)

        self._ao_task: nidaqmx.Task | None = None
        self._active = False

    def park(self):
        """Positions the scanner at the angle limit minimum."""
        analog_value = units.Voltage(self.angle_limits.min * self._volts_per_radian)
        try:
            with nidaqmx.Task() as task: 
                task.ao_channels.add_ao_voltage_chan(self._analog_control_channel)
                task.write(analog_value, auto_start=True)
        except nidaqmx.DaqError as e:
            raise RuntimeError(f"Failed to park scanner: {e}") from e
        
    def center(self):
        """Positions the scanner at zero angle."""
        analog_value = 0
        try:
            with nidaqmx.Task() as task: 
                task.ao_channels.add_ao_voltage_chan(self._analog_control_channel)
                task.write(analog_value, auto_start=True)
        except nidaqmx.DaqError as e:
            raise RuntimeError(f"Failed to center scanner: {e}") from e
    
    @cached_property
    def _volts_per_radian(self) -> float:
        """Scaling factor between analog control voltge and optical scan angle."""
        return float(self._analog_control_range.range / self.angle_limits.range)
    
    def generate_waveform(self, 
                          sample_rate: units.SampleRate,
                          rearm_time: units.Time = units.Time(0)): # TODO, njit this? parameters: waveform, amplitude, duty_cycle, waveform_length, samples_per_period
        """Returns are waveform vector in units of radian."""
        offset = self.offset
        # TODO, adjustable phase?

        exact_samples_per_period = sample_rate / self.frequency
        waveform_length = round(exact_samples_per_period - rearm_time * sample_rate)

        if self.waveform == 'sinusoid':
            t = np.linspace(start=0, stop=2*np.pi, num=waveform_length)

            waveform = (self.amplitude/2) * np.cos(t) + offset # amplitude is pk-pk hence the 1/2
        
        elif self.waveform in {'triangle', 'asymmetric triangle', 'sawtooth'}:
            # TODO clean this up
            num_up = round(exact_samples_per_period * self.duty_cycle) 
            #num_down = waveform_length - num_up
            A = self.amplitude
            F = num_up / waveform_length # fill fraction (aka duty cycle, %scan with usable data)
            F2 = (F + 1) / 2
            k = float(A) / (F*(F2**2 - F)) # acceleration constant

            # minus 0.5/waveform_length helps with some round-off issues when waveform length becomes large
            t0 = np.arange(0, F - 0.5/waveform_length, 1/waveform_length) # linear part
            t1 = np.arange(F, F2 - 0.5/waveform_length, 1/waveform_length) # flyback (negative acceleration)
            t2 = np.arange(t1[-1] + 1/waveform_length, 1 - 0.5/waveform_length, 1/waveform_length) # flyback (positive acceleration)

            parts = (                                       # Smooth raster:
                A/F * t0,                                   # linear part
                -k*t1**2/2 - k*F**2/2 + k*F*t1 + A/F*t1,    # flyback I  (-accel)
                k*t2**2/2 + A*t2/F - k*t2 + k/2 - A/F       # flyback II (+accel)
            )
            waveform = np.concatenate(parts, axis=0) - A/2 # minus A/2 to center around 0 degrees (rather than 0 to +A)

        return waveform * self._volts_per_radian
    
    def generate_clock(self, sample_rate: units.SampleRate): # TODO, njit this? parameters: waveform, amplitude, duty_cycle, waveform_length, samples_per_period
        """Returns a digital signal representing the line or frame clock."""

        # TODO, adjustable phase?

        exact_samples_per_period = sample_rate / self.frequency
        rearm_time = 0
        waveform_length = round(exact_samples_per_period - rearm_time * sample_rate)

        num_up = int(exact_samples_per_period * self.duty_cycle) + 1

        clock = np.zeros((waveform_length,), dtype=np.bool)
        clock[:num_up] = True

        return clock
        
    # Intercept some setters so they do not change value of parameters during scan
    @GalvoScanner.frequency.setter
    def frequency(self, new_frequency: units.Frequency):
        if self._active:
            raise RuntimeError("Frequency is not adjustable while scanner is active.")
        else:
            GalvoScanner.frequency.__set__(self, new_frequency)

    @GalvoScanner.waveform.setter
    def waveform(self, new_waveform: str):
        if self._active:
            raise RuntimeError("Waveform type is not adjustable while scanner is active.")
        else:
            GalvoScanner.waveform.__set__(self, new_waveform)

    @GalvoScanner.duty_cycle.setter
    def duty_cycle(self, new_duty_cycle: float):
        if self._active:
            raise RuntimeError("Duty cycle is not adjustable while scanner is active.")
        else:
            GalvoScanner.duty_cycle.__set__(self, new_duty_cycle)

    # Are adjustable: offset, amplitude
        

class GalvoWaveformWriter(threading.Thread):
    """
    Manages writing analog waveform for single fast galvo, or both waveforms in 
    galvo-galvo configurations.
    """

    def __init__(self, 
                 fast_scanner: 'FastGalvoScannerViaNI' = None, 
                 slow_scanner: 'SlowGalvoScannerViaNI' = None,
                 rearm_time: units.Time = units.Time(0)
                 ):
        super().__init__()
        self._stop_event = threading.Event()

        self._fast_scanner = fast_scanner
        self._slow_scanner = slow_scanner

        if not isinstance(rearm_time, units.Time):
            raise ValueError("Rearm time must be set with a units.Time object")
        if rearm_time < 0:
            raise ValueError("Rearm time must be greater than or equal to 0 seconds")
        self._rearm_time = rearm_time

        if self._fast_scanner:
            if not isinstance(self._fast_scanner, FastGalvoScannerViaNI):
                raise ValueError("Wrong type of fast scanner passed to "
                                 "GalvoWaveformWriter")
            
            # If a fast scanner exists, use its AO task and its sample rate 
            self._sample_rate = units.SampleRate(self._fast_scanner._sample_rate)
            self._ao_writer = AnalogMultiChannelWriter(self._fast_scanner._ao_task.out_stream)

        else:
            # If no fast scanner exist, then use the slow scanner and its sample rate
            self._sample_rate = units.SampleRate(self._slow_scanner._ao_task.timing.samp_clk_rate)
            self._ao_writer = AnalogMultiChannelWriter(self._slow_scanner._ao_task.out_stream)

        # Write once before starting the run loop
        waveforms = self.generate_waveforms()
        self._ao_writer.write_many_sample(waveforms)

    def _work(self):
        while not self._stop_event.is_set():
            try:
                waveforms = self.generate_waveforms()
                self._ao_writer.write_many_sample(waveforms)

                print("Wrote waveform with shape", waveforms.shape)
                # TODO shorter timeout and time out exception

            except nidaqmx.errors.DaqWriteError as e:
                # the task has not started yet, wait a while
                print("Skipped writing waveform")
                #print(e)
                time.sleep(units.Time('2 ms'))

    def stop(self):
        self._stop_event.set()

    def generate_waveforms(self): 
        fast_scanner = self._fast_scanner
        slow_scanner = self._slow_scanner

        if fast_scanner:
            fast_waveform = fast_scanner.generate_waveform(self._sample_rate)
            fast_waveform = np.tile(fast_waveform, reps=fast_scanner._periods_per_write)
            
            if slow_scanner:
                slow_waveform = slow_scanner.generate_waveform(self._sample_rate)
                return np.vstack((fast_waveform, slow_waveform))
            else:
                return np.vstack((fast_waveform,))
        
        else:
            slow_waveform = slow_scanner.generate_waveform(
                sample_rate=self._sample_rate,
                rearm_time=self._rearm_time # Rearm is only applicable to external line clock-synced acquisitions (e.g. resonant scanner)
            )
            return np.vstack((slow_waveform,))

        
    @property
    def _samples_per_period(self) -> float:
        """
        Exact number (e.g. fractional) of output sample clock periods per fast 
        scanner period (if available), or slow scanner (if not).
        """
        if self._fast_scanner:
            return self._sample_rate / self._fast_scanner.frequency
        else:
            return self._sample_rate / self._slow_scanner.frequency
    
    @property
    def _waveform_length(self) -> int:
        """
        Number of samples in output waveform. This is reduced by the rearm time
        and rounded to an integer. `_waveform_length` & `samples_per_period` will
        deviate when there is a non-zero rearm. For continuous AO generation,
        the rearm time should be 0, but for retriggered finite AO generation,
        it must be considered.
        """
        return round(self._samples_per_period - self._rearm_time * self._sample_rate)



class FastGalvoScannerViaNI(GalvoScannerViaNI, FastRasterScanner):
    TARGET_SAMPLE_RATE = units.SampleRate("200 kS/s")
    
    def __init__(self, 
                 line_clock_channel: str = None, 
                 **kwargs):
        
        super().__init__(**kwargs) # analog control channel and voltage limits

        self._line_clock_channel = validate_ni_channel(line_clock_channel)

        self._slow_scanner: SlowGalvoScannerViaNI = None

    def start(self, 
              input_mode: digitizer.InputMode,
              input_sample_rate: units.SampleRate, 
              sample_clock_source: digitizer.SampleClockSource,
              pixels_per_period: int, # e.g. pixels per line # TODO, samples per period??
              periods_per_write: int, # e.g. lines per frame
              adjustable = False # allows changing amplitude & offset mid-acquisition, but may be taxing for very high frame rates
              ):
        self._active = True

        try:
            # Validate sample rate
            if not isinstance(input_sample_rate, units.SampleRate):
                raise ValueError("Input sample rate must be set with SampleRate object")
            
            # Validate pixels per period
            if (not isinstance(pixels_per_period, int)) or (pixels_per_period < 1):
                raise ValueError("Pixels per period must be a positive integer")
            self._pixels_per_period = pixels_per_period

            # Validate periods per write
            if (not isinstance(periods_per_write, int)) or (periods_per_write < 1):
                raise ValueError("Periods per write must be a positive integer")
            self._periods_per_write = periods_per_write

            # Set up AO (galvo command waveform) and DO (line clock) tasks
            self._ao_task = nidaqmx.Task("Galvo waveforms") # for analog galvo command waveforms
            self._ao_task.ao_channels.add_ao_voltage_chan(
                physical_channel=self._analog_control_channel
            )

            self._do_task = nidaqmx.Task("Line and frame clocks") # for digital clocks
            self._do_task.do_channels.add_do_chan(
                lines=self._line_clock_channel,
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # If there is a slow scanner, then add its analog channel and frame clock line
            if self._slow_scanner:
                # Add more AO & DO channels
                self._ao_task.ao_channels.add_ao_voltage_chan(
                    physical_channel=self._slow_scanner._analog_control_channel
                )
                self._do_task.do_channels.add_do_chan(
                    lines=self._slow_scanner._frame_clock_channel,
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )
            
            output_sample_rate = input_sample_rate # assume these are 1:1, but we may modify below
            self._co_task = None # used if we need to divide down the input sample clock
            max_ao_rate = get_max_ao_rate(self._device)

            # To visualize: write out Analog vs Photon Counting & Internal vs External in a 2x2 chart
            if input_mode == digitizer.InputMode.ANALOG:
                if sample_clock_source == digitizer.SampleClockSource.INTERNAL:
                    ao_clock_source = "/" + self._device.name + "/ai/SampleClock"
                else:
                    raise NotImplementedError("External sample clock not yet implemented")
                
                # Add frequency divider counter if input sample rate is too high for AO
                if input_sample_rate > max_ao_rate:
                    # Counter Out task to divide input sample clock frequency
                    div_factor = 2**int(np.log2(input_sample_rate / max_ao_rate))
                    if div_factor > 32: 
                        raise RuntimeError("AI:AO clock frequency ratio cannot exceed 32 (record resolution)")
                        # up to 32 (digitizer record resolution) guarantees divisible AO samples per frame
                    high_ticks = div_factor // 2

                    self._co_task = nidaqmx.Task("AI clock frequency divided")
                    freq_div_ch = self._co_task.co_channels.add_co_pulse_chan_ticks(
                        counter         = CounterRegistry.allocate_counter(),
                        source_terminal = ao_clock_source,
                        high_ticks      = high_ticks,
                        low_ticks       = div_factor - high_ticks
                    )
                    self._co_task.timing.cfg_implicit_timing(
                        sample_mode     = AcquisitionType.CONTINUOUS,
                        samps_per_chan  = 20 # TODO, why 20?
                    )

                    ao_clock_source = freq_div_ch.co_pulse_term
                    output_sample_rate = input_sample_rate / div_factor
                
            elif input_mode == digitizer.InputMode.EDGE_COUNTING:
                if sample_clock_source == digitizer.SampleClockSource.INTERNAL:
                    ao_clock_source = None # denotes 'use internal clock'
                    if input_sample_rate > max_ao_rate:
                        raise RuntimeError("Edge counting sample rate set too high")
                        # one could also set up some sort of divider
                else:
                    raise NotImplementedError("External sample clock not yet implemented")
            
            # Set timing
            self._sample_rate = output_sample_rate
            self._ao_task.timing.cfg_samp_clk_timing(
                rate            = self._sample_rate,
                source          = ao_clock_source, # see above about frequency divider counter
                sample_mode     = AcquisitionType.CONTINUOUS,
                samps_per_chan  = self._periods_per_write * self._pixels_per_period # TODO should be samples not pixels per period
            )
            if adjustable:
                # Requires constantly supplying new waveform data, no regeneration
                self._ao_task.out_stream.regen_mode = \
                    RegenerationMode.DONT_ALLOW_REGENERATION

            self._do_task.timing.cfg_samp_clk_timing(
                rate            = self._sample_rate,
                source          = ao_clock_source, # see above about frequency divider counter
                sample_mode     = AcquisitionType.CONTINUOUS,
                samps_per_chan  = self._periods_per_write * self._pixels_per_period 
            )
            
            # Set up the waveform writer worker
            self._writer = GalvoWaveformWriter(
                fast_scanner = self, 
                slow_scanner = self._slow_scanner
            ) 
            if adjustable: self._writer.start()
                
            # Write clocks
            line_clock = np.tile(
                self.generate_clock(self._sample_rate),
                reps=self._periods_per_write
            )
            if self._slow_scanner:
                frame_clock = self._slow_scanner.generate_clock(self._sample_rate)
                clocks = np.vstack((line_clock, frame_clock))
            else:
                clocks = np.vstack((line_clock,))
            self._do_task.write(clocks)

            # Start tasks 
            if self._co_task: self._co_task.start()
            self._do_task.start() 
            self._ao_task.start()

        except Exception:
            self._active = False # actually not active
            raise # re-raise the ValueError so it propagates
        
    def pair_slow_galvo(self, slow_scanner: 'SlowGalvoScannerViaNI'):
        if isinstance(slow_scanner, SlowGalvoScannerViaNI):
            self._slow_scanner = slow_scanner
        else:
            raise ValueError("Must pass instance of SlowGalvoScannerViaNI")

    def stop(self):
        if self._writer.is_alive():
            self._writer.stop()
            self._writer.join() # Wait until done

        self._do_task.stop()
        self._ao_task.stop()
        
        self._do_task.close()
        self._ao_task.close()

        if self._co_task:
            self._co_task.stop()
            CounterRegistry.free_counter(self._co_task.co_channels[0].physical_channel.name)
            self._co_task.close()

        self._active = False
        

class SlowGalvoScannerViaNI(GalvoScannerViaNI, SlowRasterScanner):
    def __init__(self,
                 fast_scanner: FastRasterScanner, 
                 external_line_clock_channel: str = None, # to sync to external line clock (e.g. a resonant scanner)
                 frame_clock_channel: str = None, 
                 ao_sample_rate: str = "200 kS/s", # ignored if using galvo-galvo config
                 **kwargs):
        
        super().__init__(**kwargs) # analog control channel and voltage limits
        
        if isinstance(fast_scanner, FastGalvoScannerViaNI):
            self._external_line_clock_channel = None
            self._ao_sample_rate = None # for Galvo-Galvo scanning, AO rate = pixel rate
            self._fast_scanner = fast_scanner
            # pass itself as reference to the fast scanner--fast scanner will know to manage AO/DO tasks
            self._fast_scanner.pair_slow_galvo(self)
        
        elif isinstance(fast_scanner, (ResonantScannerViaNI, PolygonScannerViaNI)):
            self._external_line_clock_channel = validate_ni_channel(external_line_clock_channel)

            ao_sample_rate = units.SampleRate(ao_sample_rate)
            device = get_device(self._analog_control_channel.split('/')[0])
            low_sr = get_min_ao_rate(device)
            high_sr = get_max_ao_rate(device)
            if not low_sr < ao_sample_rate < high_sr:
                raise ValueError(f"AO sample rate outside required bounds: "
                                 f"low: {low_sr}, high {high_sr}, got {ao_sample_rate}")
            self._ao_sample_rate = ao_sample_rate

            self._fast_scanner = fast_scanner

        else:
            raise ValueError("Unsupported fast scanner type for SlowGalvoScannerViaNI")
        
        self._frame_clock_channel = validate_ni_channel(frame_clock_channel)
        
    def start(self,
              periods_per_frame: int = None,
              adjustable = False): # allows changing amplitude & offset mid-acquisition, but may be taxing for very high frame rates
        
        self._active = True

        try:
            # If external line clock specified, the fast axis is free-running 
            # (doesn't require explicit analog control signal, generate only the slow axis waveform)
            if self._external_line_clock_channel:
                if not isinstance(periods_per_frame, int) or periods_per_frame < 1:
                    raise ValueError("Periods per frame must be a positive integer")
                
                n_periods = self._fast_scanner.frequency / self.frequency
                rearm_periods = self._fast_scanner.frequency_error * n_periods
                rearm_time = units.Time(rearm_periods / self._fast_scanner.frequency) 

                self._fclock_task = nidaqmx.Task("Frame clock")

                high_ticks = round(self.duty_cycle * periods_per_frame)
                fclk_ch = self._fclock_task.co_channels.add_co_pulse_chan_ticks(
                    counter=CounterRegistry.allocate_counter(),
                    source_terminal=self._external_line_clock_channel,
                    low_ticks=periods_per_frame - high_ticks, 
                    high_ticks=high_ticks
                )

                fclk_ch.co_pulse_term = self._frame_clock_channel

                self._fclock_task.timing.cfg_implicit_timing(
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=periods_per_frame
                )

                self._ao_task = nidaqmx.Task("Slow galvo waveform")

                self._ao_task.ao_channels.add_ao_voltage_chan(
                    physical_channel=self._analog_control_channel
                )
                
                self._ao_task.timing.cfg_samp_clk_timing(
                    rate=self._ao_sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=self.generate_waveform(self._ao_sample_rate, rearm_time).shape[0]
                )

                self._ao_task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source=self._frame_clock_channel
                )
                self._ao_task.triggers.start_trigger.retriggerable = True

                if adjustable:
                    # To adjust amplitude and offset mid-acquisition
                    self._ao_task.out_stream.regen_mode = \
                        RegenerationMode.DONT_ALLOW_REGENERATION
                    
                # Set up the waveform writer worker
                self._writer = GalvoWaveformWriter(
                    slow_scanner=self,
                    rearm_time=rearm_time
                ) 
                if adjustable:
                    self._writer.start()

                self._ao_task.start()
                self._fclock_task.start()
                
        except:
            # start failed, so revert the _active flag, propagate the error up with 'raise'
            self._active = False
            raise

    def stop(self):
        self._active = False

        if self._external_line_clock_channel:
            self._fclock_task.stop()
            CounterRegistry.free_counter(self._fclock_task.channel_names[0])
            self._fclock_task.close()

            self._ao_task.stop()
            self._ao_task.close()

