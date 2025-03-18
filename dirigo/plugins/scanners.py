import math
from functools import cached_property

import nidaqmx.task
import numpy as np
import nidaqmx
import nidaqmx.errors
from nidaqmx.system import System as NISystem
from nidaqmx.constants import AcquisitionType, LineGrouping, Edge, AOIdleOutputBehavior

from dirigo import units
from dirigo.hw_interfaces.scanner import (
    RasterScanner,
    FastRasterScanner, SlowRasterScanner,
    ResonantScanner, GalvoScanner
)



def validate_ni_channel(channel_name: str):
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
    if '/' not in channel_name or channel_name.count('/') > 2:
        raise ValueError(
            f"Invalid channel name format, {channel_name}. "
            f"Valid formats: [device name]/[channel name] or /[device name]/[terminal name]. "
            f"Examples: 'Dev1/ao0', '/Dev1/PFI4'"
        )

    # Split device name from the rest of the channel name
    parts = channel_name.split('/')
    if channel_name.startswith('/'):
        # Handle terminal format: '/Dev1/PFI0'
        device_name = parts[1]
    else:
        # Handle channel format: 'Dev1/ai0'
        device_name = parts[0]

    system = NISystem.local()

    try:
        # Get the device
        device = system.devices[device_name]
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


class FrameClock:
    """
    Counts fast scanner clock pulses and emits a frame clock signal.
    
    This class is used to synchronize a slow scanner with a fast scanner.
    It generates a digital output signal based on a specified number of 
    fast scanner periods per slow scanner frame.

    Note:
        Currently supports only a 50% duty cycle.

    Attributes:
        LINE_CLOCK_MAX_ERROR (float): Safety margin added to the maximum 
            expected line clock frequency as required by NIDAQmx.
    """

    LINE_CLOCK_MAX_ERROR = 0.1  # Add 10% safety margin for maximum frequency

    def __init__(self, 
                 line_clock_channel: str, 
                 frame_clock_channel: str,
                 line_clock_frequency: units.Frequency,
                 periods_per_frame: int):
        """
        Initializes the FrameClock object.

        Args:
            line_clock_channel (str): NI DAQmx channel for the line clock input.
            frame_clock_channel (str): NI DAQmx channel for the frame clock output.
            line_clock_frequency (dirigo.Frequency): Frequency of the line clock.
            periods_per_frame (int): Number of line clock periods per frame clock.

        Raises:
            ValueError: If `periods_per_frame` is not a positive integer.
            ValueError: If the `line_clock_frequency` is not a dirigo.Frequency object.
            ValueError: If the channel name format is invalid or the channel/terminal
                        does not exist.
        """
        # Validate frame clock channel
        validate_ni_channel(frame_clock_channel)
        self._frame_clock_channel = frame_clock_channel

        # Validate line sync channel
        validate_ni_channel(line_clock_channel)
        self._line_clock_channel = line_clock_channel

        if not isinstance(line_clock_frequency, units.Frequency):
            raise ValueError("`line_clock_frequency` must be a frequency object")
        self._line_clock_frequency = line_clock_frequency

        if periods_per_frame <= 0:
            raise ValueError("`periods_per_frame must` be a positive integer.")
        self._periods_per_frame = periods_per_frame

    def start(self):
        """
        Starts the frame clock signal generation.

        Configures and starts a NIDAQmx task to output a frame clock signal
        based on the specified number of line clock periods per frame.

        Raises:
            nidaqmx.DaqError: If the DAQmx task configuration fails.
        """
        # Set up task
        self._task = nidaqmx.Task("Frame clock")
        self._task.do_channels.add_do_chan(
            lines=self._frame_clock_channel,
            line_grouping=LineGrouping.CHAN_PER_LINE
        )

        # Set timing
        max_freq = (1 + self.LINE_CLOCK_MAX_ERROR) * self._line_clock_frequency
        self._task.timing.cfg_samp_clk_timing(
            rate=max_freq, 
            source=self._line_clock_channel,
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS
        )

        # Make the clock signal and write to device
        clock_signal = np.zeros(self._periods_per_frame, dtype=np.bool_)
        clock_signal[:self._periods_per_frame // 2] = True
        self._task.write(clock_signal, auto_start=True)

    def stop(self):
        """
        Stops the frame clock signal generation.

        Stops and closes the associated NIDAQmx task.

        Raises:
            nidaqmx.DaqError: If stopping or closing the task fails.
        """
        self._task.stop()
        self._task.close()


class GalvoRasterScannerViaNI(GalvoScanner):
    REARM_TIME = units.Time("0.5 ms") # time to allow NI card to rearm after outputing waveform
    AO_TIMEBASE = units.Frequency("100 MHz") # NIDAQ cards use 100 MHz timebase divided down to generate frequency

    def __init__(self, control_channel: str, analog_control_range: dict, 
                 trigger_channel: str, line_clock_channel: str, 
                 frame_clock_channel: str, **kwargs):
        super().__init__(**kwargs)
        
        # validated and set amplitude control analog channel
        validate_ni_channel(control_channel)
        self._control_channel = control_channel

        # validate amplitude control limits and set in private attr
        if not isinstance(analog_control_range, dict):
            raise ValueError("`analog_control_range` must be a dictionary.")
        missing_keys = {'min', 'max'} - analog_control_range.keys()
        if missing_keys:
            raise ValueError(
                f"`analog_control_range` must be a dictionary with 'min' and 'max' keys."
            )
        self._analog_control_range = units.VoltageRange(**analog_control_range)

        # validate trigger channel
        validate_ni_channel(trigger_channel)
        self._trigger_channel = trigger_channel

        self._frame_clock: FrameClock = None # Set later with information from fast scanner
        validate_ni_channel(line_clock_channel) # will be validated again when FrameClock object is made, but that's OK
        self._line_clock_channel = line_clock_channel
        validate_ni_channel(frame_clock_channel)
        self._frame_clock_channel = frame_clock_channel

        self.park() # Park itself at min angle

    @cached_property
    def _output_buffer_size(self) -> int:
        """Returns the hardware output buffer size in samples."""
        # We will make a throwaway task to detect the output buffer size value
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(
            physical_channel=self._control_channel
        )
        buffer_size = task.out_stream.output_onbrd_buf_size
        task.close()
        return buffer_size

    @property
    def _sample_rate(self) -> int:
        if not self.frequency:
            raise RuntimeError(
                "Slow scanner `frequency` must be set before preparing frame clock."
            )
        exact_sample_rate = self.frequency * self._output_buffer_size
        x = math.ceil(self.AO_TIMEBASE / exact_sample_rate) 
        return units.SampleRate(self.AO_TIMEBASE / x)
     
    @property
    def _samples_per_period(self) -> float:
        """
        Exact number of output sample clock periods per slow axis scanner period.
        """
        return self._sample_rate / self.frequency
    
    @property
    def _waveform_length(self) -> int:
        """
        Number of samples in output waveform. This is reduce by the rearm time
        and rounded to an integer.
        """
        return round(self._samples_per_period - self.REARM_TIME * self._sample_rate)
    
    def prepare_frame_clock(self, fast_scanner: RasterScanner, acquisition_spec):
        """Prepares the fast scanner"""
        if not self.frequency:
            raise RuntimeError(
                "Slow scanner `frequency` must be set before preparing frame clock."
            )
        self._frame_clock = FrameClock(
            line_clock_channel=self._line_clock_channel,
            frame_clock_channel=self._frame_clock_channel,
            line_clock_frequency=fast_scanner.frequency,
            periods_per_frame=acquisition_spec.records_per_buffer
        )

    def start(self):
        # Check whether all required properties have been set
        if not self.frequency: 
            raise RuntimeError("Required scanner parameter, `frequency` not set.")
        if not self.waveform:
            raise RuntimeError("Required scanner parameter, `waveform` not set.")
        if not self.duty_cycle:
            raise RuntimeError("Required scanner parameter, `duty_cycle` not set.")
        if not self._frame_clock:
            raise RuntimeError("Frame clock must be initialized before starting. "
                               "Use the `prepare_frame_clock` method.")

        # Instantiate tasks and set channels
        self._task = nidaqmx.Task("Galvo slow raster scanner")
        self._task.ao_channels.add_ao_voltage_chan(
            physical_channel=self._control_channel
        )
        
        # Set task timings
        self._task.timing.cfg_samp_clk_timing(
            rate=self._sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self._waveform_length
        )

        self._task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=self._trigger_channel
        )
        self._task.triggers.start_trigger.retriggerable = True
        #self._task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        # Write the waveform to the buffer
        waveform_radians = self._generate_waveform()
        waveform_volts = waveform_radians * self._volts_per_radian
        self._task.write(waveform_volts, auto_start=True) # still requires triggering for output

        # Start frame clock
        self._frame_clock.start()

    def stop(self):
        #self._task.stop()
        self._task.close()
        self.park() # Parks itself at min angle

        self._frame_clock.stop()
        self._frame_clock = None

    def park(self):
        """Positions the scanner the angle limit minimum."""
        analog_value = self.angle_limits.min * self._volts_per_radian
        try:
            with nidaqmx.Task() as task: 
                task.ao_channels.add_ao_voltage_chan(self._control_channel)
                task.write(analog_value, auto_start=True)
        except nidaqmx.DaqError as e:
            raise RuntimeError(f"Failed to park scanner: {e}") from e

    def _generate_waveform(self):
        """Returns are waveform vector in units of radian."""
        amp_rad = self.amplitude / 2 # divide pk-pk amplitude by 2 to get amplitude (converts to radians)

        # TODO, adjustable phase?
        if self.waveform == 'sinusoid':
            t = np.linspace(start=0, stop=2*np.pi, num=self._waveform_length)
            return amp_rad * np.cos(t)
        
        elif self.waveform in {'triangle', 'asymmetric triangle', 'sawtooth'}:
            num_up = math.ceil(self._samples_per_period * self.duty_cycle)
            num_down = self._waveform_length - num_up
            parts = (
                np.linspace(start=-amp_rad, stop=amp_rad, num=num_up),
                np.linspace(start=amp_rad, stop=-amp_rad, num=num_down)
            )
            waveform = np.concatenate(parts, axis=0)
            # if True:
            #     window_width = 100

            #     # Create the kernel (rolling average filter)
            #     kernel = np.ones(window_width) / window_width

            #     # Apply convolution. The mode 'valid' returns output only where the kernel fully overlaps the signal.
            #     waveform = np.convolve(waveform, kernel, mode='same')
            return waveform
    
    @cached_property
    def _volts_per_radian(self) -> float:
        """Scaling factor between analog control voltge and optical scan angle."""
        return self._analog_control_range.range / self.angle_limits.range
 

class GalvoFastRasterScannerViaNI(GalvoRasterScannerViaNI, FastRasterScanner):
    """The faster of a galvo-galvo pair. """
    pass


class GalvoSlowRasterScannerViaNI(GalvoRasterScannerViaNI, SlowRasterScanner):
    """Galvo paired with a fast scanner (resonant, polygon, etc)"""
    pass