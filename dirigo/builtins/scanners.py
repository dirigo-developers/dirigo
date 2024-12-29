import nidaqmx
import nidaqmx.errors
from nidaqmx.system import System as NISystem
from nidaqmx.constants import AcquisitionType, RegenerationMode

import dirigo
from dirigo.hw_interfaces.scanner import (
    ResonantScanner,
    SlowRasterScanner
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



class ResonantScannerViaNI(ResonantScanner):
    """
    Resonant scanner operated via an NIDAQ MFIO card.
    """
    def __init__(self, amplitude_control_channel: str, **kwargs):
        super().__init__(**kwargs)

        # Validated and set ampltidue control analog channel
        validate_ni_channel(amplitude_control_channel)
        self._amplitude_control_channel = amplitude_control_channel

        # Validate the analog control range
        analog_control_range = kwargs.get('analog_control_range')
        if not isinstance(analog_control_range, dict):
            raise ValueError(
                "analog_control_range must be a dictionary."
            )
        missing_keys = {'min', 'max'} - analog_control_range.keys()
        if missing_keys:
            raise ValueError(
                f"analog_control_range must have 'min' and 'max' keys."
            )
        self._analog_control_range = dirigo.VoltageRange(**analog_control_range)

        # Default: set amplitude to 0 upon startup
        self.amplitude = 0.0

    @property
    def amplitude(self):
        """Get the peak-to-peak amplitude, in radians optical."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_ampl: dirigo.Angle | float):
        """Set the peak-to-peak amplitude."""

        # Validate that the value is within the acceptable range
        if not self.angle_limits.within_range(dirigo.Angle(new_ampl/2)):
            raise ValueError(
                f"Value for 'amplitude' outside settable range "
                f"{self.angle_limits.min} to {self.angle_limits.max}. "
                f"Got: {new_ampl}"
            )
        
        # Calculate the required analog voltage value, validate within range
        ampl_fraction = new_ampl / self.angle_limits.range
        analog_value =  ampl_fraction * self.analog_control_range.max
        if not self.analog_control_range.within_range(dirigo.Voltage(analog_value)):
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
    def enabled(self) -> bool:
        pass

    @enabled.setter
    def enabled(self, new_state: bool):
        pass

    @property
    def analog_control_range(self) -> dirigo.VoltageRange:
        """Returns an object describing the analog control range."""
        return self._analog_control_range


class GalvoSlowRasterScannerViaNI(SlowRasterScanner):
    REARM_TIME = dirigo.Time("1 ms") # time to allow NI card to rearm after outputing waveform

    def __init__(self, control_channel: str, trigger_channel: str, 
                 sample_rate: str, **kwargs):
        super().__init__(**kwargs)

        # Validated and set ampltidue control analog channel
        validate_ni_channel(control_channel)
        self._control_channel = control_channel

        validate_ni_channel(trigger_channel)
        self._trigger_channel = trigger_channel

        # Validate sample rate setting
        self._sample_rate = dirigo.Frequency(sample_rate)
    
    @property
    def amplitude(self) -> dirigo.Angle:
        """
        The peak-to-peak scan amplitude.
        """
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, new_amplitude: dirigo.Angle | float):
        ampl = dirigo.Angle(new_amplitude)
        
        # Check that proposed waveform will not exceed scanner limits
        upper = dirigo.Angle(self.offset + ampl/2)
        if not self.angle_limits.within_range(upper):
            raise ValueError(
                f"Error setting amplitude. Scan waveform would exceed scanner "
                f"upper limit ({self.angle_limits.max_degrees})."
            )
        lower = dirigo.Angle(self.offset - ampl/2)
        if not self.angle_limits.within_range(lower):
            raise ValueError(
                f"Error setting amplitude. Scan waveform would exceed scanner "
                f"lower limit ({self.angle_limits.min_degrees})."
            )

        self._amplitude = ampl

    @property
    def offset(self) -> dirigo.Angle:
        return self._offset
    
    @offset.setter
    def offset(self, new_offset: dirigo.Angle | float):
        offset = dirigo.Angle(new_offset)

        # Check that proposed waveform will not exceed scanner limits
        upper = dirigo.Angle(offset + self.amplitude/2)
        if not self.angle_limits.within_range(upper):
            raise ValueError(
                f"Error setting offset. Scan waveform would exceed scanner "
                f"upper limit ({self.angle_limits.max_degrees})."
            )
        lower = dirigo.Angle(offset - self.amplitude/2)
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
    def frequency(self, new_frequency: dirigo.Frequency | float):
        freq = dirigo.Frequency(new_frequency)

        # Check positive 
        if freq <= 0:
            raise ValueError(
                f"Error setting frequency. Must be positve, got {freq}"
            )
        # (any other contraints? likely, but whose responsibility to check?)
        
        self._frequency = freq
    
    @property
    def waveform(self) -> str: 
        return self._waveform
    
    @waveform.setter
    def waveform(self, new_waveform: str):
        if new_waveform not in {'sinusoid', 'sawtooth', 'triangle'}:
            raise ValueError(
                f"Error setting waveform type. Valid options 'sinusoid', "
                f"'sawtooth', 'triangle'. Recieved {new_waveform}"
            )
        self._waveform = new_waveform

    @property
    def enabled(self) -> bool:
        """
        Indicates whether the scanner is currently enabled. 
        
        When True, the scanner will scan a waveform for each received trigger.
        Setting this property to False disables scanner activity.
        """
        pass

    @enabled.setter
    def enabled(self, new_state: bool):
        pass

    @property
    def samples_per_period(self) -> int:
        """
        Number of analog output samples per slow axis scanner period.
        
        """
        period = 1 / self.frequency
        # The actual scan period is reduced by the scan re-arm time, which
        # should occur within the flyback time.
        samples = (period - self.REARM_TIME) * self._sample_rate

        return int(samples)

    def start(self):
        # Instantiate task object
        self._task = nidaqmx.Task("Galvo slow raster scanner")
        self._task.ao_channels.add_ao_voltage_chan(
            physical_channel=self._control_channel
        )

        # Check whether hardware output buffer size is large enough
        # It may be possible to 'stream' output data to the device in case of a 
        # large output waveform, but not yet tested. Also, it's easy enough to 
        # reduce the sample output rate.
        if self.samples_per_period > self._task.out_stream.output_onbrd_buf_size:
            raise RuntimeError(
                "Error loading slow axis scan waveform to card: too many samples. "
                "Try reducing the sample rate."
            )
        
        self._task.timing.cfg_samp_clk_timing(
            rate=self._sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self.samples_per_period
        )

        self._task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=self._trigger_channel
        )
        self._task.triggers.start_trigger.retriggerable = True
        #self._task.out_stream.regen_mode = RegenerationMode.DONT_ALLOW_REGENERATION

        # Write the waveform to the buffer
        waveform = 1
        self._task.write(waveform, auto_start=False)


# Testing
if __name__ == "__main__":
    # config = {
    #     "axis": "y",
    #     "angle_limits": {"min": "-13.0 deg", "max": "13.0 deg"},
    #     "analog_control_range" : {"min": "0 V", "max": "5 V"},
    #     "frequency": "7910 Hz",
    #     "amplitude_control_channel": "Dev1/ao2"
    # }
    # scanner = ResonantScannerViaNI(**config)

    config = {
        "axis": "x",
        "angle_limits": {"min": "-15.0 deg", "max": "15.0 deg"},
        "control_channel": "Dev1/ao3",
        "trigger_channel": "/Dev1/PFI14",
        "sample_rate": "50 kHz"
    }
    scanner = GalvoSlowRasterScannerViaNI(**config)

    scanner.frequency = 7910 / (256+16)
    scanner.start()

    None