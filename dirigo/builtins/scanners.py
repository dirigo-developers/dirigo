import nidaqmx
import nidaqmx.errors
from nidaqmx.system import System as NISystem

import dirigo
from dirigo.hw_interfaces.scanner import (
    ResonantScanner,
    SlowRasterScanner
)


class ResonantScannerViaNI(ResonantScanner):
    """
    Resonant scanner operated via an NIDAQ MFIO card.
    """
    def __init__(self, amplitude_control_channel: str, **kwargs):
        super().__init__(**kwargs)

        # Validate Analog Out channel
        try:
            # Split the device and channel name
            device_name, _ = amplitude_control_channel.split("/", 1)

            # Retrieve the system and the specified device
            system = NISystem.local()
            device = next(
                (d for d in system.devices if d.name == device_name), None
            )

            if not device:
                raise ValueError(f"Device '{device_name}' not found.")

            # Check if the channel exists in the analog output channels
            if amplitude_control_channel not in device.ao_physical_chans.channel_names:
                raise ValueError(f"Channel '{amplitude_control_channel}' does "
                                 f"not exist on device '{device_name}'.")
            
            # Set the validated channel
            self._amplitude_control_channel = amplitude_control_channel

        except ValueError as e:
            raise ValueError(f"Invalid amplitude control channel: {e}") from e
        except nidaqmx.errors.DaqError as e:
            raise nidaqmx.errors.DaqError(f"NI-DAQmx Error during initialization: {e}") from e
        
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
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
    def waveform(self) -> str: # Think on: should this be in SlowRasterScanner?
        return self._waveform
    
    @waveform.setter
    def waveform(self, new_waveform: str):
        if new_waveform not in {'sinusoid', 'sawtooth', 'triangle'}:
            raise ValueError(
                f"Error setting waveform type. Valid options 'sinusoid', "
                f"'sawtooth', 'triangle'. Recieved {new_waveform}"
            )
        self._waveform = new_waveform


# Testing
if __name__ == "__main__":
    # config = {
    #     "axis": "y",
    #     "angle_limits": {"min": "-13.0 deg", "max": "13.0 deg"},
    #     "analog_control_range" : {"min": "0 V", "max": "5 V"},
    #     "frequency": "7910 Hz",
    #     "amplitude_control_channel": "Dev1/ao3"
    # }
    # scanner = ResonantScannerViaNI(**config)

    config = {
        "axis": "x",
        "angle_limits": {"min": "-15.0 deg", "max": "15.0 deg"},
    }

    scanner = GalvoSlowRasterScannerViaNI(**config)