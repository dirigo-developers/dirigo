import nidaqmx
import nidaqmx.errors
from nidaqmx.system import System as NISystem

from dirigo.components.utilities import VoltageRange
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
        self._analog_control_range = VoltageRange(**analog_control_range)

    @property
    def amplitude(self):
        """Get the peak-to-peak amplitude, in degrees optical."""
        return self._amplitude

    @amplitude.setter
    def amplitude(self, new_ampl: float):
        """Set the peak-to-peak amplitude, in degrees optical."""
        try: 
            # Convert input to a float
            new_ampl = float(new_ampl)
        except ValueError:
            raise ValueError(f"Expected a float value for 'amplitude', got {type(new_ampl)}: {new_ampl}")
        
        # Validate that the value is within the acceptable range
        if not self.angle_limits.within_limits(new_ampl/2):
            raise ValueError(
                f"Value for 'amplitude' outside settable range "
                f"{self.angle_limits.min} to {self.angle_limits.max}. "
                f"Got: {new_ampl}"
            )
        
        # Calculate the required analog voltage value, validate within range
        ampl_fraction = new_ampl / self.angle_limits.range
        analog_value =  ampl_fraction * self.analog_control_range.max
        if not self.analog_control_range.within_limits(analog_value):
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
    def analog_control_range(self) -> VoltageRange:
        """Returns an object describing the analog control range."""
        return self._analog_control_range


class GalvoSlowRasterScannerViaNI(SlowRasterScanner):
    def __init__(self, defaults:dict):
        self._axis:str = defaults["axis"]
        self._amplitude = None
        self._frequency = None
        self._waveform = None

    @property
    def axis(self) -> str:
        return self._axis
    
    @property
    def amplitude(self) -> float:
        return self._amplitude
    
    @property
    def frequency(self) -> float:
        return self._frequency
    
    @property
    def waveform(self) -> str:
        return self._waveform


if __name__ == "__main__":
    config = {
        "axis": "y",
        "angle_limits": {"min": "-13.0 degrees", "max": "13.0 degrees"},
        "analog_control_range" : {"min": "0 V", "max": "5 V"},
        "frequency": "7910 Hz",
        "amplitude_control_channel": "Dev1/ao3"
    }
    scanner = ResonantScannerViaNI(**config)

    None