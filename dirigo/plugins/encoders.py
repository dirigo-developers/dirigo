import numpy as np
import nidaqmx
from nidaqmx.constants import AcquisitionType

from dirigo import units
from dirigo.components.hardware import Hardware
from dirigo.hw_interfaces.encoder import LinearEncoder, MultiAxisLinearEncoder
from dirigo.plugins.scanners import validate_ni_channel



class LinearEncoderViaNI(LinearEncoder):
    def __init__(self, counter_name: str, sample_clock_channel: str,
                 signal_a_channel: str, signal_b_channel: str,
                 distance_per_pulse: units.Position, 
                 samples_per_channel:int = 2048, # Size of the buffer--set to 2X the expected number of samples to read at once
                 **kwargs):
        super().__init__(**kwargs) # sets axis

        if not isinstance(counter_name, str):
            raise ValueError("`counter_name` must be a string.")
        self._counter_name = counter_name

        validate_ni_channel(sample_clock_channel)
        self._sample_clock_channel = sample_clock_channel

        validate_ni_channel(signal_a_channel)
        validate_ni_channel(signal_b_channel)
        self._signal_a_channel = signal_a_channel
        self._signal_b_channel = signal_b_channel

        self._distance_per_pulse = units.Position(distance_per_pulse)
        self._samples_per_channel = samples_per_channel

    def start(self, initial_position: units.Position, expected_sample_rate: units.SampleRate):
        """Sets up the counter input task and starts it."""
        self._task = nidaqmx.Task() # TODO give it a name

        self._task.ci_channels.add_ci_lin_encoder_chan(
            counter=self._counter_name,
            name_to_assign_to_channel=self._axis,
            dist_per_pulse=self._distance_per_pulse,
            initial_pos=initial_position
        )

        self._task.ci_channels[0].ci_encoder_a_input_term = self._signal_a_channel
        self._task.ci_channels[0].ci_encoder_b_input_term = self._signal_b_channel

        self._task.timing.cfg_samp_clk_timing(
            rate=expected_sample_rate * 1.05, # this is the max expected rate, provide 5% higher rate
            source=self._sample_clock_channel,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._samples_per_channel
        )

    def read(self, n: int):
        """Read n position samples from the task."""
        if not isinstance(n, int):
            raise ValueError("`n_samples` to read must be an integer.")
        if n < 1:
            raise ValueError("Must read at least 1 sample.")
        return self._task.read(n)

    def stop(self):
        self._task.stop()
        self._task.close()


class MultiAxisLinearEncodersViaNI(MultiAxisLinearEncoder):
    """Container for a multi-axis set of linear position quadrature encoders."""
    def __init__(self, x_config: dict = None, y_config: dict = None, 
                 z_config: dict = None, **kwargs):
        self._x = LinearEncoderViaNI(axis='x', **x_config) if x_config else None
        self._y = LinearEncoderViaNI(axis='y', **y_config) if y_config else None
        self._z = LinearEncoderViaNI(axis='z', **z_config) if z_config else None
    
    @property
    def x(self) -> LinearEncoderViaNI:
        return self._x
    
    @property
    def y(self) -> LinearEncoderViaNI:
        return self._y
    
    @property
    def z(self) -> LinearEncoderViaNI:
        return self._z
    
    def start(self, hw: Hardware):
        """Starts all the available encoders.
        
        Expects to be passed a reference to the hardware container, needed to
        query runtime hardware settings.
        """
        if self.x:
            self.x.start(hw.stage.x.position, hw.fast_raster_scanner.frequency)
        if self.y:
            self.y.start(hw.stage.y.position, hw.fast_raster_scanner.frequency)
        if self.z:
            self.z.start(hw.stage.z.position, hw.fast_raster_scanner.frequency)
    
    def read(self, n) -> np.ndarray:
        """Reads n samples from each of the available encoders.
        
        Return in dimensions: Samples, Axes. 
        Example: a 500-sample, X & Y encoder reading would have shape (500,2)
        """
        samples = []
        if self.x:
            samples.append(self.x.read(n))
        if self.y:
            samples.append(self.y.read(n))
        if self.z:
            samples.append(self.z.read(n))

        return np.copy(np.array(samples).T) # Return in dimensions: Samples, Axes
    
    def stop(self):
        """Stops all the available encoders."""
        if self.x:
            self.x.stop()
        if self.y:
            self.y.stop()
        if self.z:
            self.z.stop()