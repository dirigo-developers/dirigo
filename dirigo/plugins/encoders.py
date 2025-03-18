import numpy as np
import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, EncoderType, EncoderZIndexPhase, AngleUnits, ExportAction
)

from dirigo import units
from dirigo.components.hardware import Hardware
from dirigo.hw_interfaces.encoder import LinearEncoder, MultiAxisLinearEncoder
from dirigo.plugins.scanners import validate_ni_channel



class LinearEncoderViaNI(LinearEncoder):
    def __init__(self, counter_name: str, 
                 signal_a_channel: str, signal_b_channel: str,
                 distance_per_pulse: units.Position, 
                 sample_clock_channel: str = None, trigger_channel: str = None,
                 samples_per_channel:int = 2048, # Size of the buffer--set to 2X the expected number of samples to read at once
                 **kwargs):
        super().__init__(**kwargs) # sets axis

        if not isinstance(counter_name, str):
            raise ValueError("`counter_name` must be a string.")
        self._counter_name = counter_name

        # A & B quadrature encoder signals (required)
        validate_ni_channel(signal_a_channel)
        validate_ni_channel(signal_b_channel)
        self._signal_a_channel = signal_a_channel
        self._signal_b_channel = signal_b_channel

        # Ensure at least one of sample_clock_channel or trigger_channel is provided
        if sample_clock_channel is None and trigger_channel is None:
            raise ValueError("At least one of 'sample_clock_channel' or 'trigger_channel' must be provided.")

        # Conditionally validate and assign sample_clock_channel
        if sample_clock_channel is not None:
            validate_ni_channel(sample_clock_channel)
            self._sample_clock_channel = sample_clock_channel
        else:
            self._sample_clock_channel = None

        # Conditionally validate and assign trigger_channel
        if trigger_channel is not None:
            validate_ni_channel(trigger_channel)
            self._trigger_channel = trigger_channel
        else:
            self._trigger_channel = None

        self._distance_per_pulse = units.Position(distance_per_pulse)
        self._samples_per_channel = samples_per_channel

    def start_logging(self, initial_position: units.Position, expected_sample_rate: units.SampleRate):
        """Sets up the counter input task and starts it."""
        self._task = nidaqmx.Task() # TODO give it a name

        self._task.ci_channels.add_ci_lin_encoder_chan(
            counter=self._counter_name,
            name_to_assign_to_channel=f"{self.axis} encoder",
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
    
    def start_triggering(self, distance_per_trigger: units.Position):
        """
        """
        # Calculate the number of encoder pulses per output trigger
        # Example: Thorlabs MLS203 (100nm/pulse) & triggering at each 1um: = 10
        pulses_per_trigger = round(
            distance_per_trigger / self._distance_per_pulse
        )

        self._task = nidaqmx.Task()

        # co-opt 'angular' encoder to trig out on every complete 'revolution'
        self._task.ci_channels.add_ci_ang_encoder_chan(
            counter=self._counter_name,
            name_to_assign_to_channel=f"{self.axis} encoder-derived trigger",
            decoding_type=EncoderType.X_1, # TODO make this adjustable
            zidx_enable=True,
            zidx_val=-pulses_per_trigger,
            zidx_phase=EncoderZIndexPhase.AHIGH_BHIGH,
            units=AngleUnits.TICKS,
            pulses_per_rev=1_000_000, # this is an abritrary high value
            initial_angle=-pulses_per_trigger
        )

        self._task.export_signals.ctr_out_event_output_behavior = ExportAction.PULSE
        
        self._task.export_signals.ctr_out_event_output_term = self._trigger_channel
        
        self._task.ci_channels[0].ci_encoder_a_input_term = self._signal_a_channel
        self._task.ci_channels[0].ci_encoder_b_input_term = self._signal_b_channel

        self._task.ci_channels[0].ci_encoder_z_index_enable = True 
        parts = self._counter_name.split('/')
        ci_encoder_z_input_term = f"/{parts[0]}/Ctr{parts[1][-1]}InternalOutput" # e.g. /Dev1/Ctr1InternalOutput
        validate_ni_channel(ci_encoder_z_input_term)
        self._task.ci_channels[0].ci_encoder_z_input_term = ci_encoder_z_input_term

        self._task.start()

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
    
    def start_logging(self, hw: Hardware):
        """Starts logging on all available encoders.
        
        Expects to be passed a reference to the hardware container, needed to
        query runtime hardware settings.
        """
        if self.x:
            self.x.start_logging(hw.stage.x.position, hw.fast_raster_scanner.frequency)
        if self.y:
            self.y.start_logging(hw.stage.y.position, hw.fast_raster_scanner.frequency)
        if self.z:
            self.z.start_logging(hw.stage.z.position, hw.fast_raster_scanner.frequency)
    
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