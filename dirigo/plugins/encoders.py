from typing import Optional

import numpy as np
import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, EncoderType, EncoderZIndexPhase, AngleUnits, ExportAction,
    TimeUnits
)
from nidaqmx.stream_readers import CounterReader

from dirigo.components import units
from dirigo.components.hardware import Hardware
from dirigo.hw_interfaces.encoder import LinearEncoder, MultiAxisLinearEncoder
from dirigo.plugins.scanners import CounterRegistry, validate_ni_channel



class LinearEncoderViaNI(LinearEncoder):
    def __init__(self,
                 signal_a_channel: str, 
                 signal_b_channel: str,
                 distance_per_pulse: units.Position, 
                 sample_clock_channel: Optional[str] = None, 
                 trigger_channel: Optional[str] = None,
                 timestamp_trigger_events: bool = False,
                 samples_per_channel: int = 4096, # Size of the buffer--set to 2X the expected number of samples to read at once
                 **kwargs):
        super().__init__(**kwargs) # sets axis

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
        self._timestamp_trigger_events = timestamp_trigger_events

    def start_logging(self, 
                      initial_position: units.Position, 
                      expected_sample_rate: units.SampleRate | units.Frequency):
        """Sets up the counter input task and starts it."""
        if self._sample_clock_channel is None:
            raise RuntimeError("Sample clock channel not initialized")

        self._logging_task = nidaqmx.Task() # TODO give it a name

        self._logging_task.ci_channels.add_ci_lin_encoder_chan(
            counter=CounterRegistry.allocate_counter(),
            dist_per_pulse=self._distance_per_pulse,
            initial_pos=initial_position
        )

        self._logging_task.ci_channels[0].ci_encoder_a_input_term = self._signal_a_channel # type: ignore
        self._logging_task.ci_channels[0].ci_encoder_b_input_term = self._signal_b_channel # type: ignore

        self._logging_task.timing.cfg_samp_clk_timing(
            rate=expected_sample_rate * 1.1, # this is the max expected rate, provide 10% higher rate
            source=self._sample_clock_channel,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=self._samples_per_channel
        )

        self._reader = CounterReader(self._logging_task.in_stream)

        self._logging_task.start()

    def read_positions(self, n: int):
        """Read n position samples from the task."""
        if not isinstance(n, int):
            raise ValueError("`n_samples` to read must be an integer.")
        if n < 1:
            raise ValueError("Must read at least 1 sample.")
        
        data = np.empty((n,), dtype=np.float64) # this sort of defeats the purpose of stream readers
        
        self._reader.read_many_sample_double(data, n)
        return data
    
    def read_timestamps(self, n: int):
        """Read n timestamp samples from the task."""
        if not isinstance(n, int):
            raise ValueError("`n_samples` to read must be an integer.")
        if n < 1:
            raise ValueError("Must read at least 1 sample.")
        # timestamp task returns period between trigger up edges
        print(f'Attempting to read {n} timestamps')
        periods = np.array(self._timestamp_task.read(n)) # type: ignore
        print("Min period:", periods.min())

        # convert periods to cumulative timestamps and add previous last timestampe
        timestamps = np.cumsum(periods) + self._last_timestamp
        self._last_timestamp = timestamps[-1]
        
        return timestamps
    
    def start_triggering(self, distance_per_trigger: units.Position):
        """
        """
        # Calculate the number of encoder pulses per output trigger
        # Example: Thorlabs MLS203 (100nm/pulse) & triggering at each 1um: = 10
        pulses_per_trigger = round(
            distance_per_trigger / self._distance_per_pulse
        )

        self._trigger_task = nidaqmx.Task()

        # use angular encoder to trig out on every N pulses
        self._trigger_task.ci_channels.add_ci_ang_encoder_chan(
            counter=CounterRegistry.allocate_counter(),
            decoding_type=EncoderType.X_1, # TODO make this adjustable
            zidx_enable=True,
            zidx_val=-pulses_per_trigger-1,
            zidx_phase=EncoderZIndexPhase.AHIGH_BHIGH,
            units=AngleUnits.TICKS,
            pulses_per_rev=1_000_000, # set this to an abritrary high value
            initial_angle=-pulses_per_trigger-1
        )

        self._trigger_task.export_signals.ctr_out_event_output_behavior = ExportAction.PULSE
        
        self._trigger_task.export_signals.ctr_out_event_output_term = self._trigger_channel
        
        self._trigger_task.ci_channels[0].ci_encoder_a_input_term = self._signal_a_channel # type: ignore
        self._trigger_task.ci_channels[0].ci_encoder_b_input_term = self._signal_b_channel # type: ignore

        parts = self._trigger_task.channel_names[0].split('/')
        cizterm = f"/{parts[0]}/Ctr{parts[1][-1]}InternalOutput" # e.g. /Dev1/Ctr1InternalOutput
        validate_ni_channel(cizterm)
        self._trigger_task.ci_channels[0].ci_encoder_z_input_term = cizterm # type: ignore

        # Creates a separate task that measures the time between edges of
        # the signal coming from 'DevX/CtrYInternalOutput'.
        if self._timestamp_trigger_events:
            self._timestamp_task = nidaqmx.Task()

            # Configure a period measurement channel on free counter
            self._timestamp_task.ci_channels.add_ci_period_chan(
                counter=CounterRegistry.allocate_counter(),           
                min_val=10e-6, # 100 kHz, TODO set this flexibly
                max_val=1.0,
                units=TimeUnits.SECONDS
            )

            # Set to the internal signal exported by ctr0/1.
            parts = self._trigger_task.channel_names[0].split('/')
            cpt = f"/{parts[0]}/Ctr{parts[1][-1]}InternalOutput" # e.g. /Dev1/Ctr1InternalOutput
            self._timestamp_task.ci_channels[0].ci_period_term = cpt # type: ignore
                
            self._timestamp_task.timing.cfg_implicit_timing(
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=self._samples_per_channel
            )

            self._last_timestamp = 0.0
            self._timestamp_task.start()

        self._trigger_task.start()

    def stop(self):
        if hasattr(self, '_trigger_task'):
            self._trigger_task.stop()
            CounterRegistry.free_counter(self._trigger_task.channel_names[0])
            self._trigger_task.close()
        if hasattr(self, '_logging_task'):
            self._logging_task.stop()
            CounterRegistry.free_counter(self._logging_task.channel_names[0])
            self._logging_task.close()


        if self._timestamp_trigger_events:
            self._timestamp_task.stop()
            CounterRegistry.free_counter(self._timestamp_task.channel_names[0])
            self._timestamp_task.close()


class MultiAxisLinearEncodersViaNI(MultiAxisLinearEncoder):
    """Container for a multi-axis set of linear position quadrature encoders."""
    def __init__(self, 
                 x_config: Optional[dict] = None, 
                 y_config: Optional[dict] = None, 
                 z_config: Optional[dict] = None, 
                 **kwargs):
        self._x = LinearEncoderViaNI(axis='x', **x_config) if x_config else None
        self._y = LinearEncoderViaNI(axis='y', **y_config) if y_config else None
        self._z = LinearEncoderViaNI(axis='z', **z_config) if z_config else None
    
    @property
    def x(self) -> LinearEncoderViaNI:
        if self._x is None:
            raise RuntimeError("X encoder not initialized")
        return self._x
    
    @property
    def y(self) -> LinearEncoderViaNI:
        if self._y is None:
            raise RuntimeError("y encoder not initialized")
        return self._y
    
    @property
    def z(self) -> LinearEncoderViaNI:
        if self._z is None:
            raise RuntimeError("Z encoder not initialized")
        return self._z
    
    def start_logging(self, initial_position: list[units.Position], line_rate: units.Frequency):
        """Starts logging on all available encoders.
        
        Expects to be passed a reference to the hardware container, needed to
        query runtime hardware settings.
        """
        if self._x:
            initial_x = initial_position[0]
            self.x.start_logging(initial_x, line_rate)
        if self._y:
            initial_y = initial_position[1]
            self.y.start_logging(initial_y, line_rate)
        if self._z:
            initial_z = initial_position[2] # TODO not sure if need (will we have z-axis encoders to meaure?)
            self.z.start_logging(initial_z, line_rate)
    
    def read_positions(self, n) -> np.ndarray:
        """Reads n samples from each of the available encoders.
        
        Return in dimensions: Samples, Axes. 
        Example: a 500-sample, X & Y encoder reading would have shape (500,2)
        """
        samples = []
        if self._x:
            samples.append(self.x.read_positions(n))
        if self._y:
            samples.append(self.y.read_positions(n))
        if self._z:
            samples.append(self.z.read_positions(n))

        return np.copy(np.array(samples).T) # Return in dimensions: Samples, Axes
    
    def stop(self):
        """Stops all the available encoders."""
        if self._x:
            self.x.stop()
        if self._y:
            self.y.stop()
        if self._z:
            self.z.stop()