import numpy as np
from functools import cached_property
import time

import nidaqmx
import nidaqmx.system
from nidaqmx.stream_readers import AnalogUnscaledReader, CounterReader
from nidaqmx.constants import (
    ProductCategory, Coupling, Edge, AcquisitionType, TerminalConfiguration
)

from dirigo import units
from dirigo.components.io import load_toml
from dirigo.hw_interfaces import digitizer
from dirigo.sw_interfaces.acquisition import AcquisitionProduct
from dirigo.plugins.scanners import (
    CounterRegistry, get_device, validate_ni_channel, 
    get_min_ao_rate, get_max_ao_rate
)



def get_max_ai_rate(device: nidaqmx.system.Device, channels_enabled: int = 1) -> units.SampleRate:
    if not isinstance(channels_enabled, int):
        raise ValueError("channels_enabled must be integer")
    if channels_enabled > 1:
        aggregate_rate = units.SampleRate(device.ai_max_multi_chan_rate)
        return aggregate_rate / channels_enabled
    elif channels_enabled == 1:
        return units.SampleRate(device.ai_max_single_chan_rate)
    else:
        raise ValueError("channels_enabled must be > 1")
        
    
def get_min_ai_rate(device: nidaqmx.system.Device) -> units.SampleRate:
    return units.SampleRate(device.ai_min_rate)


class NIAnalogChannel(digitizer.Channel):
    """
    Represents a single analog input channel on an NI board.
    Implements the Channel interface with minimal NI-specific constraints.
    """
    _INDEX = 0 # Tracks number of times instantiated
    def __init__(self, device: nidaqmx.system.Device, channel_name: str):
        """
        device_name: e.g. "Dev1"
        channel_name: physical channel name, e.g. "Dev1/ai0".
        """
        super().__init__()

        self._device = device
        self._channel_name = validate_ni_channel(channel_name)

        self._index = self.__class__._INDEX
        self.__class__._INDEX += 1

        self._coupling: Coupling = None 
        self._impedance: float = None  # Not adjustable on most boards
        self._range: tuple[float, float] = None # (min, max)

    @property
    def index(self) -> int:
        # gets the numbers after ai in "Dev1/ai0"
        return self._index

    @property
    def coupling(self) -> str:
        if self._coupling == Coupling.DC:
            return "DC"
        else:
            return "AC"

    @coupling.setter
    def coupling(self, coupling: str):
        if coupling not in self.coupling_options:
            raise ValueError(f"Coupling mode, {coupling} not supported by the device")
        
        if coupling == "DC":
            coupling_enum = Coupling.DC
        elif coupling == "AC":
            coupling_enum = Coupling.AC
        else:
            raise ValueError(f"Unsupported coupling mode, {coupling}")
        
        self._coupling = coupling_enum

    @cached_property
    def coupling_options(self) -> set[str]:
        couplings = self._device.ai_couplings
        coupling_map = {Coupling.AC: "AC", Coupling.DC: "DC", Coupling.GND: "GND"}
        return {coupling_map[code] for code in couplings}

    @property
    def impedance(self) -> units.Resistance:
        if len(self.impedance_options) == 1:
            return self.impedance_options[0]
        else:
            raise NotImplementedError("Multiple impedances not yet implemented.")

    @impedance.setter
    def impedance(self, impedance: str):
        pass # just ignore

    @cached_property
    def impedance_options(self) -> set[str]:
        if self._device.product_category == ProductCategory.X_SERIES_DAQ:
            return {units.Resistance("10 Gohm")}
        
        # TODO add impedance for S series

    @property
    def range(self) -> units.VoltageRange:
        return units.VoltageRange(min=self._range[0], max=self._range[1])

    @range.setter
    def range(self, new_rng: units.VoltageRange):
        #if not any(new_rng == r for r in self.range_options):
        if new_rng not in self.range_options:
            valid = list(self.range_options)
            raise ValueError(f"Range {new_rng} invalid. Valid options: {valid}")
        # generally we store data in closest form to underlying API, here just some float values
        self._range = (float(new_rng.min), float(new_rng.max)) 

    @cached_property
    def range_options(self) -> set[units.VoltageRange]:
        r = self._device.ai_voltage_rngs
        # This returns something like:
        # [-0.1, 0.1, -0.2, 0.2, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0, -5.0, 5.0, -10.0, 10.0]

        for i in range(len(r)//2): # check that all ranged are bipolar
            if not (r[2*i] == -r[2*i+1]):
                raise RuntimeError("Encountered unexpected non-symmetric range")

        return {
            (units.VoltageRange(min=l,max=h)) for l,h in zip(r[0::2],r[1::2])
        }
    
    @property
    def channel_name(self) -> str:
        """Returns the NI physical channel name, e.g. Dev1/ai0"""
        return self._channel_name


class NICounterChannel(digitizer.Channel):
    """For edge counting (e.g. photon counting)."""
    _INDEX = 0 
    def __init__(self, device: nidaqmx.system.Device, channel_name: str):
        super().__init__()

        self._device = device
        self._channel_name = validate_ni_channel(channel_name)

        self._index = self.__class__._INDEX
        self.__class__._INDEX += 1

    @property
    def index(self) -> int:
        return self._index

    @property
    def coupling(self) -> str:
        # Digital input 
        return NotImplemented
    
    @coupling.setter
    def coupling(self, coupling: str):
        pass

    @cached_property
    def coupling_options(self) -> set[str]:
        return None
    
    @property
    def impedance(self) -> units.Resistance:
        return NotImplemented
    
    @impedance.setter
    def impedance(self, impedance: str):
        pass 

    @cached_property
    def impedance_options(self) -> set[str]:
        return None
    
    @property
    def range(self) -> units.VoltageRange:
        return NotImplemented

    @range.setter
    def range(self, new_rng: units.VoltageRange):
        pass 

    @cached_property
    def range_options(self) -> set[units.VoltageRange]:
        return None

    @property
    def inverted(self) -> bool:
        # Specifically override the inverted getter/setter methods because 
        # inverting the channel values do not make sense for edge counting
        return False # can't invert counting
    
    @inverted.setter
    def inverted(self, invert: bool):
        pass

    @property
    def channel_name(self) -> str:
        """Returns the NI physical channel name, e.g. Dev1/ai0"""
        return self._channel_name


class NISampleClock(digitizer.SampleClock):
    """
    Configures the NI sample clock. 
    For many NI boards, the typical usage is:
       - source = "OnboardClock" or e.g. "PFI0"
       - rate = up to the board’s max sampling rate
       - edge = "rising" or "falling"
    """

    def __init__(self, device: nidaqmx.system.Device, channels: list[NIAnalogChannel]):
        self._device = device
        self._channels = channels

        # Check the type of Channels to infer mode
        if isinstance(self._channels[0], NIAnalogChannel):
            self._mode = "analog"
        else:
            self._mode = "edge counting"

        self._source = None
        self._rate = None 
        self._edge = "rising"

    @property
    def source(self) -> str:
        """
        Digitizer sample clock source.
        
        Pass None to use internal AI clock engine or pass a valid terminal 
        string to use an external sample clock.
        """
        return self._source

    @source.setter
    def source(self, source: str):
        if not isinstance(source, str):
            raise ValueError("Sample clock source must be set with a string")
        if source.lower() in ["internal", "internal clock"]:
            self._source = None
        else:
            self._source = validate_ni_channel(source)

    @property
    def source_options(self) -> set[str]:
        # This is device-dependent; an example set
        return {"/Dev1/ao/SampleClock", "PFI0", "PFI1"}

    @property
    def rate(self) -> units.SampleRate:
        if self._rate:
            return units.SampleRate(self._rate)
        else:
            return None

    @rate.setter
    def rate(self, value: units.SampleRate | None):
        if value is None:
            self._rate = None
            return
        
        if not isinstance(value, units.SampleRate):
            value = units.SampleRate(value)

        if not self.rate_options.within_range(value):
            raise ValueError(
                f"Requested pixel sample rate ({value}) is outside "
                f"supported range {self.rate_options}"
            )
        
        self._rate = float(value)

    @property
    def rate_options(self) -> units.SampleRateRange:
        """
        NI boards generally support a continuous range up to some max.
        If you want to provide a discrete set, adapt. 
        For demonstration, we show 'continuous range' by returning None 
        or a wide range object.
        """
        if self._mode == "analog":
            if self._device.product_category == ProductCategory.X_SERIES_DAQ:
                # For X-series analog sampling, sample rate needs to be within the aggregrate sample rate
                nchannels_enabled = sum([channel.enabled for channel in self._channels])
                return units.SampleRateRange(
                    min=get_min_ai_rate(self._device), 
                    max=get_max_ai_rate(self._device, nchannels_enabled)
                )
            elif self._device.product_category == ProductCategory.S_SERIES_DAQ:
                # For S-series (Simultaneous sampling), sample rate is independent of number channels activated
                return units.SampleRateRange(
                    min=get_min_ai_rate(self._device), 
                    max=get_max_ai_rate(self._device)
                )
        else:
            # For edge counting, sample rate needs to just be within valid AO range
            return units.SampleRateRange(
                min=get_min_ao_rate(self._device), 
                max=get_max_ao_rate(self._device)
            )

    @property
    def edge(self) -> str:
        return self._edge

    @edge.setter
    def edge(self, edge: str):
        if edge.lower() not in {"rising", "falling"}:
            raise ValueError("NI sample clock only supports rising/falling edge.")
        self._edge = edge

    @property
    def edge_options(self) -> set[str]:
        return {"rising", "falling"}        


class NITrigger(digitizer.Trigger):
    """
    Configures triggering for NI boards. 
    """
    def __init__(self, device: nidaqmx.system.Device):
        self._device = device

        self._source = "/Dev1/ao/StartTrigger"  # e.g. "PFI0" or "None" for immediate 
        self._slope = "rising"
        self._level = units.Voltage(0.0)
        self._ext_coupling = "DC"
        self._ext_range = "+/-10V"

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, source: str):
        self._source = validate_ni_channel(source)

    @property
    def source_options(self) -> set[str]:
        return {"/Dev1/ao/StartTrigger", "None", "PFI0", "PFI1"}

    @property
    def slope(self) -> str:
        return self._slope

    @slope.setter
    def slope(self, slope: str):
        if slope.lower() not in {"rising", "falling"}:
            raise ValueError("NI only supports rising/falling slopes for digital triggers.")
        self._slope = slope

    @property
    def slope_options(self) -> set[str]:
        return {"rising", "falling"}

    @property
    def level(self) -> units.Voltage:
        return NotImplemented

    @level.setter
    def level(self, level: units.Voltage):
        pass  # Typically not used for digital triggers

    @property
    def level_limits(self) -> units.VoltageRange:
        return NotImplemented

    @property
    def external_coupling(self) -> str:
        # NI boards generally are DC-coupled on ext lines
        return "DC"

    @external_coupling.setter
    def external_coupling(self, coupling: str):
        if coupling != "DC":
            raise ValueError("NI boards typically only support DC external trigger coupling.")
        self._ext_coupling = coupling

    @property
    def external_coupling_options(self) -> set[str]:
        return {"DC"}

    @property
    def external_range(self) -> str:
        return NotImplemented

    @external_range.setter
    def external_range(self, range: str):
        pass

    @property
    def external_range_options(self) -> set[str]:
        return NotImplemented


class NIAcquire(digitizer.Acquire):
    """
    Manages data acquisition from NI boards, including buffer creation and 
    reading. For simplicity, we do “continuous” sampling with ring buffers 
    or a user-specified “finite” acquisition.
    """

    def __init__(self, device: nidaqmx.system.Device, sample_clock: NISampleClock, 
                 channels: list[NIAnalogChannel | NICounterChannel], trigger: NITrigger):
        self._device = device
        self._channels: list[NIAnalogChannel | NICounterChannel] = channels
        self._sample_clock: NISampleClock = sample_clock
        self._trigger: NITrigger = trigger

        # The Dirigo interface wants these; for slow or mid-rate NI tasks, we 
        # often do single continuous acquisitions. We'll do a rough approach:
        self._trigger_delay_samples = 0  # Not widely used in NI
        self._record_length: int = None       # e.g. 1000 samples
        self._records_per_buffer = None     
        self._buffers_per_acquisition = None
        self._buffers_allocated = 1

        self._timestamps_enabled = False

        # NI Tasks & state:
        self._task = None
        self._started = False
        self._samples_acquired = 0

    @property
    def trigger_delay_samples(self) -> int:
        return self._trigger_delay_samples

    @trigger_delay_samples.setter
    def trigger_delay_samples(self, samples: int):
        # NI doesn’t have a direct concept of “trigger delay in sample ticks” 
        # for AI tasks (except with analog triggering + advanced timing). 
        # We'll store but not apply directly.
        self._trigger_delay_samples = samples

    @property
    def trigger_delay_duration(self) -> units.Time:
        return units.Time(self._trigger_delay_samples / self._sample_clock.rate)

    @property
    def trigger_delay_sample_resolution(self) -> int:
        # Not a standard NI concept. We can return 1 as a fallback.
        return 1

    @property
    def pre_trigger_samples(self):
        # NI can do “reference trigger” tasks with pre-trigger samples. 
        # Let’s omit that for the example or store it as zero.
        return 0

    @pre_trigger_samples.setter
    def pre_trigger_samples(self, value):
        pass  # Not implementing

    @property
    def pre_trigger_resolution(self):
        return 1

    @property
    def record_length(self) -> int:
        return self._record_length

    @record_length.setter
    def record_length(self, length: int):
        self._record_length = length

    @property
    def record_duration(self) -> units.Time:
        return units.Time(self._record_length / self._sample_clock.rate)

    @property
    def record_length_minimum(self) -> int:
        return 1

    @property
    def record_length_resolution(self) -> int:
        if self._mode == "analog":
            return 32
        elif self._mode == "edge counting":
            return 1

    @property
    def records_per_buffer(self) -> int:
        return self._records_per_buffer

    @records_per_buffer.setter
    def records_per_buffer(self, records: int):
        self._records_per_buffer = records

    @property
    def buffers_per_acquisition(self) -> int:
        return self._buffers_per_acquisition

    @buffers_per_acquisition.setter
    def buffers_per_acquisition(self, buffers: int):
        self._buffers_per_acquisition = buffers

    @property
    def buffers_allocated(self) -> int:
        return self._buffers_allocated

    @buffers_allocated.setter
    def buffers_allocated(self, buffers: int):
        self._buffers_allocated = buffers

    @property
    def timestamps_enabled(self) -> bool:
        return self._timestamps_enabled

    @timestamps_enabled.setter
    def timestamps_enabled(self, enable: bool):
        # Not typically relevant for NI AI tasks; we’ll just store
        self._timestamps_enabled = enable

    def start(self):

        if self._started:
            return  # Already started
        
        if self._sample_clock.edge == "rising":
            edge = Edge.RISING 
        else:
            edge = Edge.FALLING

        if self._buffers_per_acquisition == 1:
            sample_mode = AcquisitionType.FINITE
            samples_per_chan = self._record_length
        else:
            sample_mode = AcquisitionType.CONTINUOUS
            samples_per_chan = 2 * self.records_per_buffer * self.record_length 

        if self._mode == "analog":

            self._task = nidaqmx.Task("Analog input")

            for channel in self._channels:
                if not channel.enabled:
                    continue
                
                ai_channel = self._task.ai_channels.add_ai_voltage_chan(
                    physical_channel=channel.channel_name,
                    min_val=channel.range.min,
                    max_val=channel.range.max,
                ) 
                if self._device.product_category == ProductCategory.X_SERIES_DAQ:
                    ai_channel.ai_term_cfg = TerminalConfiguration.RSE
                elif self._device.product_category == ProductCategory.S_SERIES_DAQ:
                    # S series only supports pseudo-differential
                    ai_channel.ai_term_cfg = TerminalConfiguration.PSEUDO_DIFF

            # Configure the sample clock
            self._task.timing.cfg_samp_clk_timing(
                rate=self._sample_clock.rate, 
                source=self._sample_clock.source, 
                active_edge=edge,
                sample_mode=sample_mode,
                samps_per_chan=samples_per_chan*2 #TODO, not sure about 2x
            )

            # Set up stream reader
            self._reader = AnalogUnscaledReader(self._task.in_stream)
        
        else: # For edge counting:
            self._tasks: list[nidaqmx.Task] = []
            self._readers: list[CounterReader] = []

            for channel in self._channels:
                if not channel.enabled:
                    continue

                # For counter inputs, we need to make multiple tasks and readers
                x = channel.channel_name.split('/')[-1]
                task = nidaqmx.Task(f"Edge counter input {x}")

                ci_chan = task.ci_channels.add_ci_count_edges_chan(
                    counter=CounterRegistry.allocate_counter(),
                )
                ci_chan.ci_count_edges_term = channel.channel_name

                # Configure the sample clock
                if self._sample_clock.source is None:
                    source = "/" + self._device.name + "/ao/SampleClock"
                else:
                    source = self._sample_clock.source

                task.timing.cfg_samp_clk_timing(
                    rate=self._sample_clock.rate, 
                    source=source,
                    active_edge=edge,
                    sample_mode=sample_mode,
                    samps_per_chan=samples_per_chan*4 # TODO not sure about 2x?
                )

                reader = CounterReader(task.in_stream)

                self._tasks.append(task)
                self._readers.append(reader)

            self._last_samples = np.zeros(
                shape=(1, self.n_channels_enabled),
                dtype=np.uint32
            )

        self._inverted_vector = np.array(
            [-2*channel.inverted+1 for channel in self._channels if channel.enabled],
            dtype=np.int8
        )

        # Start the task(s)
        if self._mode == "analog":
            self._task.start()
        else:
            for task in self._tasks:
                task.start()

        self._started = True
        self._buffers_acquired = 0

    @property
    def buffers_acquired(self) -> int:
        return self._buffers_acquired

    def get_next_completed_buffer(self, blocking: bool = True) -> AcquisitionProduct:
        """
        Reads the next chunk of data from the device buffer. For NI, this typically 
        means calling read once we have enough samples. 
        """
        if not self._started:
            raise RuntimeError("Acquisition not started.")
        
        # Decide how many samples to read each time. 
        nsamples = self.records_per_buffer * self.record_length 

        if self._mode == "analog":

            data = np.zeros( # TODO use preallocated array?
                shape=(self.n_channels_enabled, nsamples), 
                dtype=np.int16
            ) 
            self._reader.read_int16(
                data=data,
                number_of_samples_per_channel=nsamples
            )

            # Invert channels if necessary (TODO, may be slightly faster to broadcast multiply a +/-1 vector)
            for i, invert in enumerate(self._inverted_channels):
                if invert:
                    data[i,:] = -data[i,:] 

            # The buffer dimensions need to be reordered for further processing
            data.shape = (self.n_channels_enabled, self.records_per_buffer, self.record_length)
            data = np.transpose(data, axes=(1,2,0)) # Move channels dimension from major to minor (ie interleaved)

        else: # Edge counting mode
            data_single_channel = np.zeros((nsamples,), dtype=np.uint32) # reader only supports reading into contiguous array
            data_multiple_channels = np.zeros(
                shape=(nsamples+1, self.n_channels_enabled), #+1 b/c we will do np.diff
                dtype=np.uint32
            )
            
            for i, reader in enumerate(self._readers):
                t0 = time.perf_counter()
                reader.read_many_sample_uint32(
                    data=data_single_channel,
                    number_of_samples_per_channel=nsamples
                )
                t1 = time.perf_counter()
                data_multiple_channels[1:,i] = data_single_channel
                print(f'Read {nsamples} samples. Waited {t1-t0}')

            # Difference along the samples dim and reorder dims for further processing
            data_multiple_channels[0,:] = self._last_samples
            data = np.diff(data_multiple_channels, axis=0).astype(np.uint16)
            self._last_samples = data_multiple_channels[-1,:]
            data.shape = (self.records_per_buffer, self.record_length, self.n_channels_enabled)

        self._buffers_acquired += 1

        # Construct an AcquisitionBuffer. NI doesn’t provide built-in timestamps
        return AcquisitionProduct(data=data)

    def stop(self):
        if not self._started:
            return

        # Stop the task(s)
        if self._mode == "analog":
            self._task.stop()
            self._task.close()
        else:
            for task in self._tasks:
                task.stop()
                CounterRegistry.free_counter(task.channel_names[0])
                task.close()
                
        self._started = False
        self._samples_acquired = 0

    @cached_property
    def _mode(self):
        if isinstance(self._channels[0], NIAnalogChannel):
            return "analog"
        else:
            return "edge counting"


class NIAuxillaryIO(digitizer.AuxillaryIO):
    def __init__(self, device: nidaqmx.system.Device):
        self._device = device

    def configure_mode(self): 
        pass

    def read_input(self):
        pass

    def write_output(self, state):
        pass



class NIDigitizer(digitizer.Digitizer):
    """
    High-level aggregator for NI-based digitizer integration. 
    Wires together the Channel, SampleClock, Trigger, Acquire, and AuxillaryIO.
    """
    def __init__(self, 
                 device_name: str = "Dev1", 
                 **kwargs): 
        self._device = get_device(device_name)

        # Get channel names from default profile 
        # (NI cards have lots of AI channels, so avoid instantiating all of them)
        profile = load_toml(self.PROFILE_LOCATION / "default.toml")
        channel_names: list[str] = profile["channels"]["channels"]

        # Infer mode from profile->channels->channels
        if channel_names[0].split('/')[-1][:2] == "ai":
            # if the last two characters of channel names = "ai" then we are using analog mode
            self._mode = "analog"
        else:
            self._mode = "edge counting"

        # Create channel objects
        if self._mode == "analog":
            self.channels = \
                [NIAnalogChannel(self._device, chan) for chan in channel_names]
        else:
            self.channels = \
                [NICounterChannel(self._device, chan) for chan in channel_names]

        # Create sample clock
        self.sample_clock = NISampleClock(self._device, self.channels)

        # Create trigger
        self.trigger = NITrigger(self._device)

        # Create acquisition manager
        self.acquire = NIAcquire(
            device=self._device,
            sample_clock=self.sample_clock,
            channels=self.channels,
            trigger=self.trigger
        )

        # Create auxiliary IO
        self.aux_io = NIAuxillaryIO(device=self._device)

    @cached_property
    def data_range(self) -> units.ValueRange:
        """Range of the returned data."""
        if self._mode == "analog":
            # Make dummy task to get at the .ai_resolution property
            with nidaqmx.Task("AI dummy") as task:
                channel = task.ai_channels.add_ai_voltage_chan(
                    physical_channel=self._device.ai_physical_chans.channel_names[0]
                )
                N = int(channel.ai_resolution)
            return units.ValueRange(min=-2**N//2, max=2**N//2 - 1)
        else:
            # For edge counting, use uint8 (max 256 edges/photons per pixel)
            # technically the counters support up to 32 bits, but it's unlikely
            # anyone will need this range
            return units.ValueRange(min=0, max=2**8 - 1)
    
