import numpy as np
from functools import cached_property

import nidaqmx
import nidaqmx.system
from nidaqmx.stream_readers import AnalogUnscaledReader, CounterReader
from nidaqmx.constants import (
    ProductCategory, Coupling, Edge, AcquisitionType, TerminalConfiguration
)

from dirigo.components import units
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
    _coupling_map = { # Dirigo coupling enumerations -> NI coupling enumerations
        digitizer.ChannelCoupling.AC:       Coupling.AC,
        digitizer.ChannelCoupling.DC:       Coupling.DC, 
        digitizer.ChannelCoupling.GROUND:   Coupling.GND 
    }
    
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

        self._coupling: digitizer.ChannelCoupling | None = None 
        self._impedance: units.Resistance | digitizer.ImpedanceMode | None = None  # Not adjustable on most boards
        self._input_range: units.VoltageRange | None = None 

    @property
    def index(self) -> int:
        return self._index

    @property
    def coupling(self) -> digitizer.ChannelCoupling:
        if self._coupling is None:
            raise RuntimeError("Coupling not initialized.")
        return self._coupling

    @coupling.setter
    def coupling(self, coupling: digitizer.ChannelCoupling):
        if coupling not in self.coupling_options:
            raise ValueError(f"Coupling mode, {coupling} not supported by the device")
        self._coupling = coupling

    @cached_property
    def coupling_options(self) -> set[digitizer.ChannelCoupling]:
        ni_couplings: list[Coupling] = self._device.ai_couplings
        rvs_table = {v: k for k, v in self._coupling_map.items()}
        return set([rvs_table[ni_c] for ni_c in ni_couplings])

    @property
    def impedance(self) -> units.Resistance | digitizer.ImpedanceMode:
        if len(self.impedance_options) == 1:
            return next(iter(self.impedance_options))
        else:
            raise NotImplementedError("Multiple impedances not yet implemented.")

    @impedance.setter
    def impedance(self, impedance: units.Resistance | digitizer.ImpedanceMode):
        pass # no-op for now

    @cached_property
    def impedance_options(self) -> set[units.Resistance | digitizer.ImpedanceMode]:
        if self._device.product_category == ProductCategory.X_SERIES_DAQ:
            return set([digitizer.ImpedanceMode.HIGH,])
        elif self._device.product_category == ProductCategory.S_SERIES_DAQ:
            raise NotImplementedError("Impedance not implemented for S-series")
        else:
            raise RuntimeError(f"Unsupported device series {self._device.product_category}")

    @property
    def input_range(self) -> units.VoltageRange:
        if self._input_range is None:
            raise RuntimeError("Input range is not initialized.")
        return self._input_range

    @input_range.setter
    def input_range(self, new_rng: units.VoltageRange):
        if new_rng not in self.range_options:
            valid = list(self.range_options)
            raise ValueError(f"Range {new_rng} invalid. Valid options: {valid}")
        
        self._input_range = new_rng

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
    def coupling(self) -> digitizer.ChannelCoupling:
        return digitizer.ChannelCoupling.DC # TODO, for digital input is DC OK?
    
    @coupling.setter
    def coupling(self, coupling: str):
        if coupling != digitizer.ChannelCoupling.DC:
            raise ValueError("Counter inputs can only be DC coupled.")

    @cached_property
    def coupling_options(self) -> set[digitizer.ChannelCoupling]:
        return set([digitizer.ChannelCoupling.DC])
    
    @property
    def impedance(self) -> units.Resistance | digitizer.ImpedanceMode:
        return digitizer.ImpedanceMode.HIGH
    
    @impedance.setter
    def impedance(self, impedance: units.Resistance | digitizer.ImpedanceMode):
        if impedance != digitizer.ImpedanceMode.HIGH:
            raise ValueError("Counter input impedance can only be high") 

    @cached_property
    def impedance_options(self) -> set[units.Resistance | digitizer.ImpedanceMode]:
        return set([digitizer.ImpedanceMode.HIGH])
    
    @property
    def input_range(self) -> units.VoltageRange:
        return units.VoltageRange("0 V", "5 V")

    @input_range.setter
    def input_range(self, new_rng: units.VoltageRange):
        if new_rng != units.VoltageRange("0 V", "5 V"):
            raise ValueError("Counter input range is not settable.") 

    @cached_property
    def range_options(self) -> set[units.VoltageRange]:
        return set([units.VoltageRange("0 V", "5 V")])

    @property
    def inverted(self) -> bool:
        # Specifically override the inverted getter/setter methods because 
        # inverting the channel values do not make sense for edge counting
        return False # can't invert counting
    
    @inverted.setter
    def inverted(self, invert: bool):
        if invert is True:
            raise ValueError("Cannot invert counter input channel")

    @property
    def channel_name(self) -> str:
        """Returns the NI physical channel name, e.g. Dev1/ai0"""
        return self._channel_name


class NISampleClock(digitizer.SampleClock):
    """
    Configures the NI sample clock. 
    For many NI boards, the typical usage is:
       - source = "OnboardClock" or e.g. "PFI0"
       - rate = up to the board's max sampling rate
       - edge = "rising" or "falling"
    """

    def __init__(self, 
                 device: nidaqmx.system.Device, 
                 external_clock_channel: str | None,
                 channels: list[NIAnalogChannel]):
        self._device = device
        if external_clock_channel is not None:
            validate_ni_channel(external_clock_channel) # Typically a PFI channel
        self._external_clock_channel = external_clock_channel 
        self._channels = channels

        # Check the type of Channels to infer mode
        if isinstance(self._channels[0], NIAnalogChannel):
            self._input_mode = digitizer.InputMode.ANALOG
        else:
            self._input_mode = digitizer.InputMode.EDGE_COUNTING

        self._source: digitizer.SampleClockSource | None = None
        self._rate = None 
        self._edge = "rising"

    @property
    def source(self) -> digitizer.SampleClockSource:
        """
        Digitizer sample clock source.
        
        Pass None to use internal AI clock engine or pass a valid terminal 
        string to use an external sample clock.
        """
        if self._source is None:
            raise RuntimeError("Trigger source not initialized.")
        return self._source

    @source.setter
    def source(self, source: digitizer.SampleClockSource):
        if not isinstance(source, digitizer.SampleClockSource):
            raise ValueError("Sample clock source must be set with SampleClockSource")
        self._source = source

    @property
    def source_options(self) -> set[str]:
        return {digitizer.SampleClockSource.INTERNAL,
                digitizer.SampleClockSource.EXTERNAL}

    @property
    def rate(self) -> units.SampleRate:
        if self._rate is None:
            raise RuntimeError("Sample rate not initialized.")
        return units.SampleRate(self._rate)

    @rate.setter
    def rate(self, value: units.SampleRate):        
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
        """
        if self._input_mode == digitizer.InputMode.ANALOG:
            if self._device.product_category == ProductCategory.X_SERIES_DAQ:
                # For X-series analog sampling, sample rate is dependent on number of channels enabled
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
                raise RuntimeError(f"Unsupported product category: {self._device.product_category}")
        else:
            # For edge counting, rate must be in AO rate range (which will be sample/pixel clock)
            return units.SampleRateRange(
                min=get_min_ao_rate(self._device), 
                max=get_max_ao_rate(self._device)
            )

    @property
    def edge(self) -> str:
        return self._edge

    @edge.setter
    def edge(self, edge: digitizer.SampleClockEdge):
        if not isinstance(edge, digitizer.SampleClockEdge):
            raise ValueError("NI sample clock must be set with digitizer.SampleClockEdge")
        self._edge = edge

    @property
    def edge_options(self) -> set[str]:
        return {digitizer.SampleClockEdge.RISING,
                digitizer.SampleClockEdge.FALLING}        


class NITrigger(digitizer.Trigger):
    """
    Configures triggering for NI boards. 
    """
    def __init__(self, device: nidaqmx.system.Device):
        self._device = device

        self._source_channel: str = "/Dev1/ao/StartTrigger"  # e.g. "PFI0" or "None" for immediate 
        self._slope: digitizer.TriggerSlope | None = None
        self._level = units.Voltage(0.0)
        self._ext_coupling = "DC"
        self._ext_range = "+/-10V"

    # trigger source with NI is tricky to abstract because it could come from many different channels
    @property
    def source(self) -> digitizer.TriggerSource:
        if "/ao/StartTrigger" in self._source_channel:
            return digitizer.TriggerSource.INTERNAL
        else:
            return digitizer.TriggerSource.EXTERNAL

    @source.setter
    def source(self, source: digitizer.TriggerSource):
        if source not in self.source_options:
            raise ValueError(f"Unsupported trigger source option: {source}")
        self._source = source

    @property
    def source_options(self) -> set[digitizer.TriggerSource]:
        return {digitizer.TriggerSource.EXTERNAL, 
                digitizer.TriggerSource.INTERNAL}

    @property
    def slope(self) -> digitizer.TriggerSlope:
        if self._slope is None:
            raise RuntimeError("Trigger slope not initialized")
        return self._slope

    @slope.setter
    def slope(self, slope: digitizer.TriggerSlope):
        if slope not in self.slope_options:
            raise ValueError(f"Unsupported trigger slope option: {slope}")
        self._slope = slope

    @property
    def slope_options(self) -> set[digitizer.TriggerSlope]:
        return {digitizer.TriggerSlope.RISING, 
                digitizer.TriggerSlope.FALLING}

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
        if coupling is not None:
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

    def __init__(self, 
                 device: nidaqmx.system.Device, 
                 sample_clock: NISampleClock, 
                 channels: list[NIAnalogChannel | NICounterChannel], 
                 trigger: NITrigger):
        self._device = device
        self._channels: list[NIAnalogChannel | NICounterChannel] = channels
        self._sample_clock: NISampleClock = sample_clock
        self._trigger: NITrigger = trigger

        # The Dirigo interface wants these; for slow or mid-rate NI tasks, we 
        # often do single continuous acquisitions. We'll do a rough approach:
        self._trigger_offset: int = 0 # only support 0 (no pre-trigger or delay)
        self._record_length: int = None
        self._records_per_buffer = None     
        self._buffers_per_acquisition = None
        self._buffers_allocated = 1

        self._timestamps_enabled = False

        # NI Tasks & state:
        self._task = None
        self._ready = False # Initial start command will set ready flag to true
        self._started = False
        self._samples_acquired = 0

    @property
    def trigger_delay(self) -> int:
        return self._trigger_offset
    
    @trigger_delay.setter
    def trigger_delay(self, offset: int):
        if int(offset) != 0:
            raise ValueError("No trigger offset (pre-trigger/delay) is supported for NI DAQ")
    
    @property
    def trigger_delay_range(self) -> units.IntRange:
        return units.IntRange(min=0, max=0)
    
    @property
    def pre_trigger_resolution(self):
        return None # no pre-trigger with NI

    @property
    def post_trigger_resolution(self):
        return None # no trigger delay with NI
    
    @property
    def record_length(self) -> int:
        return self._record_length

    @record_length.setter
    def record_length(self, length: int):
        self._record_length = length

    @property
    def record_length_minimum(self) -> int:
        return 1

    @property
    def record_length_step(self) -> int:
        if self._input_mode == digitizer.InputMode.ANALOG:
            return 32
        elif self._input_mode == digitizer.InputMode.EDGE_COUNTING:
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
        if not self._ready:
            self._ready = True
            # The next time start() method is called, this will actually run
            return 
        
        if self._started:
            return  # Already started
        
        if self._sample_clock.edge == digitizer.SampleClockEdge.RISING:
            edge = Edge.RISING 
        else:
            edge = Edge.FALLING

        if self._buffers_per_acquisition == 1:
            sample_mode = AcquisitionType.FINITE
            samples_per_chan = self._record_length
        else:
            sample_mode = AcquisitionType.CONTINUOUS
            samples_per_chan = 2 * self.records_per_buffer * self.record_length 

        if self._input_mode == digitizer.InputMode.ANALOG:

            self._task = nidaqmx.Task("Analog input")

            for channel in self._channels:
                if not channel.enabled:
                    continue
                
                ai_channel = self._task.ai_channels.add_ai_voltage_chan(
                    physical_channel=channel.channel_name,
                    min_val=channel.input_range.min,
                    max_val=channel.input_range.max,
                )
                ai_channel.ai_coupling = channel._coupling # _coupling has the NI enumeration; coupling has Dirigo enumeration
                if self._device.product_category == ProductCategory.X_SERIES_DAQ:
                    ai_channel.ai_term_cfg = TerminalConfiguration.RSE
                elif self._device.product_category == ProductCategory.S_SERIES_DAQ:
                    # S series only supports pseudo-differential
                    ai_channel.ai_term_cfg = TerminalConfiguration.PSEUDO_DIFF

            # Configure the sample clock
            if self._sample_clock.source == digitizer.SampleClockSource.INTERNAL:
                source = None # This will enable use of the built-in analog input clock
            elif self._sample_clock.source == digitizer.SampleClockSource.EXTERNAL:
                source = self._sample_clock._external_clock_channel

            self._task.timing.cfg_samp_clk_timing(
                rate            = self._sample_clock.rate, 
                source          = source, 
                active_edge     = edge,
                sample_mode     = sample_mode,
                samps_per_chan  = samples_per_chan*2 #TODO, not sure about 2x
            )

            # Set up stream reader
            self._reader = AnalogUnscaledReader(self._task.in_stream)

            # Make a preallocated array
            shp = (self.n_channels_enabled,
                   self.records_per_buffer * self.record_length)
            self._prealloc = np.zeros(shape=shp, dtype=np.int16)
        
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
                if self._sample_clock.source is digitizer.SampleClockSource.INTERNAL:
                    source = "/" + self._device.name + "/ao/SampleClock"
                else:
                    source = self._sample_clock.source

                task.timing.cfg_samp_clk_timing(
                    rate=self._sample_clock.rate, 
                    source=source,
                    active_edge=edge,
                    sample_mode=sample_mode,
                    samps_per_chan=samples_per_chan*4 # TODO not sure about 4x?
                )

                reader = CounterReader(task.in_stream)

                self._tasks.append(task)
                self._readers.append(reader)

            self._last_samples = np.zeros(
                shape = (1, self.n_channels_enabled),
                dtype = np.uint32
            )

        self._inverted_vector = np.array(
            [-2*channel.inverted+1 for channel in self._channels if channel.enabled],
            dtype=np.int8
        )

        # Start the task(s)
        if self._input_mode == digitizer.InputMode.ANALOG:
            self._task.start()
        else:
            for task in self._tasks:
                task.start()

        self._started = True
        self._buffers_acquired = 0

    @property
    def buffers_acquired(self) -> int:
        return self._buffers_acquired

    def get_next_completed_buffer(self, acq_product: AcquisitionProduct):
        """
        Reads the next chunk of data from the device buffer. For NI, this typically 
        means calling read once we have enough samples. 
        """
        if not self._started:
            raise RuntimeError("Acquisition not started.")
        
        if self._input_mode == digitizer.InputMode.ANALOG:
            Ny, Ns, Nc = acq_product.data.shape
            self._reader.read_int16(
                data=self._prealloc,
                number_of_samples_per_channel=Ny*Ns
            )
            acq_product.data[...] = np.moveaxis(self._prealloc.reshape(Nc, Ny, Ns), 0, -1)

        else: # Edge counting mode
            # Decide how many samples to read each time. 
            nsamples = self.records_per_buffer * self.record_length

            data_single_channel = np.zeros((nsamples,), dtype=np.uint32) # reader only supports reading into contiguous array
            data_multiple_channels = np.zeros(
                shape=(nsamples+1, self.n_channels_enabled), #+1 b/c we will do np.diff
                dtype=np.uint32
            )
            
            for i, reader in enumerate(self._readers):
                reader.read_many_sample_uint32(
                    data=data_single_channel,
                    number_of_samples_per_channel=nsamples
                )
                data_multiple_channels[1:,i] = data_single_channel

            # Difference along the samples dim and reorder dims for further processing
            data_multiple_channels[0,:] = self._last_samples # place the last frame's final count
            data = np.diff(data_multiple_channels, axis=0).astype(np.uint16)
            self._last_samples = data_multiple_channels[-1,:]
            data.shape = (self.records_per_buffer, self.record_length, self.n_channels_enabled)
            acq_product.data[...] = data

        self._buffers_acquired += 1

    def stop(self):
        if not self._started:
            return

        # Stop the task(s)
        if self._input_mode == digitizer.InputMode.ANALOG:
            self._task.stop()
            self._task.close()
        else:
            for task in self._tasks:
                task.stop()
                CounterRegistry.free_counter(task.channel_names[0])
                task.close()
                
        self._started = False
        self._ready = False
        self._samples_acquired = 0

    @cached_property
    def _input_mode(self) -> digitizer.InputMode:
        if isinstance(self._channels[0], NIAnalogChannel):
            return digitizer.InputMode.ANALOG
        else:
            return digitizer.InputMode.EDGE_COUNTING


class NIAuxiliaryIO(digitizer.AuxiliaryIO):
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
    Wires together the Channel, SampleClock, Trigger, Acquire, and AuxiliaryIO.
    """
    def __init__(self, 
                 device_name: str = "Dev1",
                 external_clock_channel: str | None = None,
                 **kwargs): 
        self.streaming_mode = digitizer.StreamingMode.CONTINUOUS # Operate as continuous streaming (S-series lacks retriggerable support)

        self._device = get_device(device_name)

        # Get channel names from default profile 
        profile = load_toml(self.PROFILE_LOCATION / "default.toml")
        channel_names: list[str] = [c["channel"] for c in profile["channels"]]

        # Infer channel input mode from profile->channels->channels
        if channel_names[0].split('/')[-1][:2] == "ai":
            # if characters of channel names = "ai" then we are using analog mode
            self.input_mode = digitizer.InputMode.ANALOG
            self.channels = [NIAnalogChannel(self._device, chan) for chan in channel_names]
        else:
            self.input_mode = digitizer.InputMode.EDGE_COUNTING
            self.channels = [NICounterChannel(self._device, chan) for chan in channel_names]

        # Create sample clock
        self.sample_clock = NISampleClock(
            device                  = self._device,
            external_clock_channel  = external_clock_channel, 
            channels                = self.channels
        )

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
        self.aux_io = NIAuxiliaryIO(device=self._device)

    @cached_property
    def data_range(self) -> units.IntRange:
        """Range of the returned data."""
        if self.input_mode == digitizer.InputMode.ANALOG:
            # Make dummy task to get at the .ai_resolution property
            with nidaqmx.Task("AI dummy") as task:
                channel = task.ai_channels.add_ai_voltage_chan(
                    physical_channel=self._device.ai_physical_chans.channel_names[0]
                )
                N = int(channel.ai_resolution)
            return units.IntRange(min=-2**N//2, max=2**N//2 - 1)
        else:
            # For edge counting, use uint8 (max 256 edges/photons per pixel)
            # technically the counters support up to 32 bits, but it's unlikely
            # anyone will need this range
            return units.IntRange(min=0, max=2**8 - 1)
    
