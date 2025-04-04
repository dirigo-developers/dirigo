import numpy as np
from functools import cached_property

import nidaqmx
import nidaqmx.system
from nidaqmx.stream_readers import AnalogUnscaledReader
from nidaqmx.constants import (
    ProductCategory, Coupling, Edge, AcquisitionType
)

from dirigo import units
from dirigo.components.io import load_toml
from dirigo.hw_interfaces import digitizer
from dirigo.sw_interfaces.acquisition import AcquisitionBuffer
from dirigo.plugins.scanners import get_device, validate_ni_channel, get_max_ao_rate



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



class NIChannel(digitizer.Channel):
    """
    Represents a single analog input channel on an NI board.
    Implements the Channel interface with minimal NI-specific constraints.
    """

    def __init__(self, device: nidaqmx.system.Device, channel_index: int):
        """
        device_name: e.g. "Dev1"
        index: channel index, used to build the full physical channel string, e.g. "Dev1/ai0".
        """
        self._device = device
        # TODO check valid index
        self._index = channel_index

        # For demonstration, we’ll assume NI boards are DC-coupled only. 
        self._coupling: Coupling = None 
        self._impedance: float = None  # Not adjustable on most boards
        self._range: tuple[float,float] = None
        self._enabled = False

    @property
    def index(self) -> int:
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
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, state: bool):
        self._enabled = state

    @property
    def physical_channel_name(self) -> str:
        """Returns the NI physical channel name, e.g. Dev1/ai0"""
        return f"{self._device.name}/ai{self._index}"


class NISampleClock(digitizer.SampleClock):
    """
    Configures the NI sample clock. 
    For many NI boards, the typical usage is:
       - source = "OnboardClock" or e.g. "PFI0"
       - rate = up to the board’s max sampling rate
       - edge = "rising" or "falling"
    """

    def __init__(self, device: nidaqmx.system.Device, channels: list[NIChannel]):
        self._device = device
        self._channels = channels

        self._source = "/Dev1/ao/SampleClock" #TODO fix this
        self._rate = None #TODO fix this
        self._edge = "rising"

    @property
    def source(self) -> str:
        return self._source

    @source.setter
    def source(self, source: str):
        self._source = validate_ni_channel(source)

    @property
    def source_options(self) -> set[str]:
        # This is device-dependent; an example set
        return {"/Dev1/ao/SampleClock", "PFI0", "PFI1"}

    @property
    def rate(self) -> units.SampleRate:
        return units.SampleRate(self._rate)

    @rate.setter
    def rate(self, value: units.SampleRate):
        if not isinstance(value, units.SampleRate):
            value = units.SampleRate(value)
        if not self.rate_options.within_range(value):
            raise ValueError(
                f"Requested AI sample rate ({value}) is outside "
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
        nchannels_enabled = sum([channel.enabled for channel in self._channels])
        return units.SampleRateRange(
            min=get_min_ai_rate(self._device), 
            max=get_max_ai_rate(self._device, nchannels_enabled)
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
                 channels: list[NIChannel], trigger: NITrigger):
        self._device = device
        self._channels: list[NIChannel] = channels
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
        """
        Creates and configures the NI task, then starts sampling.
        For a typical “finite” acquisition (n samples) scenario, we do a 
        one-shot read. For a “continuous” scenario, you might configure 
        sample mode differently and read in a loop or with callbacks.
        """
        if self._started:
            return  # Already started

        self._task = nidaqmx.Task("Analog input")

        for channel in self._channels:
            if not channel.enabled:
                continue

            self._task.ai_channels.add_ai_voltage_chan(
                physical_channel=channel.physical_channel_name,
                min_val=channel.range.min,
                max_val=channel.range.max,
            )

        # If no channels are enabled, raise an error
        if self.n_channels_enabled < 1:
            raise RuntimeError("No NI channels enabled for acquisition.")

        # Configure the sample clock
        if self._sample_clock.edge == "rising":
            edge = Edge.RISING 
        else:
            edge = Edge.FALLING

        if self._buffers_per_acquisition == 1:
            sample_mode = AcquisitionType.FINITE
            samples_per_chan = self._record_length
        else:
            sample_mode = AcquisitionType.CONTINUOUS
            samples_per_chan = 2 * self._record_length

        self._task.timing.cfg_samp_clk_timing(
            rate=self._sample_clock.rate, 
            source=self._sample_clock.source,
            active_edge=edge,
            sample_mode=sample_mode,
            samps_per_chan=samples_per_chan
        )

        # Configure trigger
        if self._trigger.slope == "rising":
            edge = Edge.RISING 
        else:
            edge = Edge.FALLING

        self._task.triggers.start_trigger.cfg_dig_edge_start_trig(
            trigger_source=self._trigger.source,
            trigger_edge=edge
        )
        # For analog triggers or reference triggers, use cfg_anlg_edge_start_trig

        # Set up stream reader
        self._reader = AnalogUnscaledReader(self._task.in_stream)

        # Start the task
        self._task.start()

        self._started = True
        self._buffers_acquired = 0

    @property
    def buffers_acquired(self) -> int:
        # For a simpler approach, let’s store how many times we read
        return self._buffers_acquired

    def get_next_completed_buffer(self, blocking: bool = True) -> AcquisitionBuffer:
        """
        Reads the next chunk of data from the device buffer. For NI, this typically 
        means calling read once we have enough samples. 
        """
        print("getting buffer data")
        if not self._started:
            raise RuntimeError("Acquisition not started.")
        
        # Decide how many samples to read each time. 
        nsamples = self.records_per_buffer * self.record_length 

        data = np.zeros((self.n_channels_enabled, nsamples), dtype=np.uint16) # TODO use preallocated array?
        self._reader.read_uint16(
            data=data,
            number_of_samples_per_channel=nsamples
        )
        self._buffers_acquired += 1

        # The buffer dimensions need to be reordered for further processing
        data.shape = (self.n_channels_enabled, self.records_per_buffer, self.record_length)
        data = np.transpose(data, axes=(1,2,0)) # Move channels dimension from major to minor (ie interleaved)

        # Construct an AcquisitionBuffer. NI doesn’t provide built-in timestamps
        return AcquisitionBuffer(data=data)

    def stop(self):
        if self._task and self._started:
            self._task.stop()
            self._task.close()
        
        self._task = None
        self._started = False
        self._samples_acquired = 0


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
    VALID_MODES = {"analog", "edge counting"}

    def __init__(self, 
                 device_name: str = "Dev1", 
                 mode: str = "analog",
                 **kwargs): 
        self._device = get_device(device_name)

        if mode not in self.VALID_MODES:
            raise ValueError(f"Mode must be one of {self.VALID_MODES}, got {mode}")
        self._mode = mode

        # Get number of channels from default profile 
        # (NI cards have lots of AI channels, so avoid instantiating all of them)
        profile = load_toml(self.PROFILE_LOCATION / "default.toml")
        n_analog_channels = len(profile["channels"]["enabled"])

        # Create channel objects
        self.channels = [NIChannel(self._device, idx) for idx in range(n_analog_channels)]

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

    @property
    def bit_depth(self) -> int:
        """
        AI bit depth. For all X series cards, this is 16.
        """
        if self._device.product_category == ProductCategory.X_SERIES_DAQ:
            return 16
        else:
            raise RuntimeError(
                "Unsupported NI DAQ device. Only supports X-series, got "
                + str(self._device.product_category)
            )

    @property
    def data_range(self) -> units.ValueRange:
        """Range of the returned data."""
        if self._mode == "analog":
            # For analog mode, we use AnalogUnscaledReader.read_uint16()
            return units.ValueRange(min=0, max=2**16 - 1)
        else:
            # For edge counting
            return units.ValueRange(min=0, max=2**32 - 1)
    

# for testing
if __name__ == "__main__":
    # device = get_device("Dev1")
    # channels = [NIChannel(device, i) for i in range(5)]
    # channels[2].range = units.VoltageRange('-1 V', '1 V')
    # print(channels[2].range)

    digitizer = NIDigitizer("Dev1", n_analog_channels=2)

