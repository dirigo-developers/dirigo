from atsbindings import Ats, System, Buffer
from atsbindings import Board as AlazarBoard

from dirigo.plugin_registry import PluginRegistry
from dirigo.components.digitizer import (
    Digitizer, Channel, SampleClock, Trigger, Acquire, AuxillaryIO, ValidQuantityRange
)



class AlazarChannel(Channel):
    def __init__(self, board:AlazarBoard, channel_index):
        self._board = board
        self._index = channel_index
        self._enabled = False # Disable by default
        # Set parameters to None to signify that they have not been initialized
        self._coupling:Ats.Couplings = None
        self._impedance:Ats.Impedances = None
        self._range:Ats.InputRanges = None
    
    @property
    def index(self):
        """Retrieve this channel's index"""
        return self._index

    @property
    def coupling(self):
        if self._coupling:
            return str(self._coupling)
    
    @coupling.setter
    def coupling(self, coupling:str):
        coupling_enum = Ats.Couplings.from_str(coupling)
        if coupling_enum not in self._board.bsi.input_couplings:
            valid = self._board.bsi.input_couplings
            raise ValueError(f"Invalid input coupling {coupling_enum}"
                             f"Valid options are: {valid}")
        self._coupling = coupling_enum
        self._set_input_control()

    @property
    def coupling_options(self):
        options = self._board.bsi.input_couplings
        return [str(s) for s in options]

    @property
    def impedance(self):
        if self._impedance:
            return str(self._impedance)
    
    @impedance.setter
    def impedance(self, impedance):
        impedance_enum = Ats.Impedances.from_str(impedance)
        if impedance_enum not in self._board.bsi.input_impedances:
            valid_options = ', '.join([str(s) for s in self._board.bsi.input_impedances])
            raise ValueError(f"Invalid input impedance {impedance_enum}. "
                             f"Valid options are: {valid_options}")
        self._impedance = impedance_enum
        self._set_input_control()

    @property
    def impedance_options(self):
        options = self._board.bsi.input_impedances
        return [str(s) for s in options]
    
    @property
    def range(self):
        if self._range:
            return str(self._range)
    
    @range.setter
    def range(self, rng):
        range_enum = Ats.InputRanges.from_str(rng)
        current_ranges = self._board.bsi.input_ranges(self._impedance)
        if range_enum not in current_ranges:
            valid_options = ', '.join([str(s) for s in current_ranges])
            raise ValueError(f"Invalid input impedance {range_enum}. "
                             f"Valid options are: {valid_options}")
        self._range = range_enum
        self._set_input_control()
    
    @property
    def range_options(self):
        options = self._board.bsi.input_ranges(self._impedance)
        return [str(s) for s in options]

    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, enabled:bool):
        self._enabled = enabled

    def _set_input_control(self):
        """
        Helper method to set ...
        """
        if self.coupling and self.impedance and self._range:
            self._board.input_control_ex(
                channel=Ats.Channels.from_int(self.index),
                coupling=self._coupling,
                input_range=self._range,
                impedance=self._impedance,
            )


class AlazarSampleClock(SampleClock):
    def __init__(self, board:AlazarBoard):
        self._board = board
        # Set parameters to None to signify that they have not been initialized
        self._source:Ats.ClockSources = None
        self._rate:Ats.SampleRates = None
        self._external_rate:float = None # Used only with external clocking, in Hz
        self._edge:Ats.ClockEdges = Ats.ClockEdges.CLOCK_EDGE_RISING
        
    @property
    def source(self):
        if self._source:
            return str(self._source)
    
    @source.setter
    def source(self, source:str):
        previous_source_enum = self._source
        source_enum = Ats.ClockSources.from_str(source)

        # Check if same, if so return immediately
        if source_enum == previous_source_enum:
            return 
        
        # Check whether selection is supported, if so make it official
        if source_enum not in self._board.bsi.supported_clocks:
            valid_options = ', '.join([str(s) for s in self._board.bsi.supported_clocks])
            raise ValueError(f"Invalid sample clock source: {source_enum}. "
                             f"Valid options are: {valid_options}")
        self._source = source_enum

        # Reset rate
        if source_enum == Ats.ClockSources.INTERNAL_CLOCK:
            self._rate = None
        elif "external" in str(self._source).lower():
            self._rate = Ats.SampleRates.SAMPLE_RATE_USER_DEF
            self._external_rate = None

        #self._set_capture_clock()

    @property
    def source_options(self):
        return [str(s) for s in self._board.bsi.supported_clocks]

    @property
    def rate(self):
        """
        Depending on the sample clock source returns either the internal sample
        clock rate, or the user-specified rate.
        """
        if self._source == Ats.ClockSources.INTERNAL_CLOCK:
            if self._rate:
                return str(self._rate)
        elif "external" in str(self._source).lower():
            if self._external_rate:
                return str(self._external_rate) 
    
    @rate.setter
    def rate(self, rate:str):
        """
        Depending on the sample clock source, sets the rate from one of the
        board's specified internal rates, or stores a user-specified value
        """
        if self._source == Ats.ClockSources.INTERNAL_CLOCK:
            clock_rate_enum = Ats.SampleRates.from_str(rate)
            if clock_rate_enum not in self._board.bsi.sample_rates:
                valid_options = ', '.join([str(s) for s in self._board.bsi.sample_rates])
                raise ValueError(f"Invalid sample clock rate: {clock_rate_enum}. "
                                f"Valid options are: {valid_options}")
            self._rate = clock_rate_enum
            self._set_capture_clock()
        elif "external" in str(self._source).lower():
            # check that the proposed external clock is valid
            proposed_rate = float(rate)
            valid_range = self._board.bsi.external_clock_frequency_ranges(self._source)
            if valid_range.min < proposed_rate < valid_range.max:
                self._external_rate = proposed_rate
            else:
                raise ValueError(f"Tried setting external clock frequency outside "
                                 f"acceptable range for source: {self._source} "
                                 f"Requested: {proposed_rate}, "
                                 f"Min: {valid_range.min}"
                                 f"Max: {valid_range.max}")

    @property
    def rate_options(self) -> list[str] | ValidQuantityRange:
        if self._source == Ats.ClockSources.INTERNAL_CLOCK:
            return [str(option) for option in self._board.bsi.sample_rates]
        elif "external" in str(self._source).lower():
            return self._board.bsi.external_clock_frequency_ranges(self._source)
    
    @property
    def edge(self):
        if self._edge:
            return str(self._edge)
    
    @edge.setter
    def edge(self, edge:str):
        clock_edge_enum = Ats.ClockEdges.from_str(edge)
        self._edge = clock_edge_enum
        self._set_capture_clock()

    @property
    def edge_options(self):
        options = [Ats.ClockEdges.CLOCK_EDGE_RISING, # ALl non-DES boards support rising/falling edge sampling
                   Ats.ClockEdges.CLOCK_EDGE_FALLING]
        return [str(s) for s in options]

    def _set_capture_clock(self):
        """
        Helper to set capture clock if all required parameters have been set:
        source, rate, and edge
        """
        if self._source and self._rate and self._edge:
            self._board.set_capture_clock(self._source, self._rate, self._edge)


class AlazarTrigger(Trigger):
    def __init__(self, board:AlazarBoard, channels:list[AlazarChannel]):
        self._board = board
        self._channels = channels
        # Set parameters to None to signify that they have not been initialized
        self._source:Ats.TriggerSources = None
        self._slope:Ats.TriggerSlopes = None
        self._external_coupling:Ats.Couplings = None
        self._external_range:Ats.ExternalTriggerRanges = None
        self._level:int = None

    @property
    def source(self):
        if self._source:
            return str(self._source)
    
    @source.setter
    def source(self, source:str):
        source_enum = Ats.TriggerSources.from_str(source)
        trig_srcs = self._board.bsi.supported_trigger_sources
        if source_enum not in trig_srcs:
            valid_options = ', '.join([str(s) for s in trig_srcs])
            raise ValueError(f"Invalid trigger source: {source_enum}. "
                             f"Valid options are: {valid_options}")
        self._source = source_enum
        self._set_trigger_operation()

    @property
    def source_options(self):
        options = [str(s) for s in self._board.bsi.supported_trigger_sources]
        # may want to remove 'Disable' option, which would require SW trigger
        return sorted(options)

    @property
    def slope(self):
        if self._slope:
            return str(self._slope)
    
    @slope.setter
    def slope(self, slope:str):
        self._slope = Ats.TriggerSlopes.from_str(slope)
        self._set_trigger_operation()

    @property
    def slope_options(self):
        options = [Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
                   Ats.TriggerSlopes.TRIGGER_SLOPE_NEGATIVE]
        return [str(s) for s in options]

    @property
    def level(self) -> float:
        """Returns the current trigger level in volts""" #TODO does this work for external trigger?
        if self.source and self._level:
            trigger_source_range = self._channels[self._source.channel_index]._range.to_volts
            return (self._level - 128) * trigger_source_range / 127

    @level.setter
    def level(self, level:float):
        if not self._source:
            raise RuntimeError("Trigger source must be set before trigger level")
        if self._source == Ats.TriggerSources.TRIG_DISABLE:
            raise RuntimeError("Cannot set trigger level. Trigger is disabled")
        if self._source == Ats.TriggerSources.TRIG_EXTERNAL:
            trigger_source_range = self._external_range.to_volts 
        else:
            trigger_source_range = self._channels[self._source.channel_index]._range.to_volts
        if abs(level) > trigger_source_range:
            raise ValueError(f"Trigger level, {level} is outside the current trigger source range")

        self._level = int(128 + 127 * level / trigger_source_range)
        self._set_trigger_operation()

    @property
    def level_min(self): # TODO check range for TTL ext trig
        if not self.source:
            return None
        
        if self._source == Ats.TriggerSources.TRIG_EXTERNAL:
            trigger_source_range = self._external_range.to_volts 
        elif self._source in [Ats.TriggerSources.from_str(f"Channel {chr(i + ord('A'))}") for i in range(len(self._channels))]:
            trigger_source_range = self._channels[self._source.channel_index]._range.to_volts
        return -abs(trigger_source_range)

    @property
    def level_max(self):
        if self.source:
            return abs(self.level_min)

    @property
    def external_coupling(self):
        return str(self._external_coupling)

    @external_coupling.setter
    def external_coupling(self, external_coupling:str):
        external_coupling_enum = Ats.Couplings.from_str(external_coupling)
        self._external_coupling = external_coupling_enum
        self._set_external_trigger()

    @property
    def external_coupling_options(self):
        # Only support DC external trigger. Only a few old boards support AC.
        options = [Ats.Couplings.DC_COUPLING]
        return [str(s) for s in options]

    @property
    def external_range(self):
        return str(self._external_range)
    
    @external_range.setter
    def external_range(self, external_range:str):
        external_range_enum = Ats.ExternalTriggerRanges.from_str(external_range)
        supported_ranges = self._board.bsi.external_trigger_ranges
        if external_range_enum not in supported_ranges:
            valid_options = ', '.join([str(s) for s in supported_ranges])
            raise ValueError(f"Invalid trigger source: {external_range_enum}. "
                             f"Valid options are: {valid_options}")
        self._external_range = external_range_enum
        self._set_external_trigger()

    @property
    def external_range_options(self):
        options = self._board.bsi.external_trigger_ranges
        return [str(s) for s in options]
        
    def _set_trigger_operation(self):
        """
        Helper to set trigger operation if all required parameters have been set.
        By default, uses trigger engine J and disables engine K
        """
        if self._source and self._slope and self._level:
            self._board.set_trigger_operation(
                operation=Ats.TriggerOperations.TRIG_ENGINE_OP_J,
                engine1=Ats.TriggerEngines.TRIG_ENGINE_J,
                source1=self._source,
                slope1=self._slope,
                level1=self._level,
                engine2=Ats.TriggerEngines.TRIG_ENGINE_K,
                source2=Ats.TriggerSources.TRIG_DISABLE,
                slope2=Ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
                level2=0
            )

    def _set_external_trigger(self):
        """
        Helper ...
        """
        if self._external_coupling and self._external_range:
            self._board.set_external_trigger(
                self._external_coupling, 
                self._external_range
            )


class AlazarAcquire(Acquire):
    def __init__(self, board:AlazarBoard, channels:list[AlazarChannel]):
        self._board = board
        self._channels = channels

        # Set some defaults
        self._trigger_delay:int = 0
        self._pre_trigger_samples:int = 0

        self._record_length:int = None
        self._records_per_buffer:int = None
        self._buffers_per_acquisition:int = None

        self._adma_mode:Ats.ADMAModes = Ats.ADMAModes.ADMA_TRADITIONAL_MODE

        self._buffers:list[Buffer] = None
        
    @property
    def trigger_delay(self):
        return self._trigger_delay
    
    @property
    def trigger_delay(self, delay:int):
        if delay < 0:
            raise ValueError("Trigger delay must be non-negative.")
        
        dly_res = self.trigger_delay_resolution
        if delay % dly_res != 0:
            raise ValueError(f"Attempted to set trigger delay {delay}, must"
                             f"be multiple of {dly_res}")
        self._trigger_delay = delay
        self._board.set_trigger_delay(self._trigger_delay)

    @property
    def trigger_delay_resolution(self):
        # samples per timestamp resolution is also the resolution for trigger delay
        return self._board.bsi.samples_per_timestamp(self.n_channels_enabled)

    @property
    def pre_trigger_samples(self):
        return self._pre_trigger_samples
    
    @pre_trigger_samples.setter
    def pre_trigger_samples(self, samples:int):
        if samples < 0:
            raise ValueError(f"Attempted to set pre-trigger samples {samples} "
                             f"must be ≥ 0")
        pretrig_res = self.pre_trigger_resolution
        if samples % pretrig_res != 0:
            raise ValueError(f"Attempted to set pre-trigger samples {samples}, "
                             f"must be multiple of {pretrig_res}")
        self._pre_trigger_samples = samples
        self._set_record_size()

    @property
    def pre_trigger_resolution(self):
        return self._board.bsi.pretrig_alignment

    @property
    def record_length(self):
        return self._record_length

    @record_length.setter
    def record_length(self, length):
        if length < self.record_length_minimum:
            raise ValueError(f"Attempted to set record length {length}, must "
                             f"be ≥ {self.record_length_minimum}")
        rec_res = self.record_length_resolution
        if length % rec_res != 0:
            raise ValueError(f"Attempted to set record length {length}, must "
                             f"be multiple of {rec_res}")
        self._record_length = length
        self._set_record_size()

    @property
    def record_length_minimum(self):
        return self._board.bsi.min_record_size
    
    @property
    def record_length_resolution(self):
        return self._board.bsi.record_resolution
    
    @property
    def records_per_buffer(self):
        return self._records_per_buffer
    
    @records_per_buffer.setter
    def records_per_buffer(self, records:int):
        if records < 1:
            ValueError(f"Attempted to set records per buffer {records}, "
                       f"must be ≥ 1")
        self._records_per_buffer = records

    @property
    def buffers_per_acquisition(self):
        return self._buffers_per_acquisition
    
    @buffers_per_acquisition.setter
    def buffers_per_acquisition(self, buffers):
        if buffers < 1:
            ValueError(f"Attempted to set buffers per acquisition {buffers}, "
                       f"must be ≥ 1")
        self._buffers_per_acquisition = buffers

    def start(self):
        # Check whether essential parameters have been set
        if not self.record_length:
            raise RuntimeError("Must set record length before beginning acquisition")
        if not self.records_per_buffer:
            raise RuntimeError("Must set records per buffer before beginning acquisition")
        if not self.buffers_per_acquisition:
            raise RuntimeError("Must set buffers per acquisition before beginning acquisition")
        
        # Prepare board for acquisition
        channels_bit_mask = sum([c.enabled * Ats.Channels.from_int(i) 
                                for i,c in enumerate(self._channels)])
        flags = self._adma_mode | Ats.ADMAFlags.ADMA_EXTERNAL_STARTCAPTURE

        if self.buffers_per_acquisition == float('inf'):
            records_per_acquisition = 0x7FFFFFFF
        else:
            records_per_acquisition = int(
                self.buffers_per_acquisition * self.records_per_buffer
            )

        self._board.before_async_read(
            channels=channels_bit_mask,
            transfer_offset=-self.pre_trigger_samples, # note the neg. value for pre-trigger samples
            samples_per_record=self.record_length,
            records_per_buffer=self.records_per_buffer,
            records_per_acquisition=records_per_acquisition,
            flags=flags
        )

        # Allocate buffers
        self._buffers = []
        for _ in range(self.buffers_per_acquisition): #TODO allocated buffers < buffers per acq
            buffer = Buffer(
                board=self._board, 
                channels=self.n_channels_enabled,
                records_per_buffer=self.records_per_buffer,
                samples_per_record=self.record_length,
                include_header=True,
            )
            self._buffers.append(buffer)
            self._board.post_async_buffer(buffer.address, buffer.size)
        
        self._board.start_capture()

    def stop(self):
        self._board.abort_async_read()

    @property
    def adma_mode(self):
        """Alazar-specific: returns current ADMA mode"""
        return str(self._adma_mode)

    @adma_mode.setter
    def adma_mode(self, new_adma_mode:str):
        """Set ADMA mode (default without using setter is 'Traditional')"""
        self._adma_mode = Ats.ADMAModes.from_str(str(new_adma_mode))

    def _set_record_size(self):
        """Helper"""
        if self._pre_trigger_samples and self._record_length:
            self._board.set_record_size(
                pre_trigger_samples=self._pre_trigger_samples, 
                post_trigger_samples=self._record_length)


class AlazarAuxillaryIO(AuxillaryIO):
    def __init__(self, board:AlazarBoard):
        self._board = board
        self._mode:Ats.AuxIOModes = None

    def configure_mode(self, mode:Ats.AuxIOModes, **kwargs):
        if mode == Ats.AuxIOModes.AUX_OUT_TRIGGER:
            self._board.configure_aux_io(mode, 0)

        elif mode == Ats.AuxIOModes.AUX_OUT_PACER:
            divider = int(kwargs.get('divider'))
            self._board.configure_aux_io(mode, divider)

        elif mode == Ats.AuxIOModes.AUX_OUT_SERIAL_DATA:
            state = bool(kwargs.get('state'))
            self._board.configure_aux_io(mode, state)

        elif mode == Ats.AuxIOModes.AUX_IN_TRIGGER_ENABLE:
            slope:Ats.TriggerSlopes = kwargs.get('slope')
            self._board.configure_aux_io(mode, slope)

        elif mode == Ats.AuxIOModes.AUX_IN_AUXILIARY:
            self._board.configure_aux_io(mode, 0)
            # Note, read requires a call to board.get_parameter()

        else:
            raise ValueError(f"Unsupported auxillary IO mode: {mode}")
        
    def read_input(self) -> bool:
        if self._mode == Ats.AuxIOModes.AUX_IN_AUXILIARY:
            return self._board.get_parameter(
                Ats.Channels.CHANNEL_ALL, 
                Ats.Parameters.GET_AUX_INPUT_LEVEL
            )
        else:
            raise RuntimeError("Auxillary IO not configured as input.")
        
    def write_output(self, state:bool):
        self.configure_mode(Ats.AuxIOModes.AUX_OUT_SERIAL_DATA, state=state)


class AlazarDigitizer(Digitizer):
    """
    Subclass implementing the dirigo Digitizer interface.

    ATSApi has many enumeration which are used internally, but not returned to the end user
    """
    def __init__(self, system_id:int=1, board_id:int=1):
        # Check system
        nsystems = System.num_of_systems()
        if nsystems < 1:
            raise RuntimeError("No board systems found. At least one is required.")
        nboards = System.boards_in_system_by_system_id(system_id)
        if nboards < 1: # not sure this is actually possible 
            raise RuntimeError("No boards found. At least one is required.")
        
        self.driver_version = System.get_driver_version()
        self.dll_version = System.get_sdk_version() # this is sort of a misnomer

        self._board = AlazarBoard(system_id, board_id)

        self.channels = []
        for i in range(self._board.bsi.channels):
            self.channels.append(AlazarChannel(self._board, i))
        self.sample_clock = AlazarSampleClock(self._board)
        self.trigger = AlazarTrigger(self._board, self.channels)
        self.acquire = AlazarAcquire(self._board, self.channels)
        self.aux_io = AlazarAuxillaryIO(self._board)


# Register as a plugin
PluginRegistry.register_plugin(Digitizer, AlazarDigitizer)


# For testing
import time
if __name__ == "__main__":
    digitizer = AlazarDigitizer()


    # Vertical
    for i, channel in enumerate(digitizer.channels):
        channel.coupling = "DC"
        channel.impedance = "50 Ω"
        channel.range = "±2 V"
        channel.enabled = i < 2

    # Horizontal
    digitizer.sample_clock.edge = "Rising"
    digitizer.sample_clock.source = "Internal Clock"
    digitizer.sample_clock.rate = "10 MS/s"

    # Trigger
    digitizer.trigger.external_coupling = "DC"
    digitizer.trigger.external_range = "TTL"
    digitizer.trigger.source = "External"
    digitizer.trigger.slope = "Positive"
    digitizer.trigger.level = 0

    # Acquire
    digitizer.acquire.pre_trigger_samples = 0
    digitizer.acquire.record_length = 2048
    digitizer.acquire.records_per_buffer = 256
    digitizer.acquire.buffers_per_acquisition = 4

    digitizer.acquire.start()
    time.sleep(0.02)
    digitizer.acquire.stop()

    print("success")


