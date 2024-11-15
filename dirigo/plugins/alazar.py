from atsbindings import Ats, System, Board, Buffer

from dirigo.plugin_registry import PluginRegistry
from dirigo.components.digitizer import Digitizer, SampleClock, Trigger, Channel, AuxillaryIO



class AlazarSampleClock(SampleClock):
    def __init__(self, board:Board):
        self._board = board
        # Set parameters to None to signify that they have not been initialized
        self._source:Ats.ClockSources = None
        self._rate:Ats.SampleRates = None
        self._edge:Ats.ClockEdges = Ats.ClockEdges.CLOCK_EDGE_RISING
        
    @property
    def source(self):
        return str(self._source)
    
    @source.setter
    def source(self, source:str):
        source_enum = Ats.ClockSources.from_str(source)
        if source_enum not in self._board.bsi.supported_clocks:
            valid_options = ', '.join([str(s) for s in self._board.bsi.supported_clocks])
            raise ValueError(f"Invalid sample clock source: {source_enum}. "
                             f"Valid options are: {valid_options}")
        self._source = source_enum
        self._set_capture_clock()

    @property
    def rate(self):
        return str(self._rate)
    
    @rate.setter
    def rate(self, rate:str):
        clock_rate_enum = Ats.SampleRates.from_hz(rate)
        if clock_rate_enum not in self._board.bsi.sample_rates:
            valid_options = ', '.join([str(s) for s in self._board.bsi.sample_rates])
            raise ValueError(f"Invalid sample clock rate: {clock_rate_enum}. "
                             f"Valid options are: {valid_options}")
        self._rate = clock_rate_enum
        self._set_capture_clock()

    @property
    def edge(self):
        return str(self._edge)
    
    @edge.setter
    def edge(self, edge:str):
        clock_edge_enum = Ats.ClockEdges.from_str(edge)
        self._edge = clock_edge_enum
        self._set_capture_clock()

    def _set_capture_clock(self):
        """
        Helper to set capture clock if all required parameters have been set:
        source and rate
        """
        if self._source and self._rate:
            self._board.set_capture_clock(self._source, self._rate)


class AlazarChannel(Channel):
    def __init__(self, board:Board, channel_index):
        self._board = board
        self._index = channel_index
        # Set parameters to None to signify that they have not been initialized
        self._coupling:Ats.Couplings = None
        self._impedance:Ats.Impedances = None
        self._range:Ats.InputRanges = None

    @property
    def coupling(self):
        return str(self._coupling)
    
    @coupling.setter
    def coupling(self, coupling:str):
        coupling_enum = Ats.Couplings.from_str(coupling)
        #if coupling_enum not in self._board.bsi. #TODO still needs to be added to BSI
        self._coupling = coupling_enum
        self._set_input_control()

    @property
    def impedance(self):
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
    def range(self):
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

    def _set_input_control(self):
        """
        Helper method to set ...
        """
        if (hasattr(self, '_input_coupling')
            and hasattr(self, '_input_impedance')
            and hasattr(self, '_input_range')
        ):
            self._board.input_control_ex(
                channel=Ats.Channels.from_int(self._index),
                coupling=self._coupling,
                input_range=self._range,
                impedance=self._impedance,
            )


class AlazarTrigger(Trigger):
    def __init__(self, board:Board, channels:list[AlazarChannel]):
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
    def slope(self):
        return str(self._slope)
    
    @slope.setter
    def slope(self, slope:str):
        self._slope = Ats.TriggerSlopes.from_str(slope)
        self._set_trigger_operation()

    @property
    def level(self) -> float:
        """Returns the current trigger level in volts"""
        trigger_source_range = self._channels[self._source.channel_index]._range.to_volts
        return (self._level - 128) * trigger_source_range / 127

    @level.setter
    def level(self, level:float):
        if not self._source:
            raise RuntimeError("Trigger source must be set before trigger level")
        if self._source == Ats.TriggerSources.TRIG_DISABLE:
            return # Raise Error?
        if self._source == Ats.TriggerSources.TRIG_EXTERNAL:
            trigger_source_range = self._external_range.to_volts 
        else:
            trigger_source_range = self._channels[self._source.channel_index]._range.to_volts
        if abs(level) > trigger_source_range:
            raise ValueError(f"Trigger level, {level} is outside the current trigger source range")

        self._level = int(128 + 127 * level / trigger_source_range)
        self._set_trigger_operation()

    @property
    def external_coupling(self):
        return str(self._external_coupling)

    @external_coupling.setter
    def external_coupling(self, external_coupling:str):
        external_coupling_enum = Ats.Couplings.from_str(external_coupling)
        self._external_coupling = external_coupling_enum
        self._set_external_trigger()

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
        
    def _set_trigger_operation(self):
        """
        Helper to set trigger operation if all required parameters have been set.
        By default, uses trigger engine J and disables engine K
        """
        if self._source and self._slope:
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


class AlazarAuxillaryIO(AuxillaryIO):
    def __init__(self, board:Board):
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

        self._board = Board(system_id, board_id)

        self.sample_clock = AlazarSampleClock(self._board)

        self.channels = []
        for i in range(self._board.bsi.channels):
            self.channels.append(AlazarChannel(self._board, i))

        self.trigger = AlazarTrigger(self._board, self.channels)

        self.aux_io = AlazarAuxillaryIO(self._board)


# Register as a plugin
PluginRegistry.register_plugin(Digitizer, AlazarDigitizer)



# For testing
if __name__ == "__main__":
    digitizer = AlazarDigitizer()

    # Set up channels
    # Set up input clock
    # Set up trigger

