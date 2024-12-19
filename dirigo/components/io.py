from dataclasses import dataclass, fields, MISSING
from pathlib import Path
import tomllib

from pint import UnitRegistry, UndefinedUnitError, DimensionalityError
import numpy as np



def load_toml(file_name:Path|str) -> dict:
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"Can not find TOML file: {file_name}")
    if file_name.suffix != ".toml":
        raise ValueError(f"Requested to load a non-TOML file: {file_name}")
    with open(file_name, mode="rb") as toml_file:
        toml_contents = tomllib.load(toml_file)
    return toml_contents


ureg = UnitRegistry(autoconvert_offset_to_baseunit=True) # allows use of logarithm units such as dB
def to_si_units(quantity_with_units:str):
    """ 
    Converts quantity with units, represented in a string, to SI base units. 
    """
    q = ureg.Quantity(quantity_with_units)
    return q.to_base_units().magnitude


def parse_quantity(value: str):
    """Attempts to parse a string with units into a Quantity."""
    try:
        print(value)
        quantity = ureg.Quantity(value)  # Try to parse the string
        print(quantity)
        return quantity
    except (TypeError, ValueError, UndefinedUnitError, DimensionalityError) as e:
        return None


class UnitAwareDataclass:
    @classmethod
    def parse_from_dict(cls, data: dict) -> 'UnitAwareDataclass':
        init_args = {} # dict that will generate the dataclass after parsing
        for f in fields(cls):
            value = data.get(f.name, MISSING)
            if value is not MISSING:

                if type(value) is not list:
                    # if this is not a list, then try parsing
                    q = parse_quantity(value)

                    if q is not None:
                        init_args[f.name] = to_si_units(value)
                    else:
                        init_args[f.name] = value

                else:
                    # if this is a list, try parsing each element
                    for i,element in enumerate(value):
                        e = parse_quantity(element)

                        if e is not None:
                            value[i] = to_si_units(element)

                    init_args[f.name] = value

        return cls(**init_args)   


@dataclass
class LoggerConfig: # no units
    type: str
    save_path: str


@dataclass
class LaserConfig(UnitAwareDataclass):
    pulsed: bool
    pulse_frequency: float


@dataclass
class ScannerRange(UnitAwareDataclass):
    voltage_fast: list[float]
    voltage_slow: list[float]
    angle_fast: float
    angle_slow: float

    @property
    def voltage_fast_min(self):
        return self.voltage_fast[0]
    
    @property
    def voltage_fast_max(self):
        return self.voltage_fast[1]
    
    @property
    def voltage_slow_min(self):
        return self.voltage_slow[0]
    
    @property
    def voltage_slow_max(self):
        return self.voltage_slow[1]
    
    @property
    def volts_per_optical_degree_fast(self):
        return self.voltage_fast_max / self.angle_fast
    
    @property
    def volts_per_optical_degree_slow(self):
        vrange = self.voltage_slow_max - self.voltage_slow_min
        return vrange / self.angle_slow


@dataclass
class ScannerOptics(UnitAwareDataclass):
    relay_mag: int
    objective_fl: float

    def angle_to_position(self, scan_angle):
        """ Converts scan angle to position """
        angle_at_objective = scan_angle / self.relay_mag
        position = np.sin(angle_at_objective) * self.objective_fl
        return position


@dataclass
class ScannerWiring: # no units
    fast_scanner_signal_out: str
    slow_scanner_signal_out: str
    fast_scanner_sync_in: str
    frame_clock_out: str
    frame_clock_in: str


@dataclass
class ScannerConfig(UnitAwareDataclass):
    fast_axis: str
    nominal_scanner_frequency: float
    flip_fast: bool
    flip_slow: bool
    scan_range: ScannerRange
    optics: ScannerOptics 
    wiring: ScannerWiring 
    daq_sample_rate: float
    fast_scanner_settle_time: float
    slow_scanner_response_time: float

    def _voltage_to_scan_width(self, voltage):
        vpod = self.scan_range.volts_per_optical_degree_fast
        angle = voltage / vpod
        return 2*self.optics.angle_to_position(angle/2)

    @property
    def fast_width_max(self):
        voltage = self.scan_range.voltage_fast_max
        return self._voltage_to_scan_width(self, voltage)
    
    @property
    def fast_width_min(self):
        voltage = self.scan_range.voltage_fast_min
        return self._voltage_to_scan_width(self, voltage)


@dataclass
class DigitizerConfig(UnitAwareDataclass):
    manufacturer: str
    model: str
    nbuffers: int
    clock_source: str
    sample_rate: float
    clock_edge: str
    input_coupling: list[str]
    input_range: list[float]
    input_impedance: list[int]
    trigger_source: str
    trigger_slope: str
    trigger_level: float
    external_trigger_coupling: str
    external_trigger_range: str
    aux_io_mode: str
    aux_io_parameter: int


@dataclass
class StageConfig(UnitAwareDataclass):
    manufacturer: str
    model: str


@dataclass
class SystemConfig:
    logger: LoggerConfig
    laser: LaserConfig
    scanner: ScannerConfig
    digitizer: dict
    stage: dict
    fast_raster_scanner: dict # It may be better to make these dicts and have the plugin determine what is needed

    @classmethod
    def from_toml(cls, toml_path: Path) -> 'SystemConfig':
        data = load_toml(toml_path)
        return cls(
            logger=LoggerConfig(**data.get("logger", {})),
            laser=LaserConfig.parse_from_dict(data["laser"]),
            scanner=ScannerConfig(
                **{k: v for k, v in data.get("scanner", {}).items() if k not in ("scan_range", "optics", "wiring")},
                scan_range=ScannerRange.parse_from_dict(data["scanner"]["scan_range"]),
                optics=ScannerOptics.parse_from_dict(data["scanner"]["optics"]),
                wiring=ScannerWiring(**data["scanner"].get("wiring", {}))
            ),
            digitizer=data["digitizer"],
            stage=data["stage"],
            fast_raster_scanner=data["fast_raster_scanner"],
        )