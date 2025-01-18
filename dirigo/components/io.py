from dataclasses import dataclass
from pathlib import Path
import tomllib




def load_toml(file_name:Path|str) -> dict:
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"Can not find TOML file: {file_name}")
    if file_name.suffix != ".toml":
        raise ValueError(f"Requested to load a non-TOML file: {file_name}")
    with open(file_name, mode="rb") as toml_file:
        toml_contents = tomllib.load(toml_file)
    return toml_contents


@dataclass
class SystemConfig:
    optics: dict
    digitizer: dict
    stage: dict
    objective_scanner: dict
    encoders: dict
    fast_raster_scanner: dict # It may be better to make these dicts and have the plugin determine what is needed
    slow_raster_scanner: dict

    @classmethod
    def from_toml(cls, toml_path: Path) -> 'SystemConfig':
        data = load_toml(toml_path)
        return cls(
            optics=data['optics'],
            digitizer=data["digitizer"],
            stage=data["stage"],
            objective_scanner=data["objective_scanner"],
            encoders=data["encoders"],
            fast_raster_scanner=data["fast_raster_scanner"],
            slow_raster_scanner=data["slow_raster_scanner"]
        )