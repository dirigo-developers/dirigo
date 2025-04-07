from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Optional



def load_toml(file_name: Path | str) -> dict:
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
    """
    Simple data class to hold system configuration categories.
    
    All fields are technically optional.
    """
    laser_scanning_optics: Optional[dict] = None
    camera_optics: Optional[dict] = None
    digitizer: Optional[dict] = None
    stage: Optional[dict] = None
    objective_scanner: Optional[dict] = None
    encoders: Optional[dict] = None
    fast_raster_scanner: Optional[dict] = None 
    slow_raster_scanner: Optional[dict] = None
    frame_grabber: Optional[dict] = None
    line_scan_camera: Optional[dict] = None
    illuminator: Optional[dict] = None

    @classmethod
    def from_toml(cls, toml_path: Path) -> 'SystemConfig':
        toml_data = load_toml(toml_path)
        return cls(
            laser_scanning_optics=toml_data.get("laser_scanning_optics"),
            camera_optics=toml_data.get("camera_optics"),
            digitizer=toml_data.get("digitizer"),
            stage=toml_data.get("stage"),
            objective_scanner=toml_data.get("objective_scanner"),
            encoders=toml_data.get("encoders"),
            fast_raster_scanner=toml_data.get("fast_raster_scanner"),
            slow_raster_scanner=toml_data.get("slow_raster_scanner"),
            frame_grabber=toml_data.get("frame_grabber"),
            line_scan_camera=toml_data.get("line_scan_camera"),
            illuminator=toml_data.get("illuminator")
        )