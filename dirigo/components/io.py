from dataclasses import dataclass, asdict
from pathlib import Path
import tomllib
from typing import Optional

import numpy as np
import tifffile
from numpy.polynomial.polynomial import Polynomial
import platformdirs as pd

from dirigo.components import units



def config_path() -> Path:
    return pd.user_config_path("Dirigo")


def load_toml(file_name: Path | str) -> dict:
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"Can not find TOML file: {file_name}")
    if file_name.suffix != ".toml":
        raise ValueError(f"Requested to load a non-TOML file: {file_name}")
    with open(file_name, mode="rb") as toml_file:
        toml_contents = tomllib.load(toml_file)
    return toml_contents


def load_scanner_calibration(
        path: Path = config_path() / "scanner/calibration.csv"
        ) -> tuple:
    
    ampls, freqs, phases = np.loadtxt(
        path,
        delimiter=',',
        unpack=True,
        skiprows=1
    )
    return ampls, freqs, phases


def load_distortion_calibration(
        amplitude: units.Angle,
        path: Path = config_path() / "optics/distortion_calibration.csv"
):
    data = np.loadtxt(path, delimiter=',', dtype=np.float64, skiprows=1, ndmin=2)
    amplitudes = data[:,0]
    coefs = data[:,1:]

    for i,a in enumerate(amplitudes):
        if abs(a - amplitude)/amplitude < 0.001:
            return Polynomial(coefs[i])
    
    raise RuntimeError("Could not find distortion calibration")


def load_gradient_calibration(
        path: Path = config_path() / "optics/gradient_calibration.tif"
):
    return tifffile.imread(path)


def load_line_width_calibration(
        path: Path = config_path() / "scanner/line_width_calibration.csv",
        fit_deg: int = 3) -> Polynomial:
    
    amplitudes, widths =  np.loadtxt(
        path,
        delimiter=',',
        unpack=True,
        skiprows=1
    )
    return Polynomial.fit(x=widths, y=amplitudes, deg=fit_deg)


@dataclass
class SystemConfig:
    """
    Simple data class to hold system configuration categories.
    
    All fields are technically optional.
    """
    laser_scanning_optics: Optional[dict] = None
    camera_optics: Optional[dict] = None
    detectors: Optional[dict] = None
    digitizer: Optional[dict] = None
    stages: Optional[dict] = None
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
            detectors=toml_data.get("detectors"),
            digitizer=toml_data.get("digitizer"),
            stages=toml_data.get("stages"),
            objective_scanner=toml_data.get("objective_scanner"),
            encoders=toml_data.get("encoders"),
            fast_raster_scanner=toml_data.get("fast_raster_scanner"),
            slow_raster_scanner=toml_data.get("slow_raster_scanner"),
            frame_grabber=toml_data.get("frame_grabber"),
            line_scan_camera=toml_data.get("line_scan_camera"),
            illuminator=toml_data.get("illuminator")
        )
    
    def to_dict(self):
        return asdict(self)
    
