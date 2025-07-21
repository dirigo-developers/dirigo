from pathlib import Path
import tomllib
from typing import Optional, Any
from functools import cached_property

import numpy as np
import tifffile
from numpy.polynomial.polynomial import Polynomial
import platformdirs as pd

from dirigo.components import units



def config_path() -> Path:
    return pd.user_config_path("Dirigo")


def load_toml(file_name: Path | str) -> dict[str, Any]:
    file_name = Path(file_name)
    if not file_name.exists():
        raise FileNotFoundError(f"Can not find TOML file: {file_name}")
    if file_name.suffix != ".toml":
        raise ValueError(f"Requested to load a non-TOML file: {file_name}")
    with open(file_name, mode="rb") as toml_file:
        toml_contents = tomllib.load(toml_file)
    return toml_contents


try: 
    d = load_toml(config_path() / "logging.toml")           # 1st choice: user-specified path
    _data_path = Path(d['data_path'])
except:
    _data_path = pd.user_documents_path() / "Dirigo"        # Backup path
    
def data_path() -> Path:
    return _data_path

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


def load_line_distortion_calibration(
        amplitude: units.Angle,
        path: Path = config_path() / "optics/line_distortion_calibration.csv"
    ):
    data = np.loadtxt(path, delimiter=',', dtype=np.float64, skiprows=1, ndmin=2)
    amplitudes = data[:,0]
    coefs = data[:,1:]

    for i,a in enumerate(amplitudes):
        if abs(a - amplitude)/amplitude < 0.001:
            return Polynomial(coefs[i])
    
    raise RuntimeError("Could not find distortion calibration")


def load_stage_scanner_angle(
        path: Path = config_path() / "optics/stage_scanner_angle.csv"
    ) -> units.Angle:
    try:
        data = np.loadtxt(path, delimiter=',', dtype=np.float64, skiprows=1)
        return units.Angle(float(data))
    except FileNotFoundError:
        # If not calibrated, then return 0 angle (no error axis error)
        return units.Angle("0 deg")


def load_signal_offset(
        path: Path = config_path() / "digitizer/signal_offset.csv"
):
    try:
        return np.loadtxt(path, delimiter=',', dtype=np.float64, skiprows=1)
    except FileNotFoundError:
        return np.array(0)


def load_line_gradient_calibration(
        path: Path = config_path() / "optics/line_gradient_calibration.tif"
    ):
    return tifffile.imread(path)



class SystemConfig: 
    def __init__(self, data: dict[str, dict]):
        self._data = data
        
    @cached_property
    def laser_scanning_optics(self) -> dict[str, Any]:
        return self._data["laser_scanning_optics"]
    
    @cached_property
    def camera_optics(self) -> dict[str, Any]:
        return self._data["camera_optics"]
    
    @cached_property
    def digitizer(self) -> dict[str, Any]:
        return self._data["digitizer"]
    
    @cached_property
    def detectors(self) -> dict[str, Any]:
        return self._data["detectors"]
    
    @cached_property
    def objective_z_scanner(self) -> dict[str, Any]:
        return self._data["objective_z_scanner"]

    @cached_property
    def stages(self) -> dict[str, Any]:
        return self._data["stages"]
    
    @cached_property
    def line_camera(self) -> dict[str, Any]:
        return self._data["line_camera"]
    
    @cached_property
    def illuminator(self) -> dict[str, Any]:
        return self._data["illuminator"]
    
    @cached_property
    def frame_grabber(self) -> dict[str, Any]:
        return self._data["frame_grabber"]
    
    @cached_property
    def encoders(self) -> dict[str, Any]:
        return self._data["encoders"]

    @cached_property
    def fast_raster_scanner(self) -> dict[str, Any]:
        return self._data["fast_raster_scanner"]
    
    @cached_property
    def slow_raster_scanner(self) -> dict[str, Any]:
        return self._data["slow_raster_scanner"]

    @classmethod
    def from_toml(cls, toml_path: Path) -> 'SystemConfig':
        toml_data = load_toml(toml_path)
        return cls(toml_data)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return self._data # should this be a (deep) copy?
    
