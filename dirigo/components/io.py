from pathlib import Path
import tomllib
from typing import Any
from functools import cached_property
from types import MappingProxyType
from copy import deepcopy

import numpy as np
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
        return np.array([0])


def load_line_gradient_calibration(
        line_width: units.Length,
        pixel_size: units.Length,
        path: Path = config_path() / "optics/line_gradient.csv"
    ):
    """Returns a function to correct intensity vignetting."""

    entries = np.loadtxt(
        fname       = path, 
        delimiter   = ',',
        dtype       = np.float64, 
        skiprows    = 1, 
    )

    x = entries[:,0]
    if abs(line_width - (x[-1] - x[0])) / line_width > 0.001:
        raise RuntimeError("Line gradient calibrated on different line size.")
    if abs(pixel_size - (x[1] - x[0])) / pixel_size > 0.001:
        raise RuntimeError("Line gradient calibrated on different pixel size.")
        # one could interpolate here probably TODO
    
    n_x, n_c = entries.shape
    n_c -= 1 # -1 to account for x axis label
    correction = np.zeros((n_x,n_c), dtype=np.float32)
    for c in range(n_c):
        y = entries[:,c+1]
        correction[:,c] = 1 / y

    return np.average(correction,axis=1)

        
_CONFIG_KEYS = (
    "laser_scanning_optics", "camera_optics", "digitizer", "detectors",
    "objective_z_scanner", "stages", "line_camera", "illuminator",
    "frame_grabber", "encoders", "fast_raster_scanner", "slow_raster_scanner",
)

class SystemConfig:
    """
    Slotted, read-mostly config: each known section becomes an attribute whose
    value is either a dict[...] or None if the section is absent in TOML.
    No cached_property needed; lookups are direct attribute access.
    """
    __slots__ = (*_CONFIG_KEYS, "_raw")

    # explicit (optional) type hints for IDEs / type checkers
    laser_scanning_optics:  dict[str, Any] | None
    camera_optics:          dict[str, Any] | None
    digitizer:              dict[str, Any] | None
    detectors:              dict[str, Any] | None
    objective_z_scanner:    dict[str, Any] | None
    stages:                 dict[str, Any] | None
    line_camera:            dict[str, Any] | None
    illuminator:            dict[str, Any] | None
    frame_grabber:          dict[str, Any] | None
    encoders:               dict[str, Any] | None
    fast_raster_scanner:    dict[str, Any] | None
    slow_raster_scanner:    dict[str, Any] | None

    def __init__(self, data: dict[str, dict[str, Any]]):
        self._raw = data
        for key in _CONFIG_KEYS:
            setattr(self, key, data.get(key))  # missing -> None

    @classmethod
    def from_toml(cls, toml_path: "Path") -> "SystemConfig":
        return cls(load_toml(toml_path))

    def has(self, key: str) -> bool:
        return getattr(self, key) is not None

    def to_dict(self, *, copy: bool = False, readonly: bool = False) -> dict[str, dict[str, Any] | None]:
        # expose the normalized view (sections -> dict | None)
        d = {k: getattr(self, k) for k in _CONFIG_KEYS}
        if copy:
            return deepcopy(d)
        if readonly:
            return MappingProxyType(d)
        return d

    def raw(self) -> dict[str, dict[str, Any]]:
        # original TOML mapping (no None entries; missing keys absent)
        return self._raw
    
