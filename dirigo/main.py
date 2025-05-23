from typing import Optional, Literal, Any, overload
from functools import lru_cache
import importlib.metadata as im
from pathlib import Path

from dirigo import io
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces import Acquisition, Processor, Display, Logger
from dirigo.sw_interfaces.acquisition import AcquisitionSpec
from dirigo.sw_interfaces.display import DisplayPixelFormat
from dirigo.plugins.displays import FrameDisplay


_Worker = Acquisition | Processor | Display | Logger


class PluginError(RuntimeError):
    """Raised when a Dirigo plugin cannot be located or instantiated."""


@lru_cache
def _load_entry_points(group: str) -> dict[str, type]:
    eps = im.entry_points(group=f"dirigo_{group}s")
    if not eps:
        raise PluginError(f"No entry-points found for group '{group}'.")
    return {ep.name: ep.load() for ep in eps}


class Dirigo:
    """High-level facade for building Dirigo acquisition pipelines."""

    def __init__(self, 
                 system_config_path: Optional[Path] = None) -> None:
        
        if system_config_path is None:
            system_config_path = io.config_path() / "system_config.toml"
        
        if not system_config_path.exists():
            raise FileNotFoundError(
                f"Could not find system configuration file at {system_config_path}."
            )
        
        self.system_config = io.SystemConfig.from_toml(system_config_path)
        self.hw = Hardware(self.system_config) 

    def available(self, group: str) -> set[str]:
        """Return available plugin names for a given group."""
        return set(_load_entry_points(group))

    @overload
    def make(self, group: Literal["acquisition"], name: str = ..., *,
             spec: AcquisitionSpec | str | None = ...,
             **kw: Any) -> Acquisition: ...
    
    @overload
    def make(self, group: Literal["processor"],  name: str = ..., *,
             upstream: Acquisition | Processor, **kw: Any) -> Processor: ...
    
    @overload
    def make(self, group: Literal["display"], name: str = ..., *,
             upstream: Acquisition | Processor,
             pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
             **kw: Any) -> Display: ...
    
    @overload
    def make(self, group: Literal["logger"], name: str = ..., *,
             upstream: Acquisition | Processor, **kw: Any) -> Logger: ...

    def make(self, group: str, name: str, **kw: Any) -> _Worker:
        """
        Make pluggable pipeline elements (acquisitions, processors, loggers,
        displays)
        """
        plugins = _load_entry_points(group)

        try:
            cls = plugins[name]
        except KeyError:
            raise PluginError(f"No {group} plugin named '{name}'. "
                              f"Available: {set(plugins)}")

        # special handling per group
        if group == "acquisition":
            spec = kw.pop("spec", None)
            if isinstance(spec, str) or spec is None:
                spec = cls.get_specification(spec_name=spec or "default")
            return cls(self.hw, self.system_config, spec, **kw)              # type: ignore[arg-type]
        
        upstream = kw.pop("upstream")
        autostart = kw.pop("autostart", True)
        autoconnect = kw.pop("autoconnect", True)

        if group == "display" and cls is FrameDisplay:                       # built-in fast path
            pixel_fmt = kw.pop("pixel_format", DisplayPixelFormat.RGB24)
            disp = FrameDisplay(upstream, pixel_fmt, **kw)
            if autoconnect:
                upstream.add_subscriber(disp)
            if autostart:
                disp.start()
            return disp

        if group in ("processor", "logger"):
            obj = cls(upstream, **kw)
            if autoconnect:
                upstream.add_subscriber(obj)
            if autostart:
                obj.start()
            return obj

        raise PluginError(f"Unsupported plugin group '{group}'.")

    

if __name__ == "__main__":

    diri = Dirigo()

    acquisition = diri.make("acquisition", "raster_frame")
    processor   = diri.make("processor", "raster_frame", upstream=acquisition)
    display     = diri.make("display", "frame", upstream=processor)
    logger      = diri.make("logger", "tiff", upstream=processor)

    acquisition.start()
    acquisition.join(timeout=100.0)

    print("Acquisition complete")
