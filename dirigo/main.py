from typing import TYPE_CHECKING, Optional, Literal, Any, overload
from functools import lru_cache
import importlib.metadata as im
from pathlib import Path

from dirigo import io
from dirigo.components.hardware import Hardware
from dirigo.sw_interfaces.display import DisplayPixelFormat

if TYPE_CHECKING:
    from dirigo.sw_interfaces import Acquisition, Processor, Display, Logger
    from dirigo.sw_interfaces.acquisition import AcquisitionSpec, Loader
    from dirigo.plugins.processors import RollingAverageProcessor
    from dirigo.plugins.displays import FrameDisplay


class PluginError(RuntimeError): ...


@lru_cache
def _load_entry_points(group: str) -> dict[str, type]:
    eps = im.entry_points(group=f"dirigo_{group}s")
    if not eps:
        raise PluginError(f"No entry-points found for group '{group}'.")
    return {ep.name: ep.load() for ep in eps}


class Dirigo:
    """High-level facade for building Dirigo acquisition pipelines."""
    _instance: Optional["Dirigo"] = None
    _init_args: Optional[dict] = None

    def __new__(cls, *a, **k):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, 
                 system_config_path: Optional[Path | str] = None,
                 *,
                 start_services: bool = False):
        
        cfg = Path(system_config_path or (io.config_path() / "system_config.toml"))
        args = {"system_config_path": cfg, "start_services": bool(start_services)}
        if getattr(self, "_initialized", False):
            if self._init_args and self._init_args != args:
                raise RuntimeError(
                    "Dirigo is a singleton and was already created with "
                    f"{self._init_args}. New args {args} are not allowed."
                )
            return
        
        if not cfg.exists():
            raise FileNotFoundError(
                f"Could not find system configuration file at {cfg}."
            )
        
        self.system_config = io.SystemConfig.from_toml(cfg)
        self.hw = Hardware(self.system_config) 

        self._init_args = args
        self._initialized = True

    def available(self, group: str) -> set[str]:
        """Return available plugin names for a given group."""
        return set(_load_entry_points(group))

    @overload
    def make(self, group: Literal["acquisition"], name: str, *,
             spec: "AcquisitionSpec | str | None" = ...,
             **kw: Any) -> "Acquisition": ...
    
    @overload
    def make(self, group: Literal["loader"], name: str, *,
             file_path: Path) -> "Loader": ...

    @overload
    def make(self, group: Literal["processor"],  name: str, *,
             upstream: "Acquisition | Processor", **kw: Any) -> "Processor": ...
    
    @overload
    def make(self, group: Literal["display"], name: str, *,
             upstream: "Acquisition | Processor",
             pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
             **kw: Any) -> "Display": ...
    
    @overload
    def make(self, group: Literal["logger"], name: str, *,
             upstream: "Acquisition | Processor", **kw: Any) -> "Logger": ...

    def make(self, group: str, name: str, **kw: Any
             ) -> "Acquisition | Loader | Processor | Display | Logger":
        """
        Make pluggable pipeline elements (acquisitions, processors, loggers,
        displays)
        """
        plugins = _load_entry_points(group)

        try:
            cls = plugins[name]
        except KeyError:
            raise PluginError(
                f"No {group} plugin named '{name}'. Available: {set(plugins)}"
            )
        
        if group not in ("acquisition", "loader", "display", "processor", "logger"):
            raise PluginError(f"Unsupported plugin group '{group}'.")

        # special handling per group
        if group == "acquisition":
            spec = kw.pop("spec", None)
            if isinstance(spec, str) or spec is None:
                spec = cls.get_specification(spec_name=spec or "default")
            obj = cls(self.hw, self.system_config, spec, **kw)              # type: ignore[arg-type]
        
        elif group == "loader":
            # For loaders, spec information should be availabe in saved file
            file_path = kw.pop("file_path")
            obj = cls(file_path, **kw)

        else:
            upstream = kw.pop("upstream")
            autostart = kw.pop("autostart", True)
            autoconnect = kw.pop("autoconnect", True)

            obj = cls(upstream, **kw)
            if autoconnect:
                upstream.add_subscriber(obj)
            if autostart:
                obj.start()
        
        obj._dirigo_group = group
        obj._dirigo_plugin = name
        if hasattr(obj, "thread_name"):
            setattr(obj, "_dirigo_worker", obj.thread_name)

        return obj 
    
    def make_acquisition(self, name: str, **kw: Any) -> "Acquisition":
        return self.make("acquisition", name, **kw)
    
    def make_loader(self, name: str, file_path: Path, **kw: Any) -> "Loader":
        return self.make("loader", name, file_path=file_path)
    
    @overload
    def make_processor(self, name: Literal["rolling_average"], *,               # type: ignore
                       upstream: "Acquisition | Processor") -> "RollingAverageProcessor": ...
    
    def make_processor(self, name: str, *, upstream, **kw: Any) -> "Processor":
        return self.make("processor", name, upstream=upstream, **kw)
    
    @overload
    def make_display_processor(self, name: Literal["frame"], *,                 # type: ignore
                               upstream: "Acquisition | Processor",
                               pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
                               color_vector_names: Optional[list[str]] = None,
                               transfer_function_name: Optional[str] = None
                               ) -> "FrameDisplay": ...
        
    def make_display_processor(self, name: str, *, upstream, **kw: Any) -> "Display":
        return self.make("display", name, upstream=upstream, **kw)
    
    def make_logger(self, name: str, *, upstream, **kw: Any) -> "Logger":
        return self.make("logger", name, upstream=upstream, **kw)
