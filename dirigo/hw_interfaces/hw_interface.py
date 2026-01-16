import re
from abc import ABC
from typing import ClassVar, Type
from pydantic import BaseModel


class HardwareInterface(ABC):
    """
    Marker base-class for anything that can appear as an attribute
    on `Hardware`.  
    
    Subclasses should override `attr_name`.
    
    Subclasses may declare a `config_model` (Pydantic BaseModel)
    describing how the device is configured from system config.
    """
    config_model: ClassVar[Type[BaseModel]]
    attr_name: str | None = None         # default → derive automatically

    @classmethod
    def attr(cls) -> str:
        """Return the attribute name where this interface is expected."""
        if cls.attr_name:
            return cls.attr_name
        # fallback: FastRasterScanner -> fast_raster_scanner
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
    
    @classmethod
    def from_config(cls, config: dict, **kwargs):
        if cls.config_model is None:
            if config:
                raise ValueError(f"{cls.__name__} does not accept config.")
            return cls(**kwargs)

        cfg = cls.config_model(**config)
        return cls(**cfg.model_dump(), **kwargs)


class NoBuffers(Exception):
    """Raised by HardWareInterface when no buffers are available."""
    pass