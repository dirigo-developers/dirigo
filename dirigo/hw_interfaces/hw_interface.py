import re
from abc import ABC


class HardwareInterface(ABC):
    """
    Marker base-class for anything that can appear as an attribute
    on `Hardware`.  Sub-classes *should* override `attr_name`.
    """
    attr_name: str | None = None         # default → derive automatically

    @classmethod
    def attr(cls) -> str:
        """Return the attribute name where this interface is expected."""
        if cls.attr_name:
            return cls.attr_name
        # fallback: FastRasterScanner -> fast_raster_scanner
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()
    

class NoBuffers(Exception):
    """Raised by HardWareInterface when no buffers are available."""
    pass