from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Literal, Self, Tuple, ClassVar, Type, overload
from importlib.metadata import entry_points


import numpy as np

from dirigo.components.units import IntRange
from dirigo.sw_interfaces.worker import Worker, Product
from dirigo.sw_interfaces.acquisition import Acquisition
from dirigo.sw_interfaces.processor import Processor




class DisplayPixelFormat(Enum):
    RGB24   = 0     # red, green, blue, each 8-bits (Tkinter)
    BGRX32  = 1     # blue, green, red, and 1 ignored, each 8 bits (PyQt/PySide)


# ---------- Color Vectors API ----------
class ColorVector(ABC):
    """
    Abstract color vector for Dirigo's colormap processor.

    • Subclasses represent an RGB triplet (values 0.0-1.0).  
    • Register concrete vectors via the ``dirigo_color_vectors`` entry-point
      group. For example:

        [project.entry-points."dirigo_color_vectors"]
        acridine_orange = my_plugin.colors:AcridineOrange

    Minimal implementation needs to set class-level attributes: 
        slug: short, unique identifier, 
        label: human-readable label
        rgb: tuple RGB triplet
    """
    # ---------- Required metadata ----------
    #: Short, unique identifier (used internally / in URLs).
    slug: ClassVar[str]

    #: Human-readable label shown in GUIs.
    label: ClassVar[str]

    # RGB triplet
    rgb: ClassVar[Tuple[float, float, float]]

    # ---------- Validation ----------
    def __init_subclass__(cls):
        super().__init_subclass__()
        if not isinstance(cls.rgb, tuple) or len(cls.rgb) != 3:
            raise TypeError(f"{cls.__name__}.rgb must be a 3-tuple")
        if any(not (0.0 <= v <= 1.0) for v in cls.rgb):
            raise ValueError(f"{cls.__name__}.rgb components must be in [0,1]")

    # ---------- Helpers ----------
    def __iter__(self):
        """Allows tuple unpacking: r, g, b = vec."""
        yield from self.rgb

    # Pretty repr
    def __repr__(self) -> str:  # pragma: no cover
        r, g, b = self.rgb
        return f"<{self.__class__.__name__} ({r:.3f}, {g:.3f}, {b:.3f})>"


def get_available_color_vector_names() -> list[str]:
    eps = entry_points(group="dirigo_color_vectors")
    return list(eps.names)


def load_color_vector(name: str) -> Type[ColorVector]:
    """
    Import and return the object registered under ``name`` in the
    ``dirigo_color_vectors`` entry-point group.

    Raises
    ------
    ValueError
        If no matching entry-point exists.
    """
    eps = entry_points(group="dirigo_color_vectors", name=name)  # returns an iterable
    try:
        ep = next(iter(eps))                    # take the first (should be one)
    except StopIteration:                       # nothing matched
        raise ValueError(f"No entry point '{name}' in group '{"dirigo_color_vectors"}'")
    return ep.load()                            # returns the object itself



# ---------- Transfer Function API ----------
class TransferFunction(ABC):
    """
    A parameterized tone / transfer function.

    Concrete subclasses are small frozen dataclasses whose *fields* are the
    parameters (e.g. gamma, alpha).  Those classes themselves are what you
    register as entry points.

        [project.entry-points."dirigo_transfer_functions"]
        gamma           = dirigo.transfer_funcs:Gamma        # needs a γ value
        invert_gamma    = dirigo.transfer_funcs:InvertGamma  # γ + invert flag

    Users obtain *instances* via construction with named arguments:
        tf = Gamma(gamma=2.2)          # instance
    """

    # ---------- Required metadata ----------
    #: Short, unique identifier (used internally / in URLs).
    slug:  ClassVar[str]

    #: Human-readable label shown in GUIs.
    label: ClassVar[str]

    # Public convenience
    @overload
    def __call__(self, x: float) -> float: ...
    
    @overload
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    def __call__(self, x: np.ndarray | float) -> np.ndarray | float:
        """Vectorised evaluation"""
        x_arr = np.asarray(x, dtype=np.float64)
        y = self._f(x_arr)
        return y if isinstance(x, np.ndarray) else float(y)  # keep scalar type

    # Required method
    @abstractmethod
    def _f(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the transfer curve in [0,1] linear space."""


def load_transfer_function(name: str, **params) -> TransferFunction:
    eps = entry_points(group="dirigo_transfer_functions", name=name)  # returns an iterable
    try:
        ep = next(iter(eps))                    # take the first (should be one)
    except StopIteration:                       # nothing matched
        raise ValueError(f"No entry point '{name}' in group '{"dirigo_color_vectors"}'")
    cls = ep.load()                            # returns the object itself
    return cls(**params)     # instantiate with user-supplied kwargs



# ---------- Display Worker API ----------
class DisplayProduct(Product):
    pass


class Display(Worker):
    """
    Dirigo interface for display processing.
    """
    Product = DisplayProduct

    def __init__(self, 
                 upstream: Acquisition | Processor | Self,
                 monitor_bit_depth: int = 8,
                 #gamma: float = 1/2.2,
                 pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24):
        """Instantiate with either an Acquisition or Processor"""
        super().__init__("Display Worker")

        if not isinstance(monitor_bit_depth, int) or not (8 <= monitor_bit_depth <= 12):
            raise ValueError("Unsupported monitor bit depth")
        self._monitor_bit_depth = monitor_bit_depth
        
        #self.gamma = gamma # gamma setter will validate this

        if isinstance(upstream, (Processor, Display)): # TODO, refactor this into some sort of mixin (with Logger's corresponding block)
            self._processor = upstream
            self._acquisition = upstream._acquisition
        elif isinstance(upstream, Acquisition):
            self._processor = None
            self._acquisition = upstream
        else:
            raise ValueError("Upstream Worker must be either an Acquisition or a Processor")
        
        if not isinstance(pixel_format, DisplayPixelFormat):
            raise ValueError(f"Invalid pixel format: {pixel_format}")
        self._pixel_format = pixel_format

        #self.display_channels: list[DisplayChannel] = []

    @property
    def nchannels(self):
        """Returns the number of channels expected from Acquisition or Processor."""
        return self._acquisition.hw.nchannels_enabled
    
    @property
    def bits_per_pixel(self) -> Literal[3, 4]:
        """
        Returns the output bits per pixel:
        3 for RGB   (tkinter-preferred)
        4 for BGRX  (Qt-preferred)
        """
        return 3 if self._pixel_format == DisplayPixelFormat.RGB24 else 4

    @property
    def data_range(self) -> IntRange:
        """ 
        The incoming data range from either the processor or the acquisition worker. 
        """
        if self._processor:
            return self._processor.data_range
        else:
            return self._acquisition.data_acquisition_device.data_range
        
    @property
    def tf_lut_length(self) -> int:
        """
        To compute final transfer function (e.g. gamma), use a LUT of this length.
        
        Default: 2^[monitor bit depth + 4]
        """
        return 2**(self._monitor_bit_depth + 4)

    # def _init_product_pool(self, n, shape, dtype=np.uint8):
    #     for _ in range(n):
    #         prod = DisplayProduct(
    #             pool=self._product_pool,
    #             frame=np.empty(shape, dtype) # pre-allocates for large buffers
    #         )
    #         self._product_pool.put(prod)

    def _get_free_product(self) -> DisplayProduct:
        return super()._get_free_product() # type: ignore
    
    # def _receive_product(self, block = True, timeout = None) -> AcquisitionProduct | ProcessorProduct:
    #     return super()._receive_product(block, timeout)

    @abstractmethod
    def update_display(self, skip_when_acquisition_in_progress: bool = True):
        """On demand reprocessing of the last acquired frame for display."""
        pass