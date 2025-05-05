from abc import abstractmethod
from enum import Enum
from typing import Callable

import numpy as np
from numba import njit, types

from dirigo.units import IntRange
from dirigo.sw_interfaces.worker import Worker, Product
from dirigo.sw_interfaces import Acquisition, Processor




sigs = [
    (types.int16[:],  types.int64, types.int64, types.float64[:], types.uint16[:,:], types.int64),
    (types.uint16[:], types.int64, types.int64, types.float64[:], types.uint16[:,:], types.int64)
]
@njit(sigs, cache=True)
def _update_lut_kernel(input_values, display_min, display_max, colormap, lut, gamma_lut_length):
    display_min_f32 = np.float32(display_min)
    display_range = np.float32(display_max - display_min) # possibly needs to be a +1 here

    max_output_value = np.float32(gamma_lut_length - 1)
    
    max0 = max_output_value * np.float32(colormap[0])
    max1 = max_output_value * np.float32(colormap[1])
    max2 = max_output_value * np.float32(colormap[2])

    zero = np.float32(0.0)
    one = np.float32(1.0)

    for i in input_values:

        y_norm = min(
            max(i - display_min_f32, zero) / display_range, 
            one
        ) # y_norm will be 0.0 - 1.0

        lut[i, 0] = int(max0 * y_norm)
        lut[i, 1] = int(max1 * y_norm)
        lut[i, 2] = int(max2 * y_norm)


class ColorVector(Enum):
    """Standard color names.
    
    Subclass this to inherit the standard names and add new ones.
    """
    GRAY =    (1.0, 1.0, 1.0)
    RED =     (1.0, 0.0, 0.0)
    GREEN =   (0.0, 1.0, 0.0)
    BLUE =    (0.0, 0.0, 1.0)
    CYAN =    (0.0, 1.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0)
    YELLOW =  (1.0, 1.0, 0.0)

    @classmethod
    def get_color_names(cls):
        """Return a list of string of available color names."""
        return [name.capitalize() for name in cls.__members__.keys()]


class DisplayPixelFormat(Enum):
    RGB24   = 0     # red, green, blue, each 8-bits (Tkinter)
    BGRX32  = 1     # blue, green, red, and 1 ignored, each 8 bits (PyQt/PySide)


class DisplayChannel(): # should this be a ABC?
    """Represents an individual channel to be processed for display."""
    def __init__(self, lut_slice: np.ndarray, color_vector: ColorVector, 
                 display_range: IntRange, update_method: Callable[[], None],
                 pixel_format: DisplayPixelFormat, gamma_lut_length: int):

        if not isinstance(lut_slice, np.ndarray) or (lut_slice.base is None):
            raise ValueError("`lut_slice` property must be a numpy slice.")
        self._lut = lut_slice # reference to this channel's 'slice' of multichannel LUT
        self._pixel_format = pixel_format
        self._update_display_method = update_method
        self._gamma_lut_length = gamma_lut_length

        # Adjustable parameters (see getter/setters)
        self._enabled = True
        self._color_vector = color_vector
        self._display_range = display_range 

        self._input_values = np.arange(
            start=display_range.min, 
            stop=display_range.max + 1, 
            dtype=display_range.recommended_dtype
        )

        self._update_lut()

    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, new_state: bool):
        if not isinstance(new_state, bool):
            raise ValueError("`enabled` property must be set with a boolean.")
        self._enabled = new_state
        self._update_lut()

    @property
    def color_vector(self) -> ColorVector:
        return self._color_vector

    @color_vector.setter
    def color_vector(self, color_vector: ColorVector):
        if not isinstance(color_vector, ColorVector):
            raise ValueError("`color_vector` property must be set with a ColorVector object.")
        self._color_vector = color_vector
        self._update_lut()

    @property
    def display_min(self) -> int:
        return self._display_range.min

    @display_min.setter
    def display_min(self, value: int):
        self._display_range.min = int(value)
        self._update_lut()

    @property
    def display_max(self) -> int:
        return self._display_range.max
    
    @display_max.setter
    def display_max(self, value: int):
        self._display_range.max = int(value)
        self._update_lut()

    def _update_lut(self):
        if self._pixel_format == DisplayPixelFormat.RGB24:
            colormap = self._color_vector.value if self.enabled else (0.0, 0.0, 0.0)
        elif self._pixel_format == DisplayPixelFormat.BGRX32:
            colormap = tuple(reversed(self._color_vector.value)) + (0.0,) if self.enabled else (0.0, 0.0, 0.0, 0.0)

        _update_lut_kernel(
            input_values=self._input_values,
            display_min=self.display_min,
            display_max=self.display_max,
            colormap=np.array(colormap),
            lut=self._lut,
            gamma_lut_length=self._gamma_lut_length
        )
        self._update_display_method()


class DisplayProduct(Product):
    __slots__ = ("frame")
    def __init__(self, pool, frame: np.ndarray):
        super().__init__(pool)
        self.frame = frame



class Display(Worker):
    """
    Dirigo interface for display processing.
    """
    def __init__(self, 
                 upstream: Acquisition | Processor,
                 monitor_bit_depth: int = 8,
                 gamma: float = 1/2.2):
        """Instantiate with either an Acquisition or Processor"""
        super().__init__("Display Worker")

        if not isinstance(monitor_bit_depth, int) or not (8 <= monitor_bit_depth <= 12):
            raise ValueError("Unsupported monitor bit depth")
        self._monitor_bit_depth = monitor_bit_depth
        
        self.gamma = gamma # gamma setter will validate this

        if isinstance(upstream, Processor): # TODO, refactor this into some sort of mixin (with Logger's corresponding block)
            self._processor = upstream
            self._acquisition = upstream._acq
        elif isinstance(upstream, Acquisition):
            self._processor = None
            self._acquisition = upstream
        else:
            raise ValueError("Upstream Worker must be either an Acquisition or a Processor")

        self.display_channels: list[DisplayChannel] = []

    @property
    def nchannels(self):
        """Returns the number of channels expected from Acquisition or Processor."""
        return self._acquisition.hw.nchannels_enabled

    @property
    @abstractmethod
    def n_frame_average(self) -> int:
        """The number of frames used in the rolling average.
        
        To enable averaging, set with an integer greater than 1. Setting 1 will
        disable averaging.
        """
        pass

    @n_frame_average.setter
    @abstractmethod
    def n_frame_average(self, frames: int):
        pass

    @property
    def pixel_size(self):
        return self._acquisition.spec.pixel_size
        
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
    def gamma_lut_length(self) -> int:
        """
        To compute the gamma function, use a LUT of this length.
        
        Default: 2^[monitor bit depth + 4]
        """
        return 2**(self._monitor_bit_depth + 4)

    def init_product_pool(self, n, shape, dtype=np.uint8):
        for _ in range(n):
            prod = DisplayProduct(
                pool=self._product_pool,
                frame=np.empty(shape, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(prod)

    def get_free_product(self) -> DisplayProduct:
        return self._product_pool.get()
