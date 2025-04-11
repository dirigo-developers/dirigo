from abc import abstractmethod
from enum import Enum
from typing import Callable

import numpy as np
from numba import njit, types

from dirigo.units import ValueRange
from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces import Acquisition, Processor


@njit(
    (types.uint16[:], types.int64, types.int64, types.float64[:], types.uint16[:,:]),
    cache=True
)
def _update_lut(input_values, display_min, display_max, colormap, lut):
    N = input_values.size
    display_min_f32 = np.float32(display_min)
    display_range = np.float32(display_max - display_min)
    max_output_value = np.float32(2**16 - 1)

    max0 = max_output_value * np.float32(colormap[0])
    max1 = max_output_value * np.float32(colormap[1])
    max2 = max_output_value * np.float32(colormap[2])
    zero = np.float32(0.0)
    one = np.float32(1.0)

    for i in range(N):
        y_norm = min(
            max(input_values[i] - display_min_f32, zero) / display_range, 
            one
        )
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
        """Return a list of string of available color names.
        """
        return [name.capitalize() for name in cls.__members__.keys()]

class DisplayPixelFormat(Enum):
    RGB24   = 0     # red, green, blue, each 8-bits (Tkinter)
    BGRX32  = 1     # blue, green, red, and 1 ignored, each 8 bits (PyQt/PySide)


class DisplayChannel(): #ABC?
    """Represents an individual channel to be processed for display."""
    def __init__(self, lut_slice: np.ndarray, color_vector: ColorVector, 
                 display_min: int, display_max: int, update_method: Callable[[], None],
                 pixel_format: DisplayPixelFormat):
        """Constructs a DisplayChannel object
        
        Args:
            lut_slice: a view (slice) to underlying 3D LUT numpy array.
            color_vector
            display_min
            display_max
        """
        if not isinstance(lut_slice, np.ndarray) or (lut_slice.base is None):
            raise ValueError("`lut_slice` property must be a numpy slice.")
        self._lut = lut_slice # reference to this channel's 'slice' of multichannel LUT
        self._pixel_format = pixel_format
        self._update_display_method = update_method

        # Adjustable parameters (see getter/setters)
        self._enabled = True
        self._color_vector = color_vector
        self._display_min = display_min 
        self._display_max = display_max

        input_min = 0
        input_max = lut_slice.shape[0] - 1
        if input_max > 255:
            input_dtype = np.uint16 
        else:
            input_dtype = np.uint8

        self._input_values = np.arange(input_min, input_max+1, dtype=input_dtype)

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
        return self._display_min

    @display_min.setter
    def display_min(self, value: int):
        self._display_min = int(value)
        self._update_lut()

    @property
    def display_max(self) -> int:
        return self._display_max
    
    @display_max.setter
    def display_max(self, value: int):
        self._display_max = int(value)
        self._update_lut()

    def _update_lut(self):
        if self._pixel_format == DisplayPixelFormat.RGB24:
            colormap = self._color_vector.value if self.enabled else (0.0, 0.0, 0.0)
        elif self._pixel_format == DisplayPixelFormat.BGRX32:
            colormap = tuple(reversed(self._color_vector.value)) + (0.0,) if self.enabled else (0.0, 0.0, 0.0, 0.0)

        _update_lut(
            input_values=self._input_values,
            display_min=self.display_min,
            display_max=self.display_max,
            colormap=np.array(colormap),
            lut=self._lut
        )
        self._update_display_method()


class Display(Worker):
    """
    Dirigo interface for display processing.
    """
    GAMMA = 2.2 # sRGB
    def __init__(self, acquisition: Acquisition = None, processor: Processor = None):
        """Instantiate with either an Acquisition or Processor"""
        super().__init__()

        if (acquisition is not None) and (processor is not None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor, not both")
        elif (acquisition is None) and (processor is None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor.")
        if processor:
            self._processor = processor
            self._acquisition = processor._acq
        else:
            self._processor = None
            self._acquisition = acquisition

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
    def data_range(self) -> ValueRange:
        """ Returns the raw data range from the data acquisition device (digitizer or camera)"""
        return self._acquisition.data_acquisition_device.data_range

