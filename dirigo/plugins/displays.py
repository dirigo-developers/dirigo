import copy
from dataclasses import dataclass
from typing import ClassVar, Tuple, Callable, Optional

import numpy as np
from numba import njit, prange, types, uint8, int16, uint16, int64, float32

from dirigo.components.units import IntRange
from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.sw_interfaces.display import (
    Display, ColorVector, TransferFunction, DisplayPixelFormat,
    load_color_vector, load_transfer_function
)
from dirigo.plugins.acquisitions import FrameAcquisition, LineAcquisition




# ---------- Standard Color Vectors (Concrete Classes) ----------
class Gray(ColorVector):
    """Grayscale color vector. Frequently used for single-channel display."""
    slug = "gray"
    label = "Gray"
    rgb = (1.0, 1.0, 1.0)


class Red(ColorVector):
    """Red color vector."""
    slug = "red"
    label = "Red"
    rgb = (1.0, 0.0, 0.0)


class Green(ColorVector):
    """Green color vector."""
    slug = "green"
    label = "Green"
    rgb = (0.0, 1.0, 0.0)


class Blue(ColorVector):
    """Blue color vector."""
    slug = "blue"
    label = "Blue"
    rgb = (0.0, 0.0, 1.0)


class Cyan(ColorVector):
    """Cyan color vector."""
    slug = "cyan"
    label = "Cyan"
    rgb = (0.0, 1.0, 1.0)


class Magenta(ColorVector):
    """Magenta color vector."""
    slug = "magenta"
    label = "Magenta"
    rgb = (1.0, 0.0, 1.0)


class Yellow(ColorVector):
    """Yellow color vector."""
    slug = "yellow"
    label = "Yellow"
    rgb = (1.0, 1.0, 0.0)


# ---------- Standard Transfer Functions (Concrete Classes) ----------
@dataclass(frozen=True, slots=True)
class Gamma(TransferFunction):
    """
    Standard power-law (linear->display) transfer.
    gamma < 1 boosts mid-tones; gamma > 1 darkens.
    """
    gamma: float = 0.45

    slug:  ClassVar[str] = "gamma"
    label: ClassVar[str] = "Gamma"

    def _f(self, x: np.ndarray) -> np.ndarray:
        return x ** self.gamma


@dataclass(frozen=True, slots=True)
class InvertedGamma(TransferFunction):
    """
    Inverted contrast + power-law (linear->display) transfer.
    gamma < 1 boosts mid-tones; gamma > 1 darkens.
    """
    gamma: float = 0.45

    slug:  ClassVar[str] = "inverted_gamma"
    label: ClassVar[str] = "Inverted gamma"

    def _f(self, x: np.ndarray) -> np.ndarray:
        return (1 - x) ** self.gamma

# TODO add log transfer function



# ---------- Display Channel API (additive single-hue channels) ----------
UT = types.UniTuple
sigs = [
#   input_values  disp_min  disp_max  color_vector    lut          lut_max
    (int16[:],    int64,    int64,    UT(float32,3),  uint16[:,:], int64),
    (uint16[:],   int64,    int64,    UT(float32,3),  uint16[:,:], int64)
]
@njit(sigs, cache=True)
def _update_lut_kernel(input_values: np.ndarray, 
                       disp_min: int, 
                       disp_max: int, 
                       color_vector: Tuple[float, ...], 
                       lut: np.ndarray, 
                       lut_max: int):
    """
    JIT-compiled function for fast LUT recalculation.

    input_values: index of all the possible input values 
    disp_min: value to set to minimum display level (everything below is clipped)
    disp_max: value to set to maximum display level (everything above is clipped)
    color_vector: tuple of floats for RGB hue
    lut: array to place the new LUT data
    lut_max: max **Value** of the LUT (should be numebr of entries in the final
        transfer function LUT)
    """
    zero = np.float32(0.0)
    one = np.float32(1.0)

    # cast integers to FP32
    display_min_f32 = np.float32(disp_min)
    display_range = np.float32(disp_max - disp_min) # possibly needs to be a +1 here
    max_output_value = np.float32(lut_max - 1)
    
    max0 = max_output_value * np.float32(color_vector[0])
    max1 = max_output_value * np.float32(color_vector[1])
    max2 = max_output_value * np.float32(color_vector[2])

    for i in input_values:
        y_norm = min(                                 # y_norm will be 0.0 - 1.0
            max(i - display_min_f32, zero) / display_range, 
            one
        ) 
        lut[i, 0] = int(max0 * y_norm)
        lut[i, 1] = int(max1 * y_norm)
        lut[i, 2] = int(max2 * y_norm)


class DisplayChannel():
    """Represents an individual channel to be processed for display."""
    def __init__(self, 
                 lut_slice: np.ndarray, 
                 color_vector_name: str, 
                 display_range: IntRange, 
                 update_method: Callable[[], None],
                 pixel_format: DisplayPixelFormat, 
                 lut_max: int):
        """
        Sets up the display channel with required parameters and computes a LUT.
        """

        # Validators
        if not isinstance(lut_slice, np.ndarray) or (lut_slice.base is None):
            raise ValueError("``lut_slice`` property must be a numpy slice.")

        self._lut = lut_slice # reference to this channel's 'slice' of multichannel LUT
        self._pixel_format = pixel_format
        self._update_display_method = update_method
        self._lut_max = lut_max

        # Adjustable parameters (see getter/setters)
        self._enabled = True
        self._color_vector = load_color_vector(color_vector_name)
        self._display_range = copy.copy(display_range)

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
            raise ValueError("``enabled`` must be set with a boolean.")
        self._enabled = new_state
        self._update_lut()

    @property
    def color_vector_name(self) -> str:
        return self._color_vector.slug

    @color_vector_name.setter
    def color_vector_name(self, new_color_vector_name: str):
        self._color_vector = load_color_vector(new_color_vector_name)
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
        # get the RGB triplet or BGRX quadruplet
        
        if self.enabled:
            if self._pixel_format == DisplayPixelFormat.RGB24:
                color_vector = self._color_vector.rgb 
            elif self._pixel_format == DisplayPixelFormat.BGRX32:
                color_vector = tuple(reversed(self._color_vector.rgb))
        else:
            color_vector = (0.0, 0.0, 0.0)
        
        _update_lut_kernel(
            input_values    = self._input_values,
            disp_min        = self.display_min,
            disp_max        = self.display_max,
            color_vector    = color_vector,
            lut             = self._lut,
            lut_max         = self._lut_max
        )
        self._update_display_method()


# ---------- Colormapped Channel API (single-channel) ----------
# like viridis false-coloring TODO


# ---------- Standard Multichannel Frame Display ----------
int16_3d_readonly = types.Array(types.int16, 3, 'C', readonly=True)
uint16_3d_readonly = types.Array(types.uint16, 3, 'C', readonly=True)
sigs = [
#    data           luts           tf_lut    image
    (int16_3d_readonly,  uint16[:,:,:], uint8[:], uint8[:,:,:]),
    (uint16_3d_readonly, uint16[:,:,:], uint8[:], uint8[:,:,:]),
]
@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def _additive_blend_channels_kernel(data: np.ndarray, 
                                    luts: np.ndarray,
                                    tf_lut: np.ndarray, 
                                    image: np.ndarray) -> np.ndarray:
    """Applies LUTs and blends channels additively then applies a final 
    transfer function."""
    Ny, Nx, Nc = data.shape
    tf_lut_length = tf_lut.shape[0]

    for yi in prange(Ny):
        for xi in prange(Nx):
            r, g, b = 0, 0, 0 
            for ci in range(Nc):
                data_value = data[yi, xi, ci]
                r += luts[ci, data_value, 0]
                g += luts[ci, data_value, 1] 
                b += luts[ci, data_value, 2]
            image[yi, xi, 0] = tf_lut[min(r, tf_lut_length-1)]
            image[yi, xi, 1] = tf_lut[min(g, tf_lut_length-1)]
            image[yi, xi, 2] = tf_lut[min(b, tf_lut_length-1)]
    return image


def _default_colormap_lists(nchannels: int) -> list[str]:
    if nchannels == 1:
        return ['gray']
    elif nchannels == 2:
        return ['cyan', 'red']
    elif nchannels == 3:
        return ['cyan', 'magenta', 'yellow']
    elif nchannels == 4:
        return ['cyan', 'magenta', 'yellow', 'gray']
    else:
        raise ValueError("No default colormaps set yet for >4 channels")
    

class FrameDisplay(Display):
    """
    Multichannel frame display.
    """
    def __init__(self,
                 upstream: Processor, # TODO type upstream
                 color_vector_names: Optional[list[str]] = None,
                 transfer_function_name: str= "gamma",
                 **kwargs): # pixel_format, monitor_bit_depth
        super().__init__(upstream, **kwargs)
        self._prev_data = None # stores last input for on-depand reprocessing adjustments

        nchannels = upstream.product_shape[2]
        if color_vector_names is None: # use some preset defaults
            color_vector_names = _default_colormap_lists(nchannels)
        else:
            if len(color_vector_names) != nchannels:
                raise ValueError("Mismatch between color vector arguments and physical input channels present")

        bpp = self.bits_per_pixel
        self._luts = np.zeros(
            shape   = (nchannels, self.data_range.range, bpp), 
            dtype   = np.uint16
        )
        
        self.display_channels: list[DisplayChannel] = []
        for ci, colormap_name in enumerate(color_vector_names):
            dc = DisplayChannel(
                lut_slice           = self._luts[ci],
                color_vector_name   = colormap_name,
                display_range       = self.data_range,
                pixel_format        = self._pixel_format,
                update_method       = self.update_display,
                lut_max             = self.tf_lut_length # TODO rename gamma
            )
            self.display_channels.append(dc)

        shape = (*upstream.product_shape[:2], bpp)
        self._init_product_pool(n=4, shape=shape, dtype=np.uint8) # TODO

        # Make transfer function
        self._monitor_levels = 2 ** self._monitor_bit_depth
        self._monitor_levels_minus_one = self._monitor_levels - 1
        self._transfer_function = load_transfer_function(transfer_function_name)
        self._x = np.arange(self.tf_lut_length, dtype=np.float32) / (self.tf_lut_length - 1)
            
        self._update_tf_lut()

    def _update_tf_lut(self):
        self._tf_lut = np.array(
            np.round(self._monitor_levels_minus_one *self._transfer_function(self._x)), 
            dtype = np.uint8
        )
    
    @property
    def transfer_function_name(self) -> str:
        return self._transfer_function.slug
    
    @transfer_function_name.setter
    def transfer_function_name(self, value: str | tuple[str, float]):
        if isinstance(value, tuple):
            name, param = value
            kwargs = {"value": param}
        else:
            name = value
            kwargs = {}

        self._transfer_function = load_transfer_function(
            name    = name,
            params  = kwargs
        )
        self._update_tf_lut()

    def _work(self):
        try:
            while True:
                with self._receive_product() as product:
                    display_product = self._get_free_product()
                    
                    _additive_blend_channels_kernel(
                        data    = product.data, 
                        luts    = self._luts,   # LUT for each channel
                        tf_lut  = self._tf_lut, # final output transfer function LUT
                        image   = display_product.data
                    )
                   
                    self._publish(display_product)
                    self._prev_data = product.data.copy()

        except EndOfStream:
            self._publish(None) # forward sentinel None

    def update_display(self, skip_when_acquisition_in_progress: bool = True):
        """
        On demand reprocessing of the last acquired frame for display.
        
        Used when the acquisition is stopped and need to update the appearance  
        of the last acquired frame.
        """
        if self.is_alive() and skip_when_acquisition_in_progress:
            # Don't update if acquisition is in progress
            return
        
        if self._prev_data is None:
            # Don't update if no previous data exists
            return
        
        display_product = self._get_free_product()
        
        _additive_blend_channels_kernel(
            data    = self._prev_data, 
            luts    = self._luts,   # LUT for each channel
            tf_lut  = self._tf_lut, # final output transfer function LUT
            image   = display_product.data
        )
        self._publish(display_product)

            