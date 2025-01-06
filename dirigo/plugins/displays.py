from enum import Enum

import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces import Display, Acquisition, Processor


# Generate gamma correction LUT
gamma = 2.2
x = np.arange(2**16) / (2**16-1)
gamma_lut = np.round((2**8 - 1) * x**(1/gamma)).astype(np.uint8)


@njit(
    types.uint8[:,:,:](types.uint16[:,:,:], types.uint16[:,:,:]), 
    nogil=True, parallel=True, fastmath=True, cache=True
)
def additive_display_kernel(data: np.ndarray, luts: np.ndarray) -> np.ndarray:
    """Applies LUTs and blends channels additively."""
    Ny, Nx, Nc = data.shape
    
    image = np.zeros(shape=(Ny, Nx, 3), dtype=np.uint8)

    for yi in prange(Ny):
        for xi in prange(Nx):

            r, g, b = types.uint32(0), types.uint32(0), types.uint32(0)  
            for ci in range(Nc):
                lut_index = data[yi, xi, ci]
                r += luts[ci, lut_index, 0]
                g += luts[ci, lut_index, 1] 
                b += luts[ci, lut_index, 2]
            
            # Gamma correct blended values and assign to output image 
            image[yi, xi, 0] = gamma_lut[min(r, 2**16-1)]
            image[yi, xi, 1] = gamma_lut[min(g, 2**16-1)]
            image[yi, xi, 2] = gamma_lut[min(b, 2**16-1)]

    return image


@njit(
    types.uint8[:,:,:](types.uint16[:,:,:], types.uint16[:,:,:]), 
    nogil=True, parallel=True, fastmath=True, cache=True
)
def subtractive_display_kernel(data: np.ndarray, luts: np.ndarray) -> np.ndarray:
    """Applies LUTs and blends channels subtractively."""
    Ny, Nx, Nc = data.shape
    
    image = np.zeros(shape=(Ny, Nx, 3), dtype=np.uint8)

    for yi in prange(Ny):
        for xi in prange(Nx):

            r, g, b = types.int32(2**16-1), types.int32(2**16-1), types.int32(2**16-1)  
            for ci in range(Nc):
                lut_index = data[yi, xi, ci]
                r -= luts[ci, lut_index, 0]
                g -= luts[ci, lut_index, 1] 
                b -= luts[ci, lut_index, 2]
            
            # Gamma correct blended values and assign to output image 
            image[yi, xi, 0] = gamma_lut[max(r, 0)]
            image[yi, xi, 1] = gamma_lut[max(g, 0)]
            image[yi, xi, 2] = gamma_lut[max(b, 0)]

    return image


class AdditiveColorMap(Enum):
    GRAY =    [1, 1, 1]
    RED =     [1, 0, 0]
    GREEN =   [0, 1, 0]
    BLUE =    [0, 0, 1]
    CYAN =    [0, 1, 1]
    MAGENTA = [1, 0, 1]
    YELLOW =  [1, 1, 0]


def default_colormap_lists(nchannels: int) -> list[str]:
    if nchannels == 1:
        return ['gray']
    elif nchannels == 2:
        return ['cyan', 'red']
    elif nchannels == 3:
        return ['cyan', 'magenta', 'yellow']
    elif nchannels == 4:
        return ['cyan', 'magenta', 'yellow', 'gray']


class DisplayChannel:
    """Controls an individual display channel"""
    def __init__(self, colormap: AdditiveColorMap, display_min: int, display_max: int):
        # should it know its channel index?
        self.enabled: bool = True
        self._display_min = display_min
        self._display_max = display_max
        self._colormap = colormap

        self._input_dtype = np.uint16 # make this adjustable, not hardcoded

        self._update_lut()

    @property
    def input_min(self) -> int:
        return np.iinfo(self._input_dtype).min
    
    @property
    def input_max(self) -> int:
        return np.iinfo(self._input_dtype).max

    @property
    def input_range(self) -> int:
        return self.input_max - self.input_min
    
    @property
    def display_min(self) -> int:
        return self._display_min
    
    @property
    def colormap(self) -> AdditiveColorMap:
        return self._colormap

    @colormap.setter
    def colormap(self, colormap: AdditiveColorMap):
        self._colormap = colormap
        self._update_lut()

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
        x = np.arange(self.input_min, self.input_max+1)
        y_norm = np.clip(
            (x - self.display_min) / (self._display_max - self._display_min),
            a_min=0,
            a_max=1
        )
        cmap = np.array(self._colormap.value)
        lut_norm = cmap[np.newaxis, :] * y_norm[:, np.newaxis]
        self._lut = np.round((2**16 - 1) * lut_norm).astype(np.uint16)

    @property
    def lut(self) -> np.ndarray:
        return self._lut


class FrameDisplay(Display):
    # Output display bit depth

    def __init__(self, acquisition: Acquisition, processor: Processor):
        super().__init__(acquisition, processor)

        self.display_channels = []
        for ci, colormap_name in enumerate(default_colormap_lists(self.nchannels)):
            dc = DisplayChannel(AdditiveColorMap[colormap_name.upper()])
            self.display_channels.append(dc)

        # TODO: something with these attributes
        self.blending_mode = 'additive'
        self.gamma = 1.0 # 2.2 is common


    def run(self):
        while True:
            # Get new data from inbox
            data: np.ndarray = self.inbox.get(block=True) # may want to add a timeout

            if data is None: # Check for sentinel None
                self.publish(None) # pass sentinel
                print('exiting display thread')
                return # concludes run() - this thread ends
            
            processed = additive_display_kernel(data, self.luts_array)

            self.publish(processed)


# for testing
if __name__ == "__main__":

    dc = DisplayChannel(AdditiveColorMap.YELLOW, 0, 2**14)
    print(dc.lut)