from functools import cached_property
import queue
import time

import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces import Display, Acquisition, Processor


# TODO, 16 bit LUT is a bit clunky, investigate whether we can streamline this


# TODO inverted color maps (start triplets not all 0's)
class LookUpTable:
    N = 2**15 # number of points in the LUT 
    nbits = 16
    displaybits = 8

    def __init__(self, name: str, lower_limit: int, upper_limit: int):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

        if name in {"gray", "grey", "grayscale", "greyscale"}:
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [1, 1, 1]
        elif name == "red":
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [1, 0, 0]
        elif name == "green":
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [0, 1, 0]
        elif name == "blue":
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [0, 0, 1]
        elif name == "cyan":
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [0, 1, 1]
        elif name == "magenta":
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [1, 0, 1]
        elif name == "yellow":
            self.start_triplet = [0, 0, 0]
            self.end_triplet =   [1, 1, 0]
        else:
            raise ValueError("Invalide colormap name")
        
        self.name = name

    @property
    def range(self) -> int:
        return self.upper_limit - self.lower_limit

    @cached_property
    def _pixel_values(self) -> np.ndarray:
        return np.arange(
            start=-2**(self.nbits-1), 
            stop=2**(self.nbits-1), 
            dtype=np.float32
        ) 
    
    @cached_property
    def _display_max(self) -> int:
        return (2 ** self.displaybits) - 1

    @property
    def table(self) -> np.ndarray:

        clipped = np.clip(
            (self._pixel_values - self.lower_limit) / self.range, 
            a_min=0, 
            a_max=1
        )
        y = np.round(clipped * self._display_max).astype(np.uint8)

        return y[:, np.newaxis] * np.array(self.end_triplet, np.uint8)[np.newaxis, :]


@njit(
    types.uint8[:,:,:](
        types.uint16[:,:,:], 
        types.uint8[:,:,:]
    ), 
    nogil=True, 
    parallel=True, 
    fastmath=True, 
    cache=True
)
def display_kernel(data: np.ndarray, luts: np.ndarray) -> np.ndarray:
    """Adjusts image data for direct display.
    
    Applies LUT.
    """
    Ny, Nx, Nc = data.shape
    
    image = np.zeros(shape=(Ny, Nx, 3), dtype=np.uint8)

    for yi in prange(Ny):
        for xi in prange(Nx):

            # additive blending
            r, g, b = 0, 0, 0  # Scalars for blending
            for ci in range(Nc):
                lut_index = data[yi, xi, ci]
                r += luts[ci, lut_index, 0]
                g += luts[ci, lut_index, 1] 
                b += luts[ci, lut_index, 2]
            
            # Assign blended values to the output image
            image[yi, xi, 0] = r
            image[yi, xi, 1] = g
            image[yi, xi, 2] = b

    return image


# @njit(
#     types.uint8[:,:,:](
#         types.int16[:,:,:], 
#     ), 
#     nogil=True, 
#     parallel=True, 
#     fastmath=True, 
#     cache=True
# )
# def direct_display_kernel(data: np.ndarray) -> np.ndarray:
#     Ny, Nx, Nc = data.shape
    
#     image = np.zeros(shape=(Ny, Nx, 3), dtype=np.uint8)

#     for yi in prange(Ny):
#         for xi in prange(Nx):

#             for ci in range(Nc):
#                 image[yi, xi, ci] = data[yi, xi, ci] // 2**7

#     return image


class FrameDisplay(Display):
    default_1channel_colormaps = ['gray']
    default_2channel_colormaps = ['cyan', 'red']
    default_3channel_colormaps = ['cyan', 'magenta', 'yellow']
    default_lower_limit = 0
    default_upper_limit = 2**11

    def __init__(self, display_queue: queue.Queue, acquisition: Acquisition, processor: Processor):
        super().__init__(display_queue, acquisition, processor)
        
        # Create the LUTs
        if acquisition is not None:
            nchannels = acquisition.spec.nchannels
        else:
            nchannels = processor._spec.nchannels

        if nchannels == 1:
            self.luts = [LookUpTable(
                name=self.default_1channel_colormaps,
                lower_limit=self.default_lower_limit,
                upper_limit=self.default_upper_limit
            )]

        elif nchannels == 2:
            self.luts = [LookUpTable(
                name=cname,
                lower_limit=self.default_lower_limit,
                upper_limit=self.default_upper_limit
            ) for cname in self.default_2channel_colormaps]

        elif nchannels == 3:
            self.luts = [LookUpTable(
                name=cname,
                lower_limit=self.default_lower_limit,
                upper_limit=self.default_upper_limit
            ) for cname in self.default_3channel_colormaps]
                                 
        self.luts_array = np.concatenate(
            [lut.table[np.newaxis,:,:] for lut in self.luts],
            axis=0
        )

    def run(self):
        while True:
            t0 = time.perf_counter()
            data: np.ndarray = self._data_queue.get(block=True) # may want to add a timeout
            t1 = time.perf_counter()

            if data is None: # Check for sentinel None
                self.display_queue.put(None) # pass sentinel
                print('exiting display thread')
                return # concludes run() - this thread ends
            
            processed = display_kernel(data, self.luts_array)

            self.display_queue.put(processed)

            print(f'Made display frame. Waited {t1-t0}')

# for testing
if __name__ == "__main__":

    lut = LookUpTable(name='green', lower_limit=0, upper_limit=16000)

    table = lut.table

    t0 = time.perf_counter()
    table = lut.table
    t1 = time.perf_counter()
    print(t1-t0)