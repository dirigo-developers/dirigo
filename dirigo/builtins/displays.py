from functools import cached_property
import queue
import time

import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces import Display, Acquisition, Processor


# TODO, 16 bit LUT is a bit clunky, investigate whether we can stream line this


# TODO allow inverted color maps (start triplets not all 0's)
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
    def table(self) -> np.ndarray:
        # x is all possible pixel values
        x = np.arange(
            start=-2**(self.nbits-1), 
            stop=2**(self.nbits-1), 
            dtype=np.float32
        ) 

        clipped = np.clip((x - self.lower_limit) / self.range, a_min = 0, a_max=1)
        y = np.round(clipped * (2**self.displaybits-1)).astype(np.uint8)

        return y[:, np.newaxis] * np.array(self.end_triplet, np.uint8)[np.newaxis, :]


@njit(
    types.uint8[:,:,:](
        types.int16[:,:,:], 
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
            tmp = np.zeros((3,), np.uint8)
            for ci in range(Nc):
                tmp += luts[ci, data[yi, xi, ci], :]
            
            image[yi, xi, :] = tmp

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
    colormap_list = ['cyan', 'red']
    def __init__(self, display_queue: queue.Queue, acquisition: Acquisition, processor: Processor):
        super().__init__(display_queue, acquisition, processor)
        self.luts = [LookUpTable(name=cname, lower_limit=0, upper_limit=2**12) 
                     for cname in self.colormap_list]
        
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

            print(f'Would display an image. Waited {t1-t0}')


# for testing
if __name__ == "__main__":

    lut = LookUpTable(name='green', lower_limit=0, upper_limit=16000)

    print(lut.table)