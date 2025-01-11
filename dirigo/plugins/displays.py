import time

import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces import Display, Acquisition, Processor
from dirigo.sw_interfaces.display import ColorVector, DisplayChannel



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
    transfer_function_max_index = 2**16 - 1
    
    image = np.zeros(shape=(Ny, Nx, 3), dtype=np.uint8)

    for yi in prange(Ny):
        for xi in prange(Nx):

            #r, g, b = types.uint32(0), types.uint32(0), types.uint32(0)
            r, g, b = 0, 0, 0  # Explicitly typing these as uint32 for some reason triggers a conversion to float64 which can't be used to index
            for ci in range(Nc):
                lut_index = data[yi, xi, ci]
                r += luts[ci, lut_index, 0]
                g += luts[ci, lut_index, 1] 
                b += luts[ci, lut_index, 2]
            
            # Gamma correct blended values and assign to output image 
            image[yi, xi, 0] = gamma_lut[min(r, transfer_function_max_index)]
            image[yi, xi, 1] = gamma_lut[min(g, transfer_function_max_index)]
            image[yi, xi, 2] = gamma_lut[min(b, transfer_function_max_index)]

    return image


def default_colormap_lists(nchannels: int) -> list[str]:
    if nchannels == 1:
        return ['gray']
    elif nchannels == 2:
        return ['cyan', 'red']
    elif nchannels == 3:
        return ['cyan', 'magenta', 'yellow']
    elif nchannels == 4:
        return ['cyan', 'magenta', 'yellow', 'gray']


class FrameDisplay(Display):
    """Worker to perform processing for display (blending, LUTs, etc)"""

    def __init__(self, acq: Acquisition, proc: Processor):
        super().__init__(acq, proc)

        data_range = acq.hw.data_range if acq else proc._acq.hw.data_range 

        self._prev_data = None # Indicates that no data has been acquired yet

        self.luts = np.zeros(shape=(self.nchannels, data_range.range, 3), dtype=np.uint16)

        self.display_channels: list[DisplayChannel] = []

        for ci, colormap_name in enumerate(default_colormap_lists(self.nchannels)):
            dc = DisplayChannel(
                lut_slice=self.luts[ci],
                color_vector=ColorVector[colormap_name.upper()],
                display_min=data_range.min,
                display_max=data_range.max - 1,
                update_method=self.update_display
            )
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
                # Ends thread, but this object can still be used (ie to adjust 
                # the appearance of the last acquire frame)
                return 
            
            self._prev_data = data # store reference for use after thread finishes
            t0 = time.perf_counter()
            if self.blending_mode == 'additive':
                processed = additive_display_kernel(data, self.luts)
            # elif self.blending_mode == 'subtractive':
            #     processed = subtractive_display_kernel(data, self.luts)
            t1 = time.perf_counter()
            #print(f"Channel blending: {1000*(t1-t0):.1f}ms")

            self.publish(processed)

    def update_display(self, skip_when_acquisition_in_progress: bool = True):
        """On demand reprocessing of the last acquired frame for display.
        
        Used when the acquisition is stopped and need to update the appearance  
        of the last acquired frame.
        """
        if self.is_alive() and skip_when_acquisition_in_progress:
            # Don't update if acquisition is in progress
            return
        
        if self._prev_data is None:
            # Don't update if no previous data exists
            return
        
        processed = additive_display_kernel(self._prev_data, self.luts)
        self.publish(processed)


