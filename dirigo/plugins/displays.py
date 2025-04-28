import time
import threading

import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.sw_interfaces.display import Display, ColorVector, DisplayChannel, DisplayPixelFormat


sigs = [
    (types.uint16[:,:,:,:], types.int64, types.uint16[:,:,:]),
    (types.int16[:,:,:,:],  types.int64, types.int16[:,:,:] ),
]
@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def rolling_average_kernel(ring_buffer: np.ndarray, frame_index: int, averaged: np.ndarray) -> np.ndarray:
    Nf, Ny, Nx, Nc = ring_buffer.shape
    d = min(Nf, frame_index + 1)

    for iy in prange(Ny):
        for ix in prange(Nx):
            for ic in range(Nc):

                tmp = 0
                for iframe in range(d):
                    tmp += ring_buffer[iframe, iy, ix, ic]

                averaged[iy, ix, ic] = tmp // d

    return averaged


sigs = [
    types.uint8[:,:,:](types.int16[:,:,:],  types.uint16[:,:,:], types.uint8[:]),
    types.uint8[:,:,:](types.uint16[:,:,:], types.uint16[:,:,:], types.uint8[:]),
    # Not implemented yet: >8bit gamma LUT (for HDR displays)
]
@njit(sigs, nogil=True, parallel=True, fastmath=True, cache=True)
def additive_display_kernel(data: np.ndarray, luts: np.ndarray, gamma_lut: np.ndarray) -> np.ndarray:
    """Applies LUTs and blends channels additively."""
    Ny, Nx, Nc = data.shape
    bpp = luts.shape[2]
    gamma_lut_length = gamma_lut.shape[0]

    image = np.zeros(shape=(Ny, Nx, bpp), dtype=np.uint8)

    for yi in prange(Ny):
        for xi in prange(Nx):

            r, g, b = 0, 0, 0  # Typing these as uint32 for some reason triggers a conversion to float64 which can't be used to index
            for ci in range(Nc):
                lut_index = data[yi, xi, ci]
                r += luts[ci, lut_index, 0]
                g += luts[ci, lut_index, 1] 
                b += luts[ci, lut_index, 2]
            
            # Gamma correct blended values and assign to output image 
            image[yi, xi, 0] = gamma_lut[min(r, gamma_lut_length-1)]
            image[yi, xi, 1] = gamma_lut[min(g, gamma_lut_length-1)]
            image[yi, xi, 2] = gamma_lut[min(b, gamma_lut_length-1)]

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

    def __init__(self, 
                 upstream: Acquisition | Processor, 
                 display_pixel_format = DisplayPixelFormat.RGB24,
                 **kwargs):
        super().__init__(upstream, **kwargs)

        self._prev_data = None # None indicates that no data has been acquired yet
        self._prev_position = None
        
        self._reset_average_xy_cutoff_sqr = self.pixel_size ** 2 # TODO, may require some tuning to remove unwanted resets
        self._reset_average_z_cutoff = self.pixel_size # TODO, dido
        # Either of these may need some cumulative measurement

        # Frame averaging-related
        self._n_frame_average = 1
        self._average_buffer = None
        self._i: int = 0 # tracks rolling average index
        self._average_buffer_lock = threading.Lock()  # Add a lock for thread safety

        bpp = 3 if display_pixel_format == DisplayPixelFormat.RGB24 else 4
        # LUTS: look-up tables (note it's plural). LUTs for each channel enabled for display.
        self._luts = np.zeros(shape=(self.nchannels, self.data_range.range + 1, bpp), dtype=np.uint16)

        self.display_channels: list[DisplayChannel] = []

        for ci, colormap_name in enumerate(default_colormap_lists(self.nchannels)):
            dc = DisplayChannel(
                lut_slice=self._luts[ci],
                color_vector=ColorVector[colormap_name.upper()],
                display_range=self.data_range,
                pixel_format=display_pixel_format,
                update_method=self.update_display,
                gamma_lut_length=self.gamma_lut_length
            )
            self.display_channels.append(dc)

        shape = (
            self._acquisition.spec.lines_per_frame,
            self._acquisition.spec.pixels_per_line,
            bpp
        )
        self.init_product_pool(n=3, shape=shape)
        

    def run(self):
        while True:
            # Get new data from inbox
            prod: AcquisitionProduct | ProcessorProduct = self.inbox.get(block=True) # may want to add a timeout

            if prod is None: # Check for sentinel None
                self.publish(None) # pass sentinel
                print('exiting display thread')
                # Ends thread, but this object can still be used (ie to adjust 
                # the appearance of the last acquired frame)
                return 
            
            # Check whether position has changed
            if prod.positions is not None:
                if (self._prev_position is not None):
                    if isinstance(prod.positions, np.ndarray):
                        current_positions = prod.positions[-1,:] # use the final the position reading
                    elif isinstance(prod.positions, tuple):
                        current_positions = prod.positions

                    dr2 = (current_positions[0] - self._prev_position[0])**2  \
                        + (current_positions[1] - self._prev_position[1])**2
                    
                    if dr2 > self._reset_average_xy_cutoff_sqr: 
                        # if the Euclidean exceeds from last frame exceeds pixel size, then reset the rolling average index
                        # Setting _i = 0, will cause this frame to be sliced into the ring buffer at the first position
                        # and the averaging kernel will only average that 1 frame (effectively no averaging)
                        # this allows reusing the _average_buffer object.
                        self._i = 0

                    elif len(current_positions) > 2: # Z axis also included
                        dz = abs(current_positions[2] - self._prev_position[2])
                        if dz > self._reset_average_z_cutoff:
                            # Similarly, reset _i if z position changes
                            self._i = 0
                
                self._prev_position = prod.positions
            
            t0 = time.perf_counter()
            with self._average_buffer_lock:
                # _average_buffer is initially None and the first frame establishes height, width, etc
                # Averaging can also be reset or changed by setting this to None, prompting reallocation
                if self._average_buffer is None:
                    frame_shape = prod.data.shape
                    ring_frame_buffer_shape = (self.n_frame_average,) + frame_shape
                    self._average_buffer = np.zeros(
                        shape=ring_frame_buffer_shape, 
                        dtype=self.data_range.recommended_dtype
                    )
                    averaged_frame = np.zeros(
                        shape=frame_shape,
                        dtype=self.data_range.recommended_dtype
                    )
                    self._i = 0

                self._average_buffer[self._i % self.n_frame_average] = prod.data

                disp_product = self.get_free_product()
                rolling_average_kernel(self._average_buffer, self._i, averaged_frame)
                disp_product.frame[...] = self._apply_display_kernel(averaged_frame, self._luts)

                self.publish(disp_product)
                
                self._prev_data = averaged_frame # store reference for use after thread finishes  
                prod._release()

            t1 = time.perf_counter()
            self._i += 1
            print(f"Channel display processing: {1000*(t1-t0):.1f}ms. Position data shape: {prod.positions}")

    def _apply_display_kernel(self, average_frame, luts):
        return additive_display_kernel(average_frame, luts, self.gamma_lut)
                     
    @property
    def n_frame_average(self) -> int:
        return self._n_frame_average
    
    @n_frame_average.setter
    def n_frame_average(self, n_frames: int):
        if not isinstance(n_frames, int) or n_frames < 1:
            raise ValueError("Rolling frame average must be an integer >= 1.")
        with self._average_buffer_lock:
            self._n_frame_average = n_frames
            self._average_buffer = None # Triggers reset of the average buffer 

    @property
    def gamma(self) -> float:
        return self._gamma
    
    @gamma.setter
    def gamma(self, new_gamma: float):
        if not isinstance(new_gamma, float):
            raise ValueError("Gamma must be set with a float value")
        if not (0 < new_gamma <= 10):
            raise ValueError("Gamma must be between 0.0 and 10.0")
        self._gamma = new_gamma

        # Generate gamma correction LUT
        x = np.arange(self.gamma_lut_length) \
            / (self.gamma_lut_length - 1) # TODO, not sure about the -1
        
        gamma_lut = (2**self._monitor_bit_depth - 1) * x**(self._gamma)
        if self._monitor_bit_depth > 8:
            self.gamma_lut = np.round(gamma_lut).astype(np.uint16)
        else:
            self.gamma_lut = np.round(gamma_lut).astype(np.uint8)

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
        
        processed = self._apply_display_kernel(self._prev_data, self._luts)
        self.publish(processed)


