import time
import threading

import numpy as np
from numba import njit, prange, types

from dirigo.sw_interfaces.processor import Processor, ProcessedFrame
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionBuffer
from dirigo.sw_interfaces.display import Display, ColorVector, DisplayChannel, DisplayPixelFormat



@njit(
    (types.uint16[:,:,:,:], types.int64, types.uint16[:,:,:]),
    nogil=True, parallel=True, fastmath=True, cache=True
)
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
    bpp = luts.shape[2]
    transfer_function_max_index = 2**16 - 1
    
    image = np.zeros(shape=(Ny, Nx, bpp), dtype=np.uint8)

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

    def __init__(self, acq: Acquisition, proc: Processor, 
                 display_pixel_format = DisplayPixelFormat.RGB24):
        super().__init__(acq, proc)

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
        self.luts = np.zeros(shape=(self.nchannels, self.data_range.range, bpp), dtype=np.uint16)

        self.display_channels: list[DisplayChannel] = []

        for ci, colormap_name in enumerate(default_colormap_lists(self.nchannels)):
            dc = DisplayChannel(
                lut_slice=self.luts[ci],
                color_vector=ColorVector[colormap_name.upper()],
                display_min=self.data_range.min,
                display_max=self.data_range.max - 1,
                pixel_format=display_pixel_format,
                update_method=self.update_display
            )
            self.display_channels.append(dc)


    def run(self):
        while True:
            # Get new data from inbox
            buf: AcquisitionBuffer | ProcessedFrame = self.inbox.get(block=True) # may want to add a timeout

            if buf is None: # Check for sentinel None
                self.publish(None) # pass sentinel
                print('exiting display thread')
                # Ends thread, but this object can still be used (ie to adjust 
                # the appearance of the last acquired frame)
                return 
            
            # Check whether position has changed
            if buf.positions is not None:
                if (self._prev_position is not None):
                    if isinstance(buf.positions, np.ndarray):
                        current_positions = buf.positions[-1,:] # use the final the position reading
                    elif isinstance(buf.positions, tuple):
                        current_positions = buf.positions

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
                
                self._prev_position = buf.positions
            
            t0 = time.perf_counter()
            with self._average_buffer_lock:
                # _average_buffer is initially None and the first frame establishes height, width, etc
                # Averaging can also be reset or changed by setting this to None, prompting reallocation
                if self._average_buffer is None:
                    buf_shape = buf.data.shape
                    buffer_shape = (self.n_frame_average,) + buf_shape
                    self._average_buffer = np.zeros(buffer_shape, dtype=np.uint16)
                    averaged_frame = np.zeros(buf_shape, np.uint16)
                    self._i = 0

                self._average_buffer[self._i % self.n_frame_average] = buf.data

                rolling_average_kernel(self._average_buffer, self._i, averaged_frame)
                processed = self._apply_display_kernel(averaged_frame, self.luts)

                self.publish(processed)
                self._prev_data = averaged_frame # store reference for use after thread finishes  

            t1 = time.perf_counter()
            self._i += 1
            print(f"Channel display processing: {1000*(t1-t0):.1f}ms. Position data shape: {buf.positions}")

    def _apply_display_kernel(self, average_frame, luts):
        return additive_display_kernel(average_frame, luts)
                     
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


