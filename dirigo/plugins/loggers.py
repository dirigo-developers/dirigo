from functools import cached_property
import json, struct
from pathlib import Path
import time
from typing import Sequence

import tifffile
import numpy as np
from scipy import fft
from platformdirs import user_config_dir

from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces import Logger
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.plugins.acquisitions import (
    FrameAcquisitionSpec, FrameSizeCalibration, LineAcquisition
)



def serialize_float64_list(arrays: Sequence[np.ndarray]) -> bytes:
    """
    Pack a sequence of float64 NumPy arrays (all same shape) into one
    compressed bytes object.

    Header layout (little-endian):
        ndims : uint64                        # number of dims per frame
        shape : uint64[ndims]                 # size of each dim

    After the header comes the raw little-endian float64 data for *all*
    frames, laid out as `stack.ravel()` (C-order).
    """
    if not arrays:
        raise ValueError("Empty list")

    ref = arrays[0]
    if ref.dtype != np.float64:
        raise TypeError("dtype must be float64")
    if any((a.shape != ref.shape) or (a.dtype != np.float64) for a in arrays):
        raise ValueError("All arrays must share the same shape and dtype=float64")

    stack = np.stack(arrays, axis=0)                 # shape = (n_frames, *ref.shape)

    ndims  = ref.ndim
    fmt    = f"<Q{ndims}Q"                           # e.g. "<Q2Q" for 2‑D frames
    header = struct.pack(fmt, ndims, *ref.shape)     # bytes

    return header + stack.ravel().tobytes()


class TiffLogger(Logger):
    """
    Saves image stream and metadata to tiff file.

    Private fields: (65000-65535 available as re-usable)

    """
    SYSTEM_CONFIG_TAG      = 65000  # Static info
    RUNTIME_INFO_TAG       = 65001  # Dynamic runtime/driver provided info
    ACQUISITION_SPEC_TAG   = 65100  # General specification for acquisition type
    DIGITIZER_PROFILE_TAG  = 65200  # Rarely changed settings
    CAMERA_PROFILE_TAG     = 65201
    TIMESTAMPS_TAG         = 65400  # Per-frame metadata
    POSITIONS_TAG          = 65401

    def __init__(self, 
                 upstream: Acquisition | Processor,
                 max_frames_per_file: int = 1,
                 **kwargs):
        super().__init__(upstream, **kwargs)

        self.frames_per_file = int(max_frames_per_file) 
        
        self._fn = None
        self._writer = None # generated upon attempt to save first frame
        self.frames_saved = 0
        self.files_saved = 0

        self._timestamps = [] # accumulate as frames arrive from acquistion or processor
        self._positions = []
         
    def run(self):
        try:
            while True:
                product = self.inbox.get(block=True)
                product: AcquisitionProduct | ProcessorProduct 

                if product is None: # Check for sentinel None
                    self.publish(None) # pass sentinel
                    return # thread ends
                
                with product:
                    self.save_data(product)

        finally:
            self._close_and_write_metadata()

    def save_data(self, frame: AcquisitionProduct | ProcessorProduct):
        """Save data and metadata to a TIFF file"""

        # Create the writer object if necessary
        options = {
                'photometric': 'minisblack',
                'planarconfig': 'contig', # switch for single channel
                'resolution': (self._x_dpi, self._y_dpi),
                'contiguous': True
            }
        if self._writer is None:

            if self.frames_per_file == float('inf'):
                self._fn = self.save_path / f"{self.basename}.tif"
            else:
                self._fn = self.save_path / f"{self.basename}_{self.files_saved}.tif"

            self._writer = tifffile.TiffWriter(self._fn, bigtiff=self._use_big_tiff)

            options['extratags'] = self._extra_tags
        else:
            options['contiguous'] = True 

        self._writer.write( 
                frame.data, 
                **options
            )
        self.frames_saved += 1

        # Accumulate timestamps & positions
        self._timestamps.append(frame.timestamps)
        self._positions.append(frame.positions)

        # when number of frames per file reached, close writer & write metadata
        if self.frames_saved % self.frames_per_file == 0:
            self._close_and_write_metadata()
           
    def _close_and_write_metadata(self):
        if self._writer:

            self._writer.close()
            self._writer = None
            self.files_saved += 1

            # write metadata by overwrite (appends data to end of file and
            # 'patches' the offset to point at this new location, tifffile does
            # all of this automatically)
            with tifffile.TiffFile(self._fn, mode='r+b') as tif:

                if len(self._timestamps) > 0:
                    data = serialize_float64_list(self._timestamps)
                    tif.pages[0].tags[self.TIMESTAMPS_TAG].overwrite(data)

                if len(self._positions) > 0:
                    data = serialize_float64_list(self._positions)
                    tif.pages[0].tags[self.POSITIONS_TAG].overwrite(data)

            # Clear accumulants
            self._timestamps = []
            self._positions = []

    @cached_property
    def _use_big_tiff(self) -> bool:
        """Returns False (don't use BigTiff) when frames per file is 1.
        
        This strategy was chosen because little disadvantage to using BigTiff.
        Subclass and overwrite to provide different logic
        """
        if self.frames_per_file == 1:
            return False
        else:
            return True 
        
    # This uses the acquisition or processor references to retrieve resolution
    # An alternative would be to pass resolution (and other metadata) in the queue
    @property
    def _fast_axis_dpi(self) -> float:
        acq = self._acq 
        spec: FrameAcquisitionSpec = acq.spec
        
        pixel_width_inches = ((spec.pixel_size * 1000) / 25.4)
        return 1 / pixel_width_inches
        
    @property
    def _slow_axis_dpi(self) -> float:
        acq = self._acq
        spec: FrameAcquisitionSpec = acq.spec

        if hasattr(spec, 'pixel_height'):
            pixel_height = spec.pixel_height
        else:
            # fallback in case we are processing LineAcquisition data
            pixel_height = spec.pixel_size
        pixel_height_inches = ((pixel_height * 1000) / 25.4)
        return 1 / pixel_height_inches
    
    @cached_property
    def _x_dpi(self) -> float:
        acq = self._acq 
        fast_axis =  acq.hw.fast_raster_scanner.axis
        return self._fast_axis_dpi if fast_axis == 'x' else self._slow_axis_dpi
    
    @cached_property
    def _y_dpi(self) -> float:
        acq = self._acq
        slow_axis =  acq.hw.slow_raster_scanner.axis
        return self._slow_axis_dpi if slow_axis == 'y' else self._fast_axis_dpi

    @cached_property
    def _extra_tags(self) -> list:
        self._acq: LineAcquisition
        
        system_json = json.dumps(self._acq.system_config.to_dict())
        runtime_json = json.dumps(self._acq.runtime_info.to_dict())
        spec_json = json.dumps(self._acq.spec.to_dict())      
        digi_json = json.dumps(self._acq.digitizer_profile.to_dict())

        temp_entry = b' \x00' # Will be patched after metadata is collected
        return [
            (self.SYSTEM_CONFIG_TAG,     's',  0,  system_json,   True),
            (self.RUNTIME_INFO_TAG,      's',  0,  runtime_json,  True),
            (self.ACQUISITION_SPEC_TAG,  's',  0,  spec_json,     True),
            (self.DIGITIZER_PROFILE_TAG, 's',  0,  digi_json,     True),
            (self.TIMESTAMPS_TAG,        'B',  1,  temp_entry,    True),
            (self.POSITIONS_TAG,         'B',  1,  temp_entry,    True)
        ]

class BidiCalibrationLogger(Logger):
    """Logs bidirecetional phase at amplitudes."""
    def __init__(self, upstream: Processor):
        super().__init__(upstream)

        self._amplitudes = []
        self._frequencies = []
        self._phases = []
        
    def run(self):
        try:
            while True:
                product: ProcessorProduct = self.inbox.get(block=True)
                if product is None: return # Check for sentinel None

                with product:
                    self._amplitudes.append(
                        self._acq.hw.fast_raster_scanner.amplitude
                    )
                    self._frequencies.append(product.frequency)
                    self._phases.append(product.phase)
                    
        finally:
            self.publish(None) # pass sentinel
            self.save_data()

    def save_data(self):
        amplitudes = np.unique(self._amplitudes)
        frequencies = []
        phases = []
        for ampl in amplitudes:
            matching_f = []
            matching_p = []
            for a, f, p in zip(self._amplitudes, self._frequencies, self._phases):
                if a == ampl:
                    matching_f.append(f)
                    matching_p.append(p)

            frequencies.append(np.median(matching_f))
            phases.append(np.median(matching_p))

        # stack into a 2-column array
        data = np.column_stack([amplitudes, frequencies, phases])

        # write with a header comment for units
        np.savetxt(
            Path(user_config_dir('Dirigo')) / "scanner/calibration.csv",
            data,
            delimiter=',',
            header='amplitude (rad),frequency (Hz),phase (rad)',
            comments=''    # prevent numpy from prefixing "#" on header lines
        )




class FrameSizeCalibrationLogger(Logger):
    """Logs apparent translation."""
    UPSAMPLE       = 10          # global phase‑corr up‑sampling
    EPS            = 1e-1
    
    def __init__(self, upstream: Processor):
        super().__init__(upstream)
        self._acq: FrameSizeCalibration

    def run(self):
        self._frames, self._positions = [], [] # collect measurement frames/pos
        try:
            while True:
                # Get reference frame
                product: ProcessorProduct = self.inbox.get()
                if product is None: # Check for sentinel None
                    break 
                with product:
                    self._frames.append(product.data[:,:,0].copy())
                    self._positions.append(product.positions)

        finally:
            self.publish(None) # pass sentinel
            self.save_data()

    def save_data(self):
        spec = self._acq.spec
        time.sleep(0.5)
        print("Analyzing displacement field")
        PATCH          = 64 
        STRIDE         = 32 

        # Initial global estimation
        ref_frame, ref_pos = self._frames[0], self._positions[0]
        n_f = len(self._frames)
        n_y, n_x = ref_frame.shape

        dx = int(spec.translation_per_step / spec.pixel_size)

        data = np.zeros(
            shape=(n_x // STRIDE, n_f)
        )

        for f_idx, frame in enumerate(self._frames):
            if f_idx == 0: continue # first column will be field position
            for p_idx in range(n_x // STRIDE):
                p0 = (p_idx * STRIDE)
                ref_patch = ref_frame[:, p0:(p0 + PATCH)]

                m0 = p0 - dx
                if (m0 < 0) or (p0+PATCH >= n_x) or (m0 + PATCH) >= n_x: 
                    data[p_idx, f_idx] = np.nan
                    continue
                mov_patch = frame[:, m0:(m0 + PATCH)]

                i, j = self.x_corr(ref_patch, mov_patch)
                print(i,j)

                data[p_idx, f_idx] = j / dx
                data[p_idx, 0] = ((p0+m0)/2 + PATCH//2 - n_x/2)/n_x

            ref_frame = frame

        np.savetxt(
            Path(user_config_dir('Dirigo')) / "scanner/frame_calibration.csv",
            data,
            delimiter=',',
        )

    @classmethod
    def x_corr(cls, ref_frame, moving_frame):
        n_y, n_x = ref_frame.shape

        xps = fft.rfft2(ref_frame) * np.conj(fft.rfft2(moving_frame))
        s = (n_y * cls.UPSAMPLE, n_x * cls.UPSAMPLE)
        corr = fft.irfft2(xps / (np.abs(xps) + cls.EPS), s)
        arg_max = np.argmax(corr)
        i = arg_max // corr.shape[1]
        j = arg_max %  corr.shape[1]

        if i > (s[0] // 2):  # Handle wrap-around for negative shifts
            i -= s[0]
        if j > (s[1] // 2): 
            j -= s[1]

        return i / cls.UPSAMPLE, j / cls.UPSAMPLE