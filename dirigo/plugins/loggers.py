from functools import cached_property
import json, struct
from typing import Sequence

import tifffile
import numpy as np

from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces import Logger
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.hw_interfaces import Digitizer
from dirigo.plugins.acquisitions import SampleAcquisitionSpec, FrameAcquisition, FrameAcquisitionSpec


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

    
    if isinstance(arrays[0], tuple):
        # try converting to numpy array
        arrays = [np.array(arr) for arr in arrays]
    
    ref = arrays[0]

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
        self.file_ext = "tif"

        self.frames_per_file = int(max_frames_per_file) 
        
        #self._fn = None
        self._writer = None # generated upon attempt to save first frame
        self.frames_saved = 0
        self.files_saved = 0

        try:
            if upstream.product_shape[2] == 3 and upstream.product_dtype == np.uint8:
                self._photometric = 'rgb'
            else:
                self._photometric = 'minisblack'
        except RuntimeError: # when product pool not yet initialized
            self._photometric = 'minisblack'

        self._timestamps = [] # accumulate as frames arrive from acquistion or processor
        self._positions = []

    def _receive_product(self) ->  AcquisitionProduct | ProcessorProduct:
        return super()._receive_product(self) # type: ignore
         
    def run(self):
        try:
            while True:
                with self._receive_product() as product:
                    self.save_data(product)

        except EndOfStream:
            self._publish(None)

        finally:
            self._close_and_write_metadata()

    def save_data(self, frame: AcquisitionProduct | ProcessorProduct):
        """Save data and metadata to a TIFF file"""

        # Create the writer object if necessary
        options = {
                'photometric': self._photometric,
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

        if isinstance(self._acquisition.spec, SampleAcquisitionSpec):
            if sum(c.enabled for c in self._acquisition.digitizer_profile.channels) > 1:
                options['planarconfig'] = 'contig'

        self._writer.write(frame.data, **options)
        self.frames_saved += 1

        # Accumulate timestamps & positions
        if hasattr(frame, 'timestamps') and frame.timestamps is not None:
            self._timestamps.append(frame.timestamps)
        if hasattr(frame, 'positions') and frame.positions is not None:
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
            with tifffile.TiffFile(self._fn, mode='r+b') as tif: # type: ignore

                if len(self._timestamps) > 0:
                    data = serialize_float64_list(self._timestamps)
                    tif.pages[0].tags[self.TIMESTAMPS_TAG].overwrite(data) # type: ignore

                if len(self._positions) > 0:
                    data = serialize_float64_list(self._positions)
                    tif.pages[0].tags[self.POSITIONS_TAG].overwrite(data) # type: ignore

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
        acq = self._acquisition 
        spec: FrameAcquisitionSpec = acq.spec
        
        pixel_width_inches = ((spec.pixel_size * 1000) / 25.4)
        return 1 / pixel_width_inches
        
    @property
    def _slow_axis_dpi(self) -> float:
        acq = self._acquisition
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
        fast_axis = self._acquisition.system_config.fast_raster_scanner['axis']
        return self._fast_axis_dpi if fast_axis == 'x' else self._slow_axis_dpi
    
    @cached_property
    def _y_dpi(self) -> float:
        slow_axis = self._acquisition.system_config.slow_raster_scanner['axis']
        return self._slow_axis_dpi if slow_axis == 'y' else self._fast_axis_dpi

    @cached_property
    def _extra_tags(self) -> list:
        self._acquisition: FrameAcquisition
        
        system_json = json.dumps(self._acquisition.system_config.to_dict())
        runtime_json = json.dumps(self._acquisition.runtime_info.to_dict())
        spec_json = json.dumps(self._acquisition.spec.to_dict())      
        temp_entry = b' \x00' # temp will be patched (overwritten) later

        if isinstance(self._acquisition.spec, SampleAcquisitionSpec):
            digi_json = json.dumps(self._acquisition.digitizer_profile.to_dict())
            return [
                (self.SYSTEM_CONFIG_TAG,     's',  0,  system_json,   True),
                (self.RUNTIME_INFO_TAG,      's',  0,  runtime_json,  True),
                (self.ACQUISITION_SPEC_TAG,  's',  0,  spec_json,     True),
                (self.DIGITIZER_PROFILE_TAG, 's',  0,  digi_json,     True),
                (self.TIMESTAMPS_TAG,        'B',  1,  temp_entry,    True),
                (self.POSITIONS_TAG,         'B',  1,  temp_entry,    True)
            ]
        else:
            #cam_json = json.dumps(self._acq.camera_profile.to_dict()) # TODO make camera profile & to_dict()
            return [
                (self.SYSTEM_CONFIG_TAG,     's',  0,  system_json,   True),
                (self.RUNTIME_INFO_TAG,      's',  0,  runtime_json,  True),
                (self.ACQUISITION_SPEC_TAG,  's',  0,  spec_json,     True),
                #(self.CAMERA_PROFILE_TAG,    's',  0,  cam_json,      True),
                (self.TIMESTAMPS_TAG,        'B',  1,  temp_entry,    True),
                (self.POSITIONS_TAG,         'B',  1,  temp_entry,    True)
            ]

