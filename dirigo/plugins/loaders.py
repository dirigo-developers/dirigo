from pathlib import Path
import json, struct
from typing import Type

import tifffile
import numpy as np

from dirigo.sw_interfaces.acquisition import Loader
from dirigo.plugins.writers import (
    read_system_config, read_acquisition_spec, read_runtime_info,
    read_digitizer_profile, read_positions, read_timestamps,
    read_sequence_indices, read_strip_indices, read_depth_indices, read_volume_indices
)
from dirigo.plugins.acquisitions import FrameAcquisitionSpec, LineCameraAcquisitionSpec



def deserialize_float64_list(blob: bytes):
    """
    Reverse of `serialize_float64_list` (full shape in header).
    Returns a list of np.ndarray, all copies (writable).
    """
    ndims, = struct.unpack_from("<Q", blob, 0)

    fmt          = f"<Q{ndims}Q"
    header_size  = struct.calcsize(fmt)
    shape        = struct.unpack_from(fmt, blob, 0)[1:]

    items_per_frame = np.prod(shape)
    bytes_per_frame = items_per_frame * 8
    n_frames        = (len(blob) - header_size) // bytes_per_frame

    data   = np.frombuffer(blob, dtype=np.float64, offset=header_size)
    stack  = data.reshape((n_frames, *shape))
    return [stack[i].copy() for i in range(n_frames)]


class RawRasterFrameLoader(Loader):   
    def __init__(
        self, 
        file_path: str | Path, 
        spec_class: Type[FrameAcquisitionSpec] = FrameAcquisitionSpec
    ):
        super().__init__(file_path, thread_name="Raster Frame loader")

        with tifffile.TiffFile(self._file_path) as tif:   
            page = tif.pages[0]
            self._init_product_pool(n=4, shape=page.shape, dtype=page.dtype)

        self.system_config = read_system_config(self._file_path)
        self.spec = spec_class(**read_acquisition_spec(self._file_path))
        self.runtime_info = read_runtime_info(self._file_path)
        self.digitizer_profile = read_digitizer_profile(self._file_path)

        self._timestamps = read_timestamps(self._file_path)
        self._positions = read_positions(self._file_path)
        self._sequence_indices = read_sequence_indices(self._file_path)
        self._strip_indices = read_strip_indices(self._file_path)
        self._depth_indices = read_depth_indices(self._file_path)
        self._volume_indices = read_volume_indices(self._file_path)

        self.frames_read = 0

    def _work(self):
        try:
            with tifffile.TiffFile(self._file_path) as tif:   
                
                while self.frames_read < len(tif.pages):
                    frame = self._get_free_product()

                    # Copy raw data
                    frame.data[...] = tif.pages[self.frames_read].asarray()

                    # Copy metadata, if available
                    if self._timestamps:
                        frame.timestamps = self._timestamps[self.frames_read]
                    if self._positions:
                        frame.positions = self._positions[self.frames_read]
                    if self._sequence_indices:
                        frame.sequence_index = self._sequence_indices[self.frames_read]
                    if self._strip_indices:
                        frame.strip_index = self._strip_indices[self.frames_read]
                    if self._depth_indices:
                        frame.depth_index = self._depth_indices[self.frames_read]
                    if self._volume_indices:
                        frame.volume_index = self._volume_indices[self.frames_read]

                    print(f"publishing frame {self.frames_read}")
                    self._publish(frame)

                    self.frames_read += 1
        finally:
            self._publish(None) # sentinel coding finished


class RawCameraFrameLoader(Loader):
    def __init__(
        self, 
        file_path: str | Path, 
        spec_class: Type[LineCameraAcquisitionSpec] = LineCameraAcquisitionSpec
    ):
        super().__init__(file_path, thread_name="Camera Frame loader")

        with tifffile.TiffFile(self._file_path) as tif:   
            page = tif.pages[0]
            self._init_product_pool(n=4, shape=page.shape, dtype=page.dtype)

        self.system_config = read_system_config(self._file_path)
        self.spec = spec_class(**read_acquisition_spec(self._file_path))
        self.runtime_info = read_runtime_info(self._file_path)
        # self.camera_profile = read_camera_profile(self._file_path)

        self._timestamps = None
        self._positions = read_positions(self._file_path)
        self._sequence_indices = read_sequence_indices(self._file_path)
        self._strip_indices = read_strip_indices(self._file_path)
        self._depth_indices = read_depth_indices(self._file_path)
        self._volume_indices = read_volume_indices(self._file_path)

        self.frames_read = 0

    def _work(self):
        try:
            with tifffile.TiffFile(self._file_path) as tif:   

                while self.frames_read < len(tif.pages):
                    frame = self._get_free_product()

                    # Copy raw data
                    frame.data[...] = tif.pages[self.frames_read].asarray()

                    # Copy metadata, if available
                    if self._timestamps:
                        frame.timestamps = self._timestamps[self.frames_read]
                    if self._positions:
                        frame.positions = self._positions[self.frames_read]
                    if self._sequence_indices:
                        frame.sequence_index = self._sequence_indices[self.frames_read]
                    if self._strip_indices:
                        frame.strip_index = self._strip_indices[self.frames_read]
                    if self._depth_indices:
                        frame.depth_index = self._depth_indices[self.frames_read]
                    if self._volume_indices:
                        frame.volume_index = self._volume_indices[self.frames_read]

                    print(f"publishing frame {self.frames_read}")
                    self._publish(frame)

                    self.frames_read += 1
        finally:
            self._publish(None) # sentinel coding finished