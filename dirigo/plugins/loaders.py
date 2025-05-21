from pathlib import Path
import json, struct

import tifffile
import numpy as np

from dirigo.sw_interfaces.acquisition import Loader
from dirigo.plugins.loggers import TiffLogger
from dirigo.components.io import SystemConfig
from dirigo.plugins.acquisitions import LineAcquisitionRuntimeInfo, FrameAcquisitionSpec
from dirigo.hw_interfaces.digitizer import DigitizerProfile



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
    Spec = FrameAcquisitionSpec   # can be overridden
    
    def __init__(self, file_path: str | Path):
        super().__init__(file_path, thread_name="Frame loader")

        with tifffile.TiffFile(self._file_path) as tif:   

            self.init_product_pool(
                n=4, 
                shape=tif.pages[0].shape, 
                dtype=tif.pages[0].dtype
            )

            tags = tif.pages[0].tags

            cfg_dict = json.loads(tags[TiffLogger.SYSTEM_CONFIG_TAG].value)
            self.system_config = SystemConfig(**cfg_dict)

            runtime_dict = json.loads(tags[TiffLogger.RUNTIME_INFO_TAG].value)
            self.runtime_info = LineAcquisitionRuntimeInfo.from_dict(runtime_dict)

            spec_dict = json.loads(tags[TiffLogger.ACQUISITION_SPEC_TAG].value)
            self.spec = self.Spec(**spec_dict)

            digi_dict = json.loads(tags[TiffLogger.DIGITIZER_PROFILE_TAG].value)
            self.digitizer_profile = DigitizerProfile.from_dict(digi_dict)

            self.frames_read = 0

    def run(self):
        try:
            with tifffile.TiffFile(self._file_path) as tif:   
                
                n_frames = len(tif.pages)
                
                self._timestamps = deserialize_float64_list(
                    tif.pages[0].tags[TiffLogger.TIMESTAMPS_TAG].value
                )
                self._positions = deserialize_float64_list(
                    tif.pages[0].tags[TiffLogger.POSITIONS_TAG].value
                )

                while self.frames_read < n_frames:
                    frame = self.get_free_product()

                    # Copy raw data
                    frame.data[...] = tif.pages[self.frames_read].asarray()

                    # Copy metadata
                    frame.timestamps = self._timestamps[self.frames_read]
                    frame.positions = self._positions[self.frames_read]

                    print(f"publishing frame {self.frames_read}")
                    self.publish(frame)

                    self.frames_read += 1
        finally:
            self.publish(None) # sentinel coding finished