from functools import cached_property
import json, struct
from typing import Sequence, Literal
from pathlib import Path

import tifffile
import numpy as np

from dirigo.sw_interfaces.worker import EndOfStream
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces import Writer
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct
from dirigo.hw_interfaces.digitizer import DigitizerProfile
from dirigo.plugins.acquisitions import (
    SampleAcquisitionSpec, FrameAcquisition, FrameAcquisitionSpec, 
    StackAcquisitionSpec, LineAcquisitionRuntimeInfo
)
from dirigo.components.io import SystemConfig
    

_INDEX_DTYPE = np.dtype("<u8")  # little-endian uint64
_FLOAT_DTYPE = np.dtype("<f8")  # little-endian float64
_TEMP_ENTRY = b"\x00"


def _serialize_float64_list(arrays: Sequence[np.ndarray]) -> bytes:
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

    return header + stack.astype(_FLOAT_DTYPE, copy=False).ravel().tobytes()


def _deserialize_float64_list(blob: bytes):
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

    data   = np.frombuffer(blob, dtype=_FLOAT_DTYPE, offset=header_size)
    stack  = data.reshape((n_frames, *shape))
    return [stack[i].copy() for i in range(n_frames)]


def _serialize_uint64_list(values: list[int]) -> bytes:
    """Serialize non-negative integer indices as little-endian uint64 values."""
    return np.asarray(values, dtype=_INDEX_DTYPE).tobytes(order="C")


def _deserialize_uint64_list(blob: bytes) -> list[int]:
    """Deserialize a little-endian uint64 index vector."""
    return np.frombuffer(blob, dtype=_INDEX_DTYPE).tolist()
    

class TiffWriter(Writer):
    """
    Saves image stream and metadata to tiff file.

    Private fields: (65000-65535 available as re-usable)

    """
    SYSTEM_CONFIG_TAG = 65000  # Static info
    RUNTIME_INFO_TAG = 65001  # Dynamic runtime/driver provided info
    ACQUISITION_SPEC_TAG = 65100  # General specification for acquisition type
    DIGITIZER_PROFILE_TAG = 65200  # Rarely changed settings
    CAMERA_PROFILE_TAG = 65201
    TIMESTAMPS_TAG = 65400  # Per-frame metadata
    POSITIONS_TAG = 65401
    SEQUENCE_INDEX_TAG = 65402
    STRIP_INDEX_TAG = 65403
    DEPTH_INDEX_TAG = 65404
    VOLUME_INDEX_TAG = 65405

    def __init__(self, 
                 upstream: Acquisition | Processor,
                 max_frames_per_file: int = 1,
                 mode: Literal['z-stack', 't-series'] = 't-series',   # TODO make this an enum
                 **kwargs):
        super().__init__(upstream, **kwargs)
        self.file_ext = "tif"

        self.frames_per_file = int(max_frames_per_file) 
        
        self._writer = None # generated upon attempt to save first frame
        self.frames_saved = 0
        self.files_saved = 0
        self.mode = mode

        try:
            if upstream.product_shape[2] == 3 and upstream.product_dtype == np.uint8:
                self._photometric = 'rgb'
            else:
                self._photometric = 'minisblack'
        except RuntimeError: # when product pool not yet initialized
            self._photometric = 'minisblack'

        self._timestamps = [] # accumulate as frames arrive from acquistion or processor
        self._positions = []
        self._sequence_indices = []
        self._strip_indices = []
        self._depth_indices = []
        self._volume_indices = []

        self._stack_data = []

    def _receive_product(self) ->  AcquisitionProduct | ProcessorProduct:
        return super()._receive_product() # type: ignore
         
    def _work(self):
        try:
            while True:
                with self._receive_product() as product:
                    self.save_data(product)

        except EndOfStream:
            self._publish(None)

        finally:
            if self.mode == 't-series':
                self._close_and_write_metadata()
            elif self.mode == 'z-stack':
                self._write_stack()

    def save_data(self, frame: AcquisitionProduct | ProcessorProduct):
        """Save data and metadata to a TIFF file"""

        if self.mode == 't-series':
            self._save_frame(frame)
        elif self.mode == 'z-stack':
            self._store_z_plane(frame)
        else:
            raise ValueError("Unsupported TiffWriter mode: {self.mode}")

    def _accumulate_frame_metadata(
        self,
        frame: AcquisitionProduct | ProcessorProduct,
    ) -> None:
        
        if frame.timestamps is not None:
            self._timestamps.append(
                np.asarray(frame.timestamps, dtype=np.float64).copy()
            )

        if frame.positions is not None:
            self._positions.append(
                np.asarray(frame.positions, dtype=np.float64).copy()
            )

        if frame.sequence_index is not None:
            self._sequence_indices.append(int(frame.sequence_index))

        if frame.strip_index is not None:
            self._strip_indices.append(int(frame.strip_index))

        if frame.depth_index is not None:
            self._depth_indices.append(int(frame.depth_index))

        if frame.volume_index is not None:
            self._volume_indices.append(int(frame.volume_index))

    def _save_frame(self, frame: AcquisitionProduct | ProcessorProduct):
        # Create the writer object if necessary
        options = {
            'photometric':  self._photometric,
            'resolution':   (self._x_dpi, self._y_dpi),
            'contiguous':   True,
        }
        
        if self._writer is None:
            if self.frames_per_file == float('inf'):
                self._fn = self.save_path / f"{self.basename}.tif"
            else:
                self._fn = self.save_path / f"{self.basename}_{self.files_saved}.tif"
            self._writer = tifffile.TiffWriter(self._fn, bigtiff=self._use_big_tiff)
            options['extratags'] = self._extra_tags

        if isinstance(self._acquisition.spec, SampleAcquisitionSpec):
            if sum(c.enabled for c in self._acquisition.digitizer_profile.channels) > 1:
                options['planarconfig'] = 'contig'

        print("Saving shape", frame.data.shape)
        self._writer.write(
            data        = frame.data, 
            metadata    = {'axes': 'TYXC'},
            **options
        )
        self.frames_saved += 1

        self._accumulate_frame_metadata(frame)

        # when number of frames per file reached, close writer & write metadata
        if self.frames_saved % self.frames_per_file == 0:
            self._close_and_write_metadata()

    def _store_z_plane(self, frame: AcquisitionProduct | ProcessorProduct):
        self._stack_data.append(frame.data.copy())
        print(f"stored z plane {len(self._stack_data)}")

    def _close_and_write_metadata(self):
        if self._writer:
            
            self.last_saved_file_path = self._fn
            self._writer.close()
            self._writer = None
            self.files_saved += 1

            has_frame_metadata = any((
                self._timestamps,
                self._positions,
                self._sequence_indices,
                self._strip_indices,
                self._depth_indices,
                self._volume_indices,
            ))

            if has_frame_metadata:
                # write metadata by overwrite (appends data to end of file and
                # patches the offset to point at this new location, tifffile does
                # all of this automatically)
                with tifffile.TiffFile(self._fn, mode='r+b') as tif: # type: ignore
                    page = tif.pages[0]

                    if self._timestamps:
                        page.tags[self.TIMESTAMPS_TAG].overwrite(
                            _serialize_float64_list(self._timestamps)
                        )

                    if self._positions:
                        page.tags[self.POSITIONS_TAG].overwrite(
                            _serialize_float64_list(self._positions)
                        )

                    for tag_code, values in (
                        (self.SEQUENCE_INDEX_TAG, self._sequence_indices),
                        (self.STRIP_INDEX_TAG, self._strip_indices),
                        (self.DEPTH_INDEX_TAG, self._depth_indices),
                        (self.VOLUME_INDEX_TAG, self._volume_indices),
                    ):
                        if values:
                            page.tags[tag_code].overwrite(
                                _serialize_uint64_list(values)
                            )

            # Clear accumulants
            self._timestamps.clear()
            self._positions.clear()
            self._sequence_indices.clear()
            self._strip_indices.clear()
            self._depth_indices.clear()
            self._volume_indices.clear()
    
    def _write_stack(self):
        spec: StackAcquisitionSpec = self._acquisition.spec     # type: ignore
        options = {
            'photometric':  self._photometric,
            'resolution':   (self._x_dpi, self._y_dpi),
        }
        metadata = {
            'axes': 'ZYXC',
            'PhysicalSizeX': float(spec.pixel_size),
            'PhysicalSizeXUnit': 'm',
            'PhysicalSizeY': float(spec.pixel_size),
            'PhysicalSizeYUnit': 'm',
            'PhysicalSizeZ': float(spec.depth_spacing),
            'PhysicalSizeZUnit': 'm',
        }
        if isinstance(self._acquisition.spec, SampleAcquisitionSpec):
            if sum(c.enabled for c in self._acquisition.digitizer_profile.channels) > 1:
                options['planarconfig'] = 'contig'

        self._fn = self.save_path / f"{self.basename}.tif"
        
        with tifffile.TiffWriter(self._file_path(), bigtiff=True, ome=True) as tif:
            tif.write(
                np.array(self._stack_data),
                metadata=metadata,
                **options
            )

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

        per_frame_tags = [
            (self.TIMESTAMPS_TAG,     "B", 1, _TEMP_ENTRY, True),
            (self.POSITIONS_TAG,      "B", 1, _TEMP_ENTRY, True),
            (self.SEQUENCE_INDEX_TAG, "B", 1, _TEMP_ENTRY, True),
            (self.STRIP_INDEX_TAG,    "B", 1, _TEMP_ENTRY, True),
            (self.DEPTH_INDEX_TAG,    "B", 1, _TEMP_ENTRY, True),
            (self.VOLUME_INDEX_TAG,   "B", 1, _TEMP_ENTRY, True),
        ]
        
        if isinstance(self._acquisition.spec, SampleAcquisitionSpec):
            digi_json = json.dumps(self._acquisition.digitizer_profile.to_dict())
            return [
                (self.SYSTEM_CONFIG_TAG,     's',  0,  system_json,   True),
                (self.RUNTIME_INFO_TAG,      's',  0,  runtime_json,  True),
                (self.ACQUISITION_SPEC_TAG,  's',  0,  spec_json,     True),
                (self.DIGITIZER_PROFILE_TAG, 's',  0,  digi_json,     True),
                *per_frame_tags,
            ]
        else:
            #cam_json = json.dumps(self._acq.camera_profile.to_dict()) # TODO make camera profile & to_dict()
            return [
                (self.SYSTEM_CONFIG_TAG,     's',  0,  system_json,   True),
                (self.RUNTIME_INFO_TAG,      's',  0,  runtime_json,  True),
                (self.ACQUISITION_SPEC_TAG,  's',  0,  spec_json,     True),
                #(self.CAMERA_PROFILE_TAG,    's',  0,  cam_json,      True),
                *per_frame_tags,
            ]



def read_system_config(filepath: Path):

    with tifffile.TiffFile(filepath) as tif:
        if len(tif.pages) == 0:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        page = tif.pages[0]
        tag = page.tags.get(TiffWriter.SYSTEM_CONFIG_TAG)

        if tag is None:
            raise KeyError(
                f"TIFF file has no System Config tag "
                f"({TiffWriter.SYSTEM_CONFIG_TAG}): {filepath}"
            )

        value = tag.value

    if not isinstance(value, str):
        raise ValueError(
            f"SystemConfig tag has unexpected type {type(value).__name__}; "
            "expected str"
        )
    
    text = value.rstrip("\x00")

    try:
        system_config_dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"SystemConfig tag in {filepath} does not contain valid JSON"
        ) from exc

    return SystemConfig.from_dict(system_config_dict)


def read_acquisition_spec(filepath: Path):
    """Returns a dictionary of values for the Acquisition Spec."""

    with tifffile.TiffFile(filepath) as tif:
        if len(tif.pages) == 0:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        page = tif.pages[0]
        tag = page.tags.get(TiffWriter.ACQUISITION_SPEC_TAG)

        if tag is None:
            raise KeyError(
                f"TIFF file has no Acquisition Spec tag "
                f"({TiffWriter.ACQUISITION_SPEC_TAG}): {filepath}"
            )

        value = tag.value

    if not isinstance(value, str):
        raise ValueError(
            f"Acquisition Spec tag has unexpected type {type(value).__name__}; "
            "expected str"
        )
    
    text = value.rstrip("\x00")

    try:
        acquisition_spec_dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Acquisition Spec tag in {filepath} does not contain valid JSON"
        ) from exc

    return acquisition_spec_dict


def read_runtime_info(filepath: Path):
    """Returns a dictionary of values for the Runtime Info."""

    with tifffile.TiffFile(filepath) as tif:
        if len(tif.pages) == 0:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        page = tif.pages[0]
        tag = page.tags.get(TiffWriter.RUNTIME_INFO_TAG)

        if tag is None:
            raise KeyError(
                f"TIFF file has no Runtime Info tag "
                f"({TiffWriter.RUNTIME_INFO_TAG}): {filepath}"
            )

        value = tag.value

    if not isinstance(value, str):
        raise ValueError(
            f"Runtime Info tag has unexpected type {type(value).__name__}; "
            "expected str"
        )
    
    text = value.rstrip("\x00")

    try:
        runtime_info_dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Runtime Info tag in {filepath} does not contain valid JSON"
        ) from exc

    return LineAcquisitionRuntimeInfo.from_dict(runtime_info_dict)


def read_digitizer_profile(filepath: Path) -> DigitizerProfile:
    with tifffile.TiffFile(filepath) as tif:
        if len(tif.pages) == 0:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        page = tif.pages[0]
        tag = page.tags.get(TiffWriter.DIGITIZER_PROFILE_TAG)

        if tag is None:
            raise KeyError(
                f"TIFF file has no Digitizer Profile tag "
                f"({TiffWriter.DIGITIZER_PROFILE_TAG}): {filepath}"
            )

        value = tag.value

    if not isinstance(value, str):
        raise ValueError(
            f"Digitizter Profile tag has unexpected type {type(value).__name__}; "
            "expected str"
        )
    
    text = value.rstrip("\x00")

    try:
        digitizer_profile_dict = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Digitizer Profile tag in {filepath} does not contain valid JSON"
        ) from exc

    return DigitizerProfile.from_dict(digitizer_profile_dict)


def read_timestamps(filepath: Path):

    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        tag = tif.pages[0].tags.get(TiffWriter.TIMESTAMPS_TAG)
        if tag is None or (tag.value == _TEMP_ENTRY):
           raise ValueError(
               f"Timestamps tag in {filepath} does not contain value position data."
           )

        return _deserialize_float64_list(tag.value)


def read_positions(filepath: Path):
    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        tag = tif.pages[0].tags.get(TiffWriter.POSITIONS_TAG)
        if tag is None or (tag.value == _TEMP_ENTRY):
           raise ValueError(
               f"Positions tag in {filepath} does not contain value position data."
           )

        return _deserialize_float64_list(tag.value)


def read_sequence_indices(filepath: Path):
    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        tag = tif.pages[0].tags.get(TiffWriter.SEQUENCE_INDEX_TAG)
        if tag is None or (tag.value == _TEMP_ENTRY):
            return None
        else:
            return _deserialize_uint64_list(tag.value)


def read_strip_indices(filepath: Path):
    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        tag = tif.pages[0].tags.get(TiffWriter.STRIP_INDEX_TAG)
        if tag is None or (tag.value == _TEMP_ENTRY):
            return None
        else:
            return _deserialize_uint64_list(tag.value)


def read_depth_indices(filepath: Path):
    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        tag = tif.pages[0].tags.get(TiffWriter.DEPTH_INDEX_TAG)
        if tag is None or (tag.value == _TEMP_ENTRY):
            return None
        else:
            return _deserialize_uint64_list(tag.value)


def read_volume_indices(filepath: Path):
    with tifffile.TiffFile(filepath) as tif:
        if not tif.pages:
            raise ValueError(f"TIFF file contains no pages: {filepath}")

        tag = tif.pages[0].tags.get(TiffWriter.VOLUME_INDEX_TAG)
        if tag is None or (tag.value == _TEMP_ENTRY):
            return None
        else:
            return _deserialize_uint64_list(tag.value)