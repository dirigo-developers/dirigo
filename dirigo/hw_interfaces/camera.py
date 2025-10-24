from enum import Enum
from abc import abstractmethod
from typing import Optional, Any, List

from dirigo.components import units
from dirigo.sw_interfaces.acquisition import AcquisitionProduct
from dirigo.hw_interfaces.hw_interface import HardwareInterface


class CameraConnectionType(Enum):
    CAMERA_LINK = 0
    GIGE        = 1
    USB         = 2


class FrameGrabber(HardwareInterface):
    attr_name = "frame_grabber"

    def __init__(self):
        self._buffers: List[Any] # TODO refine the buffer object typehint
        self._camera: Optional['Camera'] = None

    @abstractmethod
    def serial_write(self, message):
        pass

    @abstractmethod
    def serial_read(self, nbytes: Optional[int] = None) -> str:
        pass

    @property
    @abstractmethod
    def pixels_width(self) -> int:
        """Total number of sensor pixels wide."""
        pass

    @property
    @abstractmethod
    def roi_height(self) -> int:
        pass

    @property
    @abstractmethod
    def roi_width(self) -> int:
        pass

    @roi_width.setter
    @abstractmethod
    def roi_width(self, width: int):
        pass

    @property
    @abstractmethod
    def roi_left(self) -> int:
        pass

    @roi_left.setter
    @abstractmethod
    def roi_left(self, left: int):
        pass
    
    @property
    @abstractmethod
    def lines_per_buffer(self) -> int:
        pass

    @lines_per_buffer.setter
    @abstractmethod
    def lines_per_buffer(self, lines: int):
        pass

    @property
    @abstractmethod
    def bytes_per_pixel(self) -> int:
        pass

    @property
    def bytes_per_buffer(self):
        if self.lines_per_buffer is None or self.roi_width is None:
            raise RuntimeError("Lines per buffer or ROI width not initialized")
        return self.lines_per_buffer * self.roi_width * self.bytes_per_pixel

    @abstractmethod
    def prepare_buffers(self, nbuffers: int):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_next_completed_buffer(self, product: AcquisitionProduct) -> None:
        pass

    @property
    @abstractmethod
    def buffers_acquired(self) -> int:
        pass

    @property
    @abstractmethod
    def data_range(self) -> units.IntRange:
        """ Returns the range of values returned by the frame grabber. """
        pass


class TriggerModes(Enum):
    FREE_RUN            = 0
    EXTERNAL_TRIGGER    = 1


class Camera(HardwareInterface):
    attr_name = "camera"
    def __init__(self, 
                 frame_grabber: Optional[FrameGrabber], 
                 pixel_size: str, 
                 **kwargs):
        self._frame_grabber = frame_grabber
        if self._frame_grabber is not None:
            self._frame_grabber._camera = self # give the frame grabber reference to camera
        self._pixel_size = units.Length(pixel_size)

    # essential parameters
    # sensor shape (max resolution)
    # roi shape
    # frame rate / interval

    @property
    def pixel_size(self) -> units.Length:
        return self._pixel_size

    @property
    @abstractmethod
    def integration_time(self) -> units.Time:
        pass
    
    @integration_time.setter
    @abstractmethod
    def integration_time(self, new_value: units.Time):
        pass

    @property
    @abstractmethod
    def gain(self):
        pass
    
    @gain.setter
    @abstractmethod
    def gain(self, new_value):
        pass

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        pass
    
    @bit_depth.setter
    @abstractmethod
    def bit_depth(self, new_value: int):
        pass

    @property
    @abstractmethod
    def data_range(self) -> units.IntRange:
        """
        Returns the range of values returned by the camera. 
        
        The returned data range may exceed the bit depth, which can be useful
        for in-place averaging.
        """
        pass

    # IO
    @property
    @abstractmethod
    def trigger_mode(self):
        pass
    
    @trigger_mode.setter
    @abstractmethod
    def trigger_mode(self, new_value):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def load_profile(self):
        pass


class LineCamera(Camera):
    attr_name = "line_camera"
    VALID_AXES = {'x', 'y'} # make these enumerations

    def __init__(self, 
                 axis: str, 
                 flip_line: bool, # only used by Processor
                 **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    @property
    def axis(self):
        return self._axis
    
    @axis.setter
    def axis(self, new_axis: str):
        if new_axis in self.VALID_AXES:
            self._axis = new_axis
        else:
            raise ValueError(f"Error setting encoder axis: Got '{new_axis}'")