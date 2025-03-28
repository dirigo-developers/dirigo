from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional

from dirigo import units


class CameraConnectionType(Enum):
    CAMERA_LINK = 0
    GIGE        = 1
    USB         = 2


class FrameGrabber(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def serial_write(self, message):
        pass

    @abstractmethod
    def serial_read(self):
        pass

    @property
    @abstractmethod
    def pixels_width(self) -> int:
        """Total number of sensor pixels wide."""
        pass

    @property
    @abstractmethod
    def roi_height(self):
        pass

    @property
    @abstractmethod
    def roi_width(self):
        pass

    @roi_width.setter
    @abstractmethod
    def roi_width(self, width: int):
        pass

    @property
    @abstractmethod
    def roi_left(self):
        pass

    @roi_left.setter
    @abstractmethod
    def roi_left(self, left: int):
        pass
    
    @property
    @abstractmethod
    def bytes_per_pixel(self):
        pass

    @property
    def bytes_per_buffer(self):
        return self.roi_height * self.roi_width * self.bytes_per_pixel

    @abstractmethod
    def prepare_buffers(self, nbuffers: int):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @property
    @abstractmethod
    def buffers_acquired(self) -> int:
        pass


class Camera(ABC):
    def __init__(self, frame_grabber: Optional[FrameGrabber], 
                 pixel_size: str, **kwargs):
        self._frame_grabber = frame_grabber 
        self._pixel_size = units.Position(pixel_size)

    # essential parameters
    # sensor shape (max resolution)
    # roi shape
    # frame rate / interval

    @property
    def pixel_size(self):
        return self._pixel_size

    @property
    @abstractmethod
    def integration_time(self):
        pass
    
    @integration_time.setter
    @abstractmethod
    def integration_time(self, new_value):
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
    def bit_depth(self):
        pass
    
    @bit_depth.setter
    @abstractmethod
    def bit_depth(self, new_value):
        pass

    @property
    @abstractmethod
    def data_range(self) -> units.ValueRange:
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


class LineScanCamera(Camera):
    VALID_AXES = {'x', 'y'}

    def __init__(self, axis: str, **kwargs):
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