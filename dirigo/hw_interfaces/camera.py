from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional


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
    def roi_height(self):
        pass

    @property
    @abstractmethod
    def roi_width(self):
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

    @property
    @abstractmethod
    def buffers_acquired(self) -> int:
        pass


class Camera(ABC):
    def __init__(self, frame_grabber: Optional[FrameGrabber], **kwargs):

        self._frame_grabber = frame_grabber 

    # essential parameters
    # sensor shape (max resolution)
    # roi shape
    # frame rate / interval
    # 

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
    def bits_per_pixel(self):
        pass
    
    @bits_per_pixel.setter
    @abstractmethod
    def bits_per_pixel(self, new_value):
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
    pass