from abc import ABC, abstractmethod
from typing import Optional

"""
Dirigo camera interface.

The camera object is a composition of abstact subclasses designed to emulate
GenICam's Standard Features Naming Conventions (SFNC).
"""



class AcquisitionControl(ABC):
    """Abstract base for SFNC-like acquisition controls."""

    @property
    @abstractmethod
    def trigger_mode(self) -> str:
        """
        Gets/sets the trigger mode (e.g., "Off", "On", "Software", "External").
        SFNC name: 'TriggerMode'
        """
        pass

    @trigger_mode.setter
    @abstractmethod
    def trigger_mode(self, mode: str):
        pass

    @property
    @abstractmethod
    def trigger_source(self) -> str:
        """
        SFNC name: 'TriggerSource'
        """
        pass

    @trigger_source.setter
    @abstractmethod
    def trigger_source(self, source: str):
        pass

    @property
    @abstractmethod
    def exposure_time(self) -> float:
        """
        Exposure time in microseconds, for example.
        SFNC name: 'ExposureTime'
        """
        pass

    @exposure_time.setter
    @abstractmethod
    def exposure_time(self, microseconds: float):
        pass

    @property
    @abstractmethod
    def acquisition_frame_rate(self) -> float:
        """
        Frames per second if camera can run in free-run mode.
        SFNC name: 'AcquisitionFrameRate'
        """
        pass

    @acquisition_frame_rate.setter
    @abstractmethod
    def acquisition_frame_rate(self, fps: float):
        pass

    @abstractmethod
    def start_acquisition(self):
        """SFNC name: 'AcquisitionStart'"""
        pass

    @abstractmethod
    def stop_acquisition(self):
        """SFNC name: 'AcquisitionStop'"""
        pass


class ImageFormatControl(ABC):
    """Abstract base for SFNC-like image format controls."""

    @property
    @abstractmethod
    def width(self) -> int:
        """SFNC name: 'Width'"""
        pass

    @width.setter
    @abstractmethod
    def width(self, pixels: int):
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        """SFNC name: 'Height'"""
        pass

    @height.setter
    @abstractmethod
    def height(self, pixels: int):
        pass

    @property
    @abstractmethod
    def pixel_format(self) -> str:
        """
        SFNC name: 'PixelFormat'
        Could be e.g. "Mono8", "BayerRG8", "RGB8", etc.
        """
        pass

    @pixel_format.setter
    @abstractmethod
    def pixel_format(self, fmt: str):
        pass

    @property
    @abstractmethod
    def binning_horizontal(self) -> int:
        """SFNC name: 'BinningHorizontal'"""
        pass

    @binning_horizontal.setter
    @abstractmethod
    def binning_horizontal(self, value: int):
        pass

    @property
    @abstractmethod
    def binning_vertical(self) -> int:
        """SFNC name: 'BinningVertical'"""
        pass

    @binning_vertical.setter
    @abstractmethod
    def binning_vertical(self, value: int):
        pass


class AnalogControl(ABC):
    """Abstract base for SFNC-like analog controls (gain, black level, etc.)."""

    @property
    @abstractmethod
    def gain(self) -> float:
        """SFNC name: 'Gain' (in dB or camera-specific units)"""
        pass

    @gain.setter
    @abstractmethod
    def gain(self, value: float):
        pass

    @property
    @abstractmethod
    def gamma(self) -> float:
        """SFNC name: 'Gamma'"""
        pass

    @gamma.setter
    @abstractmethod
    def gamma(self, value: float):
        pass


class Camera(ABC):
    """Abstract base class for a GenICam-like camera interface."""

    def __init__(self):
        # Each vendor plugin would instantiate the sub-classes
        # for whichever SFNC categories it supports:
        self.acquisition: AcquisitionControl
        self.image_format: ImageFormatControl
        self.analog: AnalogControl
        # self.digital_io: DigitalIOControl
        # self.transport: TransportLayerControl
        # self.user_sets: UserSetControl
        # etc.

    @abstractmethod
    def initialize(self):
        """Perform device initialization, e.g., open camera connection."""
        pass

    @abstractmethod
    def close(self):
        """Close camera connection, free resources."""
        pass
