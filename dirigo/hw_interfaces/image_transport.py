from abc import abstractmethod
from typing import ClassVar, Protocol
from typing import Protocol, runtime_checkable

from dirigo.hw_interfaces.hw_interface import DeviceConfig, Device
from dirigo.sw_interfaces.acquisition import AcquisitionProduct



class ImageTransportConfig(DeviceConfig):
    """
    Base config for host-side image transports.

    Concrete transports should extend this with whatever identifiers they need
    (e.g. NI-IMAQ device name, network interface/IP for GigE, USB serial, etc.).
    """
    pass


class ImageTransport(Device):
    """
    Host-side receiver/streamer that delivers image buffers into AcquisitionProducts.

    This can be:
      - a Camera Link framegrabber (PCIe card)
      - a GigE Vision receiver/streamer
      - a USB camera host-side streamer
      - any other transport that delivers image data to the PC

    Lifecycle:
      - connect(): reserve/open host-side resources (driver sessions, sockets, DMA, etc.)
      - prepare_buffers(): allocate/register buffers for streaming
      - start_stream()/stop_stream(): start/stop streaming on the host side
      - close(): release all resources
    """

    config_model: ClassVar[type[DeviceConfig]] = ImageTransportConfig

    # ---- Transport properties ----

    @property
    @abstractmethod
    def bytes_per_pixel(self) -> int:
        """Number of bytes per pixel delivered in the stream (e.g. 1, 2, 3, 4)."""
        ...

    @property
    @abstractmethod
    def acq_width(self) -> int:
        """Width of the acquisition window (upstream stream entering the transport)."""
        ...

    @property
    @abstractmethod
    def acq_height(self) -> int:
        """Height of the acquisition window (upstream stream entering the transport)."""
        ...
    
    @property
    def frame_width(self) -> int:
        """Width (px) of frames delivered into host buffers."""
        return self.acq_width

    @property
    def frame_height(self) -> int:
        """Height (px) of frames delivered into host buffers."""
        return self.acq_height
    
    @property
    def bytes_per_frame(self) -> int:
        return self.frame_width * self.frame_height * self.bytes_per_pixel

    # ---- Buffer lifecycle ----

    @abstractmethod
    def prepare_buffers(self, nbuffers: int) -> None:
        """
        Allocate/register host-side buffers for acquisition.

        Must be called after connect() and before start_stream().
        """
        ...

    @abstractmethod
    def start_stream(self) -> None:
        """Begin host-side streaming (DMA, socket receive, etc.)."""
        ...

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop host-side streaming."""
        ...

    @abstractmethod
    def get_next_completed_buffer(self, product: AcquisitionProduct) -> None:
        """
        Copy or attach the next completed buffer into `product`.

        Implementations should raise a transport-specific exception if no buffer is ready.
        (Dirigo can standardize this later; keep it simple for now.)
        """
        ...

    @property
    @abstractmethod
    def buffers_acquired(self) -> int:
        """Number of *Dirigo buffers* acquired so far (not necessarily raw frames)."""
        ...


@runtime_checkable
class SerialControl(Protocol):
    """
    Optional capability: serial control channel exposed by some transports
    (common for Camera Link cameras via framegrabber serial).
    """
    def serial_write(self, message: str) -> None: ...
    def serial_read(self, nbytes: int | None = None) -> str: ...


class FrameGrabberConfig(ImageTransportConfig):
    """
    Config for PCIe/host-card framegrabbers (Camera Link, CoaXPress, etc.).
    Concrete implementations can add details like device_name, camera_file, tap geometry, etc.
    """
    pass


class FrameGrabber(ImageTransport):
    """
    Specialized ImageTransport for add-in framegrabber cards.

    Adds optional capabilities commonly associated with framegrabbers:
      - host-side ROI (for some APIs)
      - line-scan style 'lines_per_buffer' semantics
    """

    config_model: ClassVar[type[DeviceConfig]] = FrameGrabberConfig

    # Frame width (delivered to host) set by ROI geometry
    @property
    def frame_width(self) -> int:
        return self.roi_width

    @property
    def frame_height(self) -> int:
        return self.roi_height

    # ---- ROI geometry ----
    @property
    @abstractmethod
    def roi_width(self) -> int:
        ...

    @roi_width.setter
    @abstractmethod
    def roi_width(self, width: int) -> None:
        ...

    @property
    @abstractmethod
    def roi_height(self) -> int:
        ...

    @roi_height.setter
    @abstractmethod
    def roi_height(self, height: int) -> None:
        ...

    @property
    @abstractmethod
    def roi_left(self) -> int:
        ...

    @roi_left.setter
    @abstractmethod
    def roi_left(self, left: int) -> None:
        ...

    @property
    @abstractmethod
    def roi_top(self) -> int:
        ...

    @roi_top.setter
    @abstractmethod
    def roi_top(self, top: int) -> None:
        ...



# class CameraConnectionType(Enum):
#     CAMERA_LINK = 0
#     GIGE        = 1
#     USB         = 2

# class FrameGrabber(Device):
#     attr_name = "frame_grabber"

#     def __init__(self):
#         self._buffers: List[Any] # TODO refine the buffer object typehint
#         self._camera: Optional['Camera'] = None

#     @abstractmethod
#     def serial_write(self, message):
#         pass

#     @abstractmethod
#     def serial_read(self, nbytes: Optional[int] = None) -> str:
#         pass

#     @property
#     @abstractmethod
#     def pixels_width(self) -> int:
#         """Total number of sensor pixels wide."""
#         pass

#     @property
#     @abstractmethod
#     def roi_height(self) -> int:
#         pass

#     @property
#     @abstractmethod
#     def roi_width(self) -> int:
#         pass

#     @roi_width.setter
#     @abstractmethod
#     def roi_width(self, width: int):
#         pass

#     @property
#     @abstractmethod
#     def roi_left(self) -> int:
#         pass

#     @roi_left.setter
#     @abstractmethod
#     def roi_left(self, left: int):
#         pass
    
#     @property
#     @abstractmethod
#     def lines_per_buffer(self) -> int:
#         pass

#     @lines_per_buffer.setter
#     @abstractmethod
#     def lines_per_buffer(self, lines: int):
#         pass

#     @property
#     @abstractmethod
#     def bytes_per_pixel(self) -> int:
#         pass

#     @property
#     def bytes_per_buffer(self):
#         if self.lines_per_buffer is None or self.roi_width is None:
#             raise RuntimeError("Lines per buffer or ROI width not initialized")
#         return self.lines_per_buffer * self.roi_width * self.bytes_per_pixel

#     @abstractmethod
#     def prepare_buffers(self, nbuffers: int):
#         pass

#     @abstractmethod
#     def start(self):
#         pass

#     @abstractmethod
#     def stop(self):
#         pass

#     @abstractmethod
#     def get_next_completed_buffer(self, product: AcquisitionProduct) -> None:
#         pass

#     @property
#     @abstractmethod
#     def buffers_acquired(self) -> int:
#         pass

#     @property
#     @abstractmethod
#     def data_range(self) -> units.IntRange:
#         """ Returns the range of values returned by the frame grabber. """
#         pass
