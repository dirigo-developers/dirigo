from enum import StrEnum
from abc import abstractmethod
from typing import ClassVar, Protocol
from typing import Protocol, runtime_checkable

from pydantic import ConfigDict, Field

from dirigo import units
from dirigo.hw_interfaces.hw_interface import Device, DeviceConfig, DeviceSettings
from dirigo.sw_interfaces.acquisition import AcquisitionProduct



class DropPolicy(StrEnum):
    BLOCK       = "block"          # backpressure
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    ERROR       = "error"


class DeliveryMode(StrEnum):
    COPY      = "copy"
    ZERO_COPY = "zero_copy"  # attach/view into transport-owned buffer when possible


class ImageTransportConfig(DeviceConfig):
    """
    Base config for host-side image transports.

    Concrete transports should extend this with whatever identifiers they need
    (e.g. NI-IMAQ device name, network interface/IP for GigE, USB serial, etc.).
    """
    pass


class ImageTransportSettings(DeviceSettings):
    nbuffers: int = Field(
        default     = 8,
        ge          = 1,
        description = "Number of host-side buffers in the streaming pool/ring."
    )

    frames_per_buffer: int = Field(
        default     = 1,
        ge          = 1,
        description = "How many frames constitute one Dirigo AcquisitionProduct."
    )

    get_timeout: units.Time = Field(
        default     = units.Time("2 s"),
        description = "Timeout for waiting on the next completed Dirigo buffer."
    )

    drop_policy: DropPolicy = Field(
        default     = DropPolicy.BLOCK,
        description = "Behavior when acquisition outpaces consumption."
    )

    delivery_mode: DeliveryMode = Field(
        default     = DeliveryMode.COPY,
        description = "Whether to copy into product or attach/view transport memory when supported."
    )



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
    settings_model: ClassVar[type[DeviceSettings]] = ImageTransportSettings

    # --- Transport settings ---

    @property
    @abstractmethod
    def nbuffers(self):
        ...

    @nbuffers.setter
    @abstractmethod
    def nbuffers(self, n: int):
        ...

    # --- Read-only transport properties ---
    
    @property
    @abstractmethod
    def bytes_per_pixel(self) -> int:
        """
        Bytes per pixel delivered by the transport.

        Derived from upstream pixel format / packing.
        """
        ...

    @property
    @abstractmethod
    def input_width(self) -> int:
        """Width (px) of the incoming image stream."""
        ...

    @property
    @abstractmethod
    def input_height(self) -> int:
        """Height (px) of the incoming image stream."""
        ...

    @property
    def output_width(self) -> int:
        """Width (px) of frames delivered into Dirigo buffers."""
        return self.input_width

    @property
    def output_height(self) -> int:
        """Height (px) of frames delivered into Dirigo buffers."""
        return self.input_height
    
    # --- Convenience ---
    @property
    def bytes_per_frame(self) -> int:
        """Total bytes in the delivered frame."""
        return self.output_width * self.output_height * self.bytes_per_pixel

    # ---- Buffer lifecycle ----

    @abstractmethod
    def prepare_buffers(self) -> None:
        """
        Allocate/register host-side buffers for acquisition.

        Must be called after connect() and before start_stream().
        """
        # use self.nbuffers (from settings)
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
        """Number of Dirigo buffers acquired so far (not necessarily raw frames)."""
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
