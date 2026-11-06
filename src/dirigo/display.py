from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from dirigo.frames import FrameArray
from dirigo.pipeline.product import Metadata
from dirigo.units import IntRange

if TYPE_CHECKING:
    from dirigo.pipeline.worker import Worker
    from dirigo.processors.display import DisplayProcessor
    from dirigo.rendering import FrameRenderer


class DisplayPixelFormat(Enum):
    """Byte layout of one rendered pixel."""

    RGB24 = "rgb24"
    BGRX32 = "bgrx32"

    @property
    def bytes_per_pixel(self) -> int:
        return 3 if self is DisplayPixelFormat.RGB24 else 4


@dataclass(frozen=True, slots=True)
class ColorVector:
    """RGB contributions per unit of channel intensity."""

    rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, tuple) or len(self.rgb) != 3:
            raise TypeError("rgb must be a tuple of three numbers.")

        for component in self.rgb:
            if (
                not isinstance(component, (int, float))
                or isinstance(component, bool)
            ):
                raise TypeError("Color vector components must be numbers.")

            if not isfinite(component) or component < 0:
                raise ValueError(
                    "Color vector components must be finite and nonnegative."
                )


class TransferFunction(ABC):
    """Map accumulated RGB contributions to normalized display values.

    Implementations must be immutable and safe for concurrent evaluation.
    Evaluation must not modify the input array.

    The renderer samples this function to construct its final lookup table.
    Each RGB component is mapped independently using the same function.
    """

    @property
    @abstractmethod
    def input_max(self) -> float:
        """Finite, positive input value at the upper end of the LUT.

        Accumulated contributions above this value saturate at this endpoint.
        Linear and gamma mappings use 1.0. Other mappings may use a larger
        domain, including future absorption-based rendering.
        """

    @abstractmethod
    def __call__(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate values in [0, input_max].

        Return an array with the same shape and finite values in [0, 1].
        The mapping need not be increasing: an exponential absorption
        mapping can produce white at zero input.
        """


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelSettings:
    """Rendering settings for one scientific input channel.

    Values at or below limits.min normalize to zero.
    Values at or above limits.max normalize to one.
    Intermediate values are normalized linearly.

    An enabled channel contributes its normalized intensity multiplied
    by its color vector. A disabled channel contributes zero.
    """

    limits: IntRange
    color: ColorVector
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.limits, IntRange):
            raise TypeError("limits must be an IntRange.")

        if not isinstance(self.color, ColorVector):
            raise TypeError("color must be a ColorVector.")

        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")


@dataclass(frozen=True, slots=True, kw_only=True)
class DisplaySettings:
    """One complete rendering configuration.

    Channel order matches the input frame's final axis. Every input
    channel has one settings record, including disabled channels.

    A frame is rendered using one complete settings snapshot.
    """

    channels: tuple[ChannelSettings, ...]
    transfer_function: TransferFunction

    def __post_init__(self) -> None:
        if not isinstance(self.channels, tuple):
            raise TypeError("channels must be a tuple.")

        if not self.channels:
            raise ValueError("At least one channel is required.")

        if any(
            not isinstance(channel, ChannelSettings)
            for channel in self.channels
        ):
            raise TypeError("Every channel must be ChannelSettings.")

        if not isinstance(self.transfer_function, TransferFunction):
            raise TypeError("transfer_function must be a TransferFunction.")

        maximum = self.transfer_function.input_max
        if not isfinite(maximum) or maximum <= 0:
            raise ValueError(
                "Transfer function input_max must be finite and positive."
            )


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class DisplayImage:
    """A rendered snapshot available independently of pipeline execution.

    pixels is a contiguous uint8 array with shape:
      RGB24:  (Y, X, 3), byte order R, G, B
      BGRX32: (Y, X, 4), byte order B, G, R, X; X is set to 255

    The producer must supply read-only storage and must not subsequently
    modify or recycle it while this snapshot remains referenced.

    metadata identifies the scientific frame. Rerendering that frame with
    different settings preserves its metadata.

    Freezing this envelope does not itself freeze the NumPy array.
    """

    pixels: npt.NDArray[np.uint8]
    pixel_format: DisplayPixelFormat
    metadata: Metadata


@dataclass(frozen=True, slots=True)
class Linear(TransferFunction):
    """Map normalized intensities directly to normalized display values."""

    @property
    def input_max(self) -> float:
        return 1.0

    def __call__(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return values.copy()


@dataclass(frozen=True, slots=True)
class Gamma(TransferFunction):
    """Apply a power-law mapping to normalized intensities.

    Exponents below one brighten intermediate values.
    Exponents above one darken intermediate values.
    An exponent of one gives the linear mapping.
    """

    gamma: float = 0.45

    def __post_init__(self) -> None:
        if not isinstance(self.gamma, (int, float)) or isinstance(self.gamma, bool):
            raise TypeError("gamma must be a number.")

        if not isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive.")

    @property
    def input_max(self) -> float:
        return 1.0

    def __call__(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return np.power(values, self.gamma)


class Display:
    """Retain scientific data and independent, read-only rendered snapshots.

    Frame submissions and synchronous settings redraws are serialized. Failed
    operations preserve the previous valid state. Callers must keep scientific
    input stable until submit() returns; subsequent changes cannot affect redraw.

    An attached processor has exclusive frame submission access. Settings remain
    adjustable during acquisition, and images remain available after it finishes.
    """

    def __init__(
        self,
        *,
        settings: DisplaySettings,
        pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
    ) -> None:
        if not isinstance(settings, DisplaySettings):
            raise TypeError("settings must be DisplaySettings.")
        if not isinstance(pixel_format, DisplayPixelFormat):
            raise TypeError("pixel_format must be a DisplayPixelFormat.")

        self._lock = Lock()
        self._settings = settings
        self._pixel_format = pixel_format
        self._frame: FrameArray | None = None
        self._metadata = Metadata()
        self._renderer: FrameRenderer | None = None
        self._latest_image: DisplayImage | None = None
        self._active_processor: object | None = None

    @property
    def pixel_format(self) -> DisplayPixelFormat:
        """Return the output format selected at construction."""
        return self._pixel_format

    @property
    def settings(self) -> DisplaySettings:
        """Return the current immutable settings snapshot."""
        with self._lock:
            return self._settings

    @property
    def latest_image(self) -> DisplayImage | None:
        """Return the latest image, or None before the first successful submission."""
        with self._lock:
            return self._latest_image

    def create_processor(
        self,
        *,
        upstream: Worker[Any, FrameArray],
        name: str | None = None,
    ) -> DisplayProcessor:
        """Create a single-use worker; display attachment begins at preparation."""
        from dirigo.processors.display import DisplayProcessor

        return DisplayProcessor(upstream=upstream, display=self, name=name)

    def submit(
        self,
        frame: FrameArray,
        *,
        metadata: Metadata | None = None,
    ) -> DisplayImage:
        """Copy and render a frame while no processor is attached.

        Return this submission's snapshot, which remains valid after redraws
        or later submissions. Omitted metadata means unspecified identifiers.
        """
        return self._submit(frame, metadata=metadata, processor=None)

    def _submit(
        self,
        frame: FrameArray,
        *,
        metadata: Metadata | None,
        processor: object | None,
    ) -> DisplayImage:
        """Render for the current frame owner and return the committed snapshot."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("Display frames must be NumPy arrays.")
        if frame.ndim != 3 or frame.size == 0:
            raise ValueError("Display frames must have nonempty shape (Y, X, C).")
        if metadata is None:
            metadata = Metadata()
        elif not isinstance(metadata, Metadata):
            raise TypeError("metadata must be Metadata or None.")

        with self._lock:
            if self._active_processor is not processor:
                raise RuntimeError(
                    "Display accepts frames only from its active processor."
                )

            retained = frame.copy(order="C")
            retained.flags.writeable = False

            renderer = self._renderer
            if self._frame is not None and retained.dtype != self._frame.dtype:
                renderer = None

            renderer, image = self._render_frame(
                retained,
                metadata,
                self._settings,
                renderer,
            )

            self._frame = retained
            self._metadata = metadata
            self._renderer = renderer
            self._latest_image = image
            return image

    def set_settings(self, settings: DisplaySettings) -> None:
        """Apply settings and synchronously redraw any retained scientific frame."""
        if not isinstance(settings, DisplaySettings):
            raise TypeError("settings must be DisplaySettings.")

        with self._lock:
            frame = self._frame
            if frame is None:
                self._settings = settings
                return

            renderer, image = self._render_frame(frame, self._metadata, settings)

            self._settings = settings
            self._renderer = renderer
            self._latest_image = image

    def _attach_processor(self, processor: object) -> None:
        """Reserve frame submission for one prepared processor."""
        with self._lock:
            if self._active_processor is not None:
                raise RuntimeError("Display already has an active processor.")
            self._active_processor = processor

    def _detach_processor(self, processor: object) -> None:
        """Release only this processor's reservation, including during rollback."""
        with self._lock:
            if self._active_processor is processor:
                self._active_processor = None

    def _render_frame(
        self,
        frame: FrameArray,
        metadata: Metadata,
        settings: DisplaySettings,
        renderer: FrameRenderer | None = None,
    ) -> tuple[FrameRenderer, DisplayImage]:
        """Build a complete image without changing the display's current state."""
        from dirigo.rendering import FrameRenderer

        if renderer is None:
            renderer = FrameRenderer(
                settings=settings,
                dtype=frame.dtype,
                pixel_format=self._pixel_format,
            )

        pixels = np.empty(
            (*frame.shape[:2], self._pixel_format.bytes_per_pixel),
            dtype=np.uint8,
        )
        renderer.render_into(frame, pixels)
        pixels.flags.writeable = False

        image = DisplayImage(
            pixels=pixels,
            pixel_format=self._pixel_format,
            metadata=metadata,
        )
        return renderer, image
