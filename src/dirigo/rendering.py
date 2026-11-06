import numpy as np
import numpy.typing as npt
from numba import njit, prange

from dirigo.display import DisplayPixelFormat, DisplaySettings
from dirigo.frames import FrameArray

_TRANSFER_LEVELS = 2**16


@njit(cache=True, nogil=True, parallel=True)
def _render_kernel(
    frame: FrameArray,
    channel_luts: npt.NDArray[np.uint16],
    transfer_lut: npt.NDArray[np.uint8],
    input_min: int,
    output: npt.NDArray[np.uint8],
) -> None:
    height, width, channels = frame.shape
    last = transfer_lut.size - 1
    bgrx = output.shape[2] == 4

    for y in prange(height):
        for x in range(width):
            red, green, blue = 0, 0, 0
            for channel in range(channels):
                index = int(frame[y, x, channel]) - input_min
                red += int(channel_luts[channel, index, 0])
                green += int(channel_luts[channel, index, 1])
                blue += int(channel_luts[channel, index, 2])

            r = transfer_lut[min(red, last)]
            g = transfer_lut[min(green, last)]
            b = transfer_lut[min(blue, last)]

            if bgrx:
                output[y, x, 0] = b
                output[y, x, 1] = g
                output[y, x, 2] = r
                output[y, x, 3] = 255
            else:
                output[y, x, 0] = r
                output[y, x, 1] = g
                output[y, x, 2] = b


class FrameRenderer:
    """Prepare LUTs for one settings snapshot, input dtype, and output format.

    Input frames have nonempty shape (Y, X, C), including C=1. Supported
    dtypes are uint8, uint16, and int16. Strided and read-only inputs are
    supported. Image dimensions may vary between calls.

    Output storage belongs to the caller and must be writable, contiguous,
    uint8, and disjoint from the input. No frame or output storage is retained.

    Each channel contribution is rounded to an integer index in a fixed
    65536-entry transfer domain. Contributions are accumulated using wider
    integers, saturated at index 65535, then mapped to uint8 through the
    transfer LUT. LUT construction uses NumPy's nearest rounding, with
    ties to even.

    Intermediate contributions and final display values are quantized,
    so results can differ from direct floating-point evaluation.
    Scientific input values are never changed.

    Prepared tables are read-only. Concurrent calls require separate,
    nonoverlapping output buffers. The first call may incur JIT compilation.
    """

    def __init__(
        self,
        *,
        settings: DisplaySettings,
        dtype: npt.DTypeLike,
        pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
    ) -> None:
        if not isinstance(settings, DisplaySettings):
            raise TypeError("settings must be DisplaySettings.")
        if not isinstance(pixel_format, DisplayPixelFormat):
            raise TypeError("pixel_format must be a DisplayPixelFormat.")

        input_dtype = np.dtype(dtype)
        if input_dtype not in (
            np.dtype(np.uint8),
            np.dtype(np.uint16),
            np.dtype(np.int16),
        ):
            raise ValueError("dtype must be uint8, uint16, or int16.")

        self._dtype = input_dtype
        self._pixel_format = pixel_format
        self._channels = len(settings.channels)

        info = np.iinfo(input_dtype)
        self._input_min = int(info.min)
        maximum = settings.transfer_function.input_max
        domain = np.linspace(0.0, maximum, _TRANSFER_LEVELS, dtype=np.float64)
        mapped = np.asarray(settings.transfer_function(domain), dtype=np.float64)

        if mapped.shape != domain.shape:
            raise ValueError("Transfer function must preserve the input shape.")
        if not np.isfinite(mapped).all() or ((mapped < 0) | (mapped > 1)).any():
            raise ValueError("Transfer function must return finite values in [0, 1].")

        self._transfer_lut = np.rint(255.0 * mapped).astype(np.uint8)
        values = np.arange(info.min, info.max + 1, dtype=np.float64)
        self._channel_luts = np.zeros(
            (self._channels, values.size, 3),
            dtype=np.uint16,
        )

        for index, channel in enumerate(settings.channels):
            if not channel.enabled:
                continue

            normalized = (values - channel.limits.min) / (
                channel.limits.max - channel.limits.min
            )
            np.clip(normalized, 0.0, 1.0, out=normalized)

            contributions = normalized[:, None] * np.asarray(
                channel.color.rgb,
                dtype=np.float64,
            )
            np.clip(contributions, 0.0, maximum, out=contributions)
            contributions /= maximum
            contributions *= _TRANSFER_LEVELS - 1

            self._channel_luts[index] = np.rint(contributions).astype(np.uint16)

        self._channel_luts.flags.writeable = False
        self._transfer_lut.flags.writeable = False

    def render_into(
        self,
        frame: FrameArray,
        output: npt.NDArray[np.uint8],
    ) -> None:
        """Fill output with rendered pixels without modifying the input frame."""
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a NumPy array.")
        if frame.ndim != 3 or frame.size == 0:
            raise ValueError("Input frame must have nonempty shape (Y, X, C).")
        if frame.dtype != self._dtype:
            raise ValueError("Input dtype must match the prepared dtype.")
        if frame.shape[2] != self._channels:
            raise ValueError("Input channel count must match the settings.")

        if not isinstance(output, np.ndarray):
            raise TypeError("Output must be a NumPy array.")

        expected_shape = (
            frame.shape[0],
            frame.shape[1],
            self._pixel_format.bytes_per_pixel,
        )
        if output.shape != expected_shape:
            raise ValueError("Output shape does not match the frame and pixel format.")
        if output.dtype != np.dtype(np.uint8):
            raise ValueError("Output dtype must be uint8.")
        if not output.flags.c_contiguous or not output.flags.writeable:
            raise ValueError("Output must be writable and C-contiguous.")
        if np.shares_memory(frame, output):
            raise ValueError("Input and output must not overlap.")

        _render_kernel(
            frame,
            self._channel_luts,
            self._transfer_lut,
            self._input_min,
            output,
        )
