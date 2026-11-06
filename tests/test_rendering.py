"""Test LUT-based scientific image rendering and output buffer validation."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import dirigo.rendering as rendering
from dirigo.display import (
    ChannelSettings,
    ColorVector,
    DisplayPixelFormat,
    DisplaySettings,
    Gamma,
    Linear,
    TransferFunction,
)
from dirigo.rendering import FrameRenderer
from dirigo.units import IntRange


@dataclass(frozen=True, slots=True)
class InvertedTransfer(TransferFunction):
    """Exercise decreasing mappings over an input domain extending above one."""

    @property
    def input_max(self) -> float:
        return 2.0

    def __call__(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return 1.0 - values / 2.0


@pytest.fixture(params=("compiled", "python"), autouse=True)
def kernel_backend(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run each rendering test against compiled and original Python execution."""
    if request.param == "python":
        monkeypatch.setattr(
            rendering,
            "_render_kernel",
            rendering._render_kernel.py_func,
        )


@pytest.fixture
def settings() -> DisplaySettings:
    return DisplaySettings(
        channels=(
            ChannelSettings(
                limits=IntRange(0, 255),
                color=ColorVector((1.0, 1.0, 1.0)),
            ),
        ),
        transfer_function=Linear(),
    )


def render(
    frame: npt.NDArray[Any],
    settings: DisplaySettings,
    pixel_format: DisplayPixelFormat = DisplayPixelFormat.RGB24,
) -> npt.NDArray[np.uint8]:
    """Render into initialized storage so unwritten output bytes are detectable."""
    renderer = FrameRenderer(
        settings=settings,
        dtype=frame.dtype,
        pixel_format=pixel_format,
    )
    output = np.full(
        (*frame.shape[:2], pixel_format.bytes_per_pixel),
        23,
        dtype=np.uint8,
    )
    renderer.render_into(frame, output)
    return output


@pytest.mark.parametrize(
    ("dtype", "limits", "samples"),
    [
        (np.uint8, IntRange(0, 255), [0, 127, 128, 255]),
        (np.uint16, IntRange(0, 65535), [0, 32767, 32768, 65535]),
        (np.int16, IntRange(-32768, 32767), [-32768, -1, 0, 32767]),
    ],
)
def test_supported_dtypes_cover_the_full_input_range(
    dtype: npt.DTypeLike,
    limits: IntRange,
    samples: list[int],
) -> None:
    """Map unsigned and signed endpoints and central values to grayscale."""
    settings = DisplaySettings(
        channels=(
            ChannelSettings(
                limits=limits,
                color=ColorVector((1.0, 1.0, 1.0)),
            ),
        ),
        transfer_function=Linear(),
    )
    frame = np.array(samples, dtype=dtype).reshape(1, 4, 1)

    output = render(frame, settings)

    np.testing.assert_array_equal(
        output,
        [[[0, 0, 0], [127, 127, 127], [128, 128, 128], [255, 255, 255]]],
    )


def test_display_limits_normalize_and_clip_input_values() -> None:
    """Clip values outside the display limits and normalize interior values."""
    settings = DisplaySettings(
        channels=(
            ChannelSettings(
                limits=IntRange(100, 200),
                color=ColorVector((1.0, 1.0, 1.0)),
            ),
        ),
        transfer_function=Linear(),
    )
    frame = np.array([0, 100, 150, 200, 255], dtype=np.uint16).reshape(1, 5, 1)

    output = render(frame, settings)

    np.testing.assert_array_equal(
        output,
        [[[0, 0, 0], [0, 0, 0], [128, 128, 128], [255, 255, 255], [255, 255, 255]]],
    )


@pytest.mark.parametrize(
    ("pixel_format", "expected"),
    [
        (DisplayPixelFormat.RGB24, [255, 128, 64]),
        (DisplayPixelFormat.BGRX32, [64, 128, 255, 255]),
    ],
)
def test_color_vectors_and_output_byte_order(
    pixel_format: DisplayPixelFormat,
    expected: list[int],
) -> None:
    """Map scientific channels through their vectors and fill every output byte."""
    settings = DisplaySettings(
        channels=tuple(
            ChannelSettings(
                limits=IntRange(0, 255),
                color=ColorVector(rgb),
            )
            for rgb in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        transfer_function=Linear(),
    )
    frame = np.array([[[255, 128, 64]]], dtype=np.uint16)

    output = render(frame, settings, pixel_format)

    np.testing.assert_array_equal(output, [[expected]])


def test_channel_accumulation_saturates_without_integer_wraparound() -> None:
    """Accumulate beyond uint16 capacity before saturating the displayed value."""
    channel = ChannelSettings(
        limits=IntRange(0, 2),
        color=ColorVector((1.0, 0.0, 0.0)),
    )
    settings = DisplaySettings(
        channels=(channel,) * 20,
        transfer_function=Linear(),
    )
    frame = np.ones((1, 1, 20), dtype=np.uint16)

    output = render(frame, settings)

    np.testing.assert_array_equal(output, [[[255, 0, 0]]])


def test_disabled_channel_contributes_zero() -> None:
    """Exclude a disabled channel even when its scientific intensity is maximal."""
    settings = DisplaySettings(
        channels=(
            ChannelSettings(
                limits=IntRange(0, 255),
                color=ColorVector((1.0, 0.0, 0.0)),
                enabled=False,
            ),
            ChannelSettings(
                limits=IntRange(0, 255),
                color=ColorVector((0.0, 1.0, 0.0)),
            ),
        ),
        transfer_function=Linear(),
    )
    frame = np.full((1, 1, 2), 255, dtype=np.uint16)

    output = render(frame, settings)

    np.testing.assert_array_equal(output, [[[0, 255, 0]]])


def test_gamma_is_applied_after_channel_mixing() -> None:
    """Apply gamma to the summed signal rather than to each channel separately."""
    channel = ChannelSettings(
        limits=IntRange(0, 4),
        color=ColorVector((1.0, 0.0, 0.0)),
    )
    settings = DisplaySettings(
        channels=(channel, channel),
        transfer_function=Gamma(gamma=0.5),
    )
    frame = np.ones((1, 1, 2), dtype=np.uint16)

    output = render(frame, settings)

    np.testing.assert_array_equal(output, [[[180, 0, 0]]])


@pytest.mark.parametrize(
    ("enabled", "expected"),
    [
        (True, [0, 255, 255]),
        (False, [255, 255, 255]),
    ],
)
def test_transfer_function_controls_domain_and_zero_signal_color(
    enabled: bool,
    expected: list[int],
) -> None:
    """Support accumulation >1.0 and map zero contributions through the transfer."""
    channel = ChannelSettings(
        limits=IntRange(0, 255),
        color=ColorVector((1.0, 0.0, 0.0)),
        enabled=enabled,
    )
    settings = DisplaySettings(
        channels=(channel, channel),
        transfer_function=InvertedTransfer(),
    )
    frame = np.full((1, 1, 2), 255, dtype=np.uint16)

    output = render(frame, settings)

    np.testing.assert_array_equal(output, [[expected]])


def test_strided_readonly_input_is_not_modified(
    settings: DisplaySettings,
) -> None:
    """Render a strided read-only view without altering its scientific storage."""
    storage = np.array(
        [[[0], [99], [64], [99]], [[128], [99], [255], [99]]],
        dtype=np.uint16,
    )
    original = storage.copy()
    frame = storage[:, ::2, :]
    frame.flags.writeable = False

    output = render(frame, settings)

    np.testing.assert_array_equal(
        output,
        [[[0, 0, 0], [64, 64, 64]], [[128, 128, 128], [255, 255, 255]]],
    )
    np.testing.assert_array_equal(storage, original)


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.int32])
def test_renderer_rejects_unsupported_dtypes(
    settings: DisplaySettings,
    dtype: npt.DTypeLike,
) -> None:
    """Reject input dtypes without a supported integer lookup table."""
    with pytest.raises(ValueError, match="dtype must be"):
        FrameRenderer(settings=settings, dtype=dtype)


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (None, TypeError),
        ([], TypeError),
        (np.zeros((1, 2), dtype=np.uint16), ValueError),
        (np.zeros((1, 1, 2, 1), dtype=np.uint16), ValueError),
        (np.zeros((0, 2, 1), dtype=np.uint16), ValueError),
        (np.zeros((1, 2, 2), dtype=np.uint16), ValueError),
        (np.zeros((1, 2, 1), dtype=np.uint8), ValueError),
        (np.zeros((1, 2, 1), dtype=np.float32), ValueError),
    ],
)
def test_invalid_input_is_rejected_before_writing_output(
    settings: DisplaySettings,
    frame,  # ruff: ignore[missing-type-function-argument]
    error: type[Exception],
) -> None:
    """Reject invalid input layout, dtype, or channel count without writing pixels."""
    renderer = FrameRenderer(settings=settings, dtype=np.uint16)
    output = np.full((1, 2, 3), 23, dtype=np.uint8)

    with pytest.raises(error):
        renderer.render_into(frame, output)

    np.testing.assert_array_equal(output, np.full((1, 2, 3), 23, dtype=np.uint8))


@pytest.mark.parametrize(
    ("problem", "error"),
    [
        ("not_array", TypeError),
        ("shape", ValueError),
        ("dtype", ValueError),
        ("readonly", ValueError),
        ("strided", ValueError),
    ],
)
def test_invalid_output_storage_is_rejected(
    settings: DisplaySettings,
    problem: str,
    error: type[Exception],
) -> None:
    """Require writable contiguous uint8 output with the expected pixel layout."""
    renderer = FrameRenderer(settings=settings, dtype=np.uint16)
    frame = np.zeros((1, 2, 1), dtype=np.uint16)
    output: Any = np.full((1, 2, 3), 23, dtype=np.uint8)

    if problem == "not_array":
        output = None
    elif problem == "shape":
        output = np.full((1, 2, 4), 23, dtype=np.uint8)
    elif problem == "dtype":
        output = np.full((1, 2, 3), 23, dtype=np.uint16)
    elif problem == "readonly":
        output.flags.writeable = False
    elif problem == "strided":
        output = np.full((1, 4, 3), 23, dtype=np.uint8)[:, ::2, :]

    with pytest.raises(error):
        renderer.render_into(frame, output)  # ty: ignore[invalid-argument-type]

    if isinstance(output, np.ndarray):
        assert (output == 23).all()


def test_overlapping_input_and_output_are_rejected(
    settings: DisplaySettings,
) -> None:
    """Prevent rendering into storage shared with the scientific input."""
    settings = DisplaySettings(
        channels=settings.channels * 3,
        transfer_function=Linear(),
    )
    renderer = FrameRenderer(settings=settings, dtype=np.uint8)
    frame = np.full((1, 2, 3), 23, dtype=np.uint8)
    original = frame.copy()

    with pytest.raises(ValueError, match="must not overlap"):
        renderer.render_into(frame, frame)

    np.testing.assert_array_equal(frame, original)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_transfer_values_are_rejected(
    settings: DisplaySettings,
    value: float,
) -> None:
    """Reject invalid transfer outputs before converting them to uint8."""

    class BrokenTransfer(Linear):
        def __call__(
            self,
            values: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return np.full_like(values, value)

    settings = DisplaySettings(
        channels=settings.channels,
        transfer_function=BrokenTransfer(),
    )

    with pytest.raises(ValueError, match="finite values"):
        FrameRenderer(settings=settings, dtype=np.uint16)


def test_transfer_function_must_preserve_sample_shape(
    settings: DisplaySettings,
) -> None:
    """Reject transfer functions that return the wrong number of LUT samples."""

    class BrokenTransfer(Linear):
        def __call__(
            self,
            values: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return values[:-1]

    settings = DisplaySettings(
        channels=settings.channels,
        transfer_function=BrokenTransfer(),
    )

    with pytest.raises(ValueError, match="preserve the input shape"):
        FrameRenderer(settings=settings, dtype=np.uint16)


@pytest.mark.parametrize("settings", [None, object(), {}])
def test_renderer_requires_display_settings(settings) -> None:  # ruff: ignore[missing-type-function-argument]
    """Reject settings that are not a DisplaySettings instance."""
    with pytest.raises(TypeError, match="settings must be DisplaySettings"):
        FrameRenderer(settings=settings, dtype=np.uint16)


@pytest.mark.parametrize("pixel_format", [None, "RGB24", 3])
def test_renderer_requires_a_display_pixel_format(
    settings: DisplaySettings,
    pixel_format,  # ruff: ignore[missing-type-function-argument]
) -> None:
    """Reject pixel formats that are not DisplayPixelFormat members."""
    with pytest.raises(TypeError, match="pixel_format must be a DisplayPixelFormat"):
        FrameRenderer(
            settings=settings,
            dtype=np.uint16,
            pixel_format=pixel_format,
        )
