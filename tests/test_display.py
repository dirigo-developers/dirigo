"""Test display configuration validation and immutable settings snapshots.

Rendering, buffer ownership, and paused redraw behavior are tested separately
when their implementations are introduced.
"""

from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from dirigo.display import (
    ChannelSettings,
    ColorVector,
    DisplayImage,
    DisplayPixelFormat,
    DisplaySettings,
    TransferFunction,
)
from dirigo.frames import FrameMetadata
from dirigo.units import IntRange


@dataclass(frozen=True, slots=True)
class StubTransferFunction(TransferFunction):
    """Provide an immutable transfer function with a configurable input domain."""

    maximum: float = 1.0

    @property
    def input_max(self) -> float:
        return self.maximum

    def __call__(
        self,
        values: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return values / self.maximum


@pytest.fixture
def channel() -> ChannelSettings:
    return ChannelSettings(
        limits=IntRange(0, 4095),
        color=ColorVector((0.0, 1.0, 1.0)),
    )


@pytest.mark.parametrize(
    ("pixel_format", "expected_bytes"),
    [
        (DisplayPixelFormat.RGB24, 3),
        (DisplayPixelFormat.BGRX32, 4),
    ],
)
def test_pixel_format_byte_count(
    pixel_format: DisplayPixelFormat,
    expected_bytes: int,
) -> None:
    """Report three bytes for RGB24 and four bytes for BGRX32."""
    assert pixel_format.bytes_per_pixel == expected_bytes


@pytest.mark.parametrize(
    "rgb",
    [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.25, 0.5, 0.75),
        (2.0, 0.5, 3.0),
    ],
)
def test_color_vector_accepts_nonnegative_coefficients(
    rgb: tuple[float, float, float],
) -> None:
    """Preserve finite nonnegative coefficients, including values above one."""
    assert ColorVector(rgb).rgb == rgb


@pytest.mark.parametrize(
    "rgb",
    [
        None,
        [1.0, 0.0, 0.0],
        (),
        (1.0, 0.0),
        (1.0, 0.0, 0.0, 1.0),
    ],
)
def test_color_vector_requires_a_three_element_tuple(rgb) -> None:  # ruff: ignore[missing-type-function-argument]
    """Reject non-tuples and tuples without exactly three components."""
    with pytest.raises(TypeError, match="tuple of three"):
        ColorVector(rgb)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize(
    ("component", "error"),
    [
        (True, TypeError),
        ("1.0", TypeError),
        (None, TypeError),
        (-0.1, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (float("-inf"), ValueError),
    ],
)
def test_color_vector_rejects_invalid_components(
    index: int,
    component,  # ruff: ignore[missing-type-function-argument]
    error: type[Exception],
) -> None:
    """Reject invalid types, negative values, and nonfinite values in any component."""
    rgb: list[Any] = [0.0, 0.0, 0.0]
    rgb[index] = component

    with pytest.raises(error, match="components must be"):
        ColorVector(tuple(rgb))


def test_channel_accepts_signed_limits_and_can_be_disabled() -> None:
    """Preserve signed display limits and an explicitly disabled channel."""
    channel = ChannelSettings(
        limits=IntRange(-2048, 2047),
        color=ColorVector((1.0, 0.0, 0.0)),
        enabled=False,
    )

    assert channel.limits == IntRange(-2048, 2047)
    assert channel.enabled is False


def test_channel_is_enabled_by_default(channel: ChannelSettings) -> None:
    """Enable a channel when no explicit enabled state is supplied."""
    assert channel.enabled is True


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"limits": (0, 255)}, "limits must be an IntRange"),
        ({"limits": None}, "limits must be an IntRange"),
        ({"color": (1.0, 0.0, 0.0)}, "color must be a ColorVector"),
        ({"color": None}, "color must be a ColorVector"),
        ({"enabled": 0}, "enabled must be a boolean"),
        ({"enabled": 1}, "enabled must be a boolean"),
        ({"enabled": "true"}, "enabled must be a boolean"),
        ({"enabled": None}, "enabled must be a boolean"),
    ],
)
def test_channel_rejects_invalid_fields(
    changes: dict[str, Any],
    message: str,
) -> None:
    """Reject incorrectly typed channel settings without coercing their values."""
    arguments: dict[str, Any] = {
        "limits": IntRange(0, 255),
        "color": ColorVector((1.0, 1.0, 1.0)),
    }
    arguments.update(changes)

    with pytest.raises(TypeError, match=message):
        ChannelSettings(**arguments)


def test_display_settings_preserve_channel_order(
    channel: ChannelSettings,
) -> None:
    """Preserve channel order, disabled channels, and the supplied transfer function."""
    second = ChannelSettings(
        limits=IntRange(100, 2000),
        color=ColorVector((1.0, 0.0, 0.0)),
        enabled=False,
    )
    transfer = StubTransferFunction()

    settings = DisplaySettings(
        channels=(channel, second),
        transfer_function=transfer,
    )

    assert settings.channels == (channel, second)
    assert settings.transfer_function is transfer


@pytest.mark.parametrize(
    ("channels", "error", "message"),
    [
        (None, TypeError, "channels must be a tuple"),
        ([], TypeError, "channels must be a tuple"),
        ((), ValueError, "At least one channel"),
        ((object(),), TypeError, "Every channel must be ChannelSettings"),
    ],
)
def test_display_settings_reject_invalid_channels(
    channels,  # ruff: ignore[missing-type-function-argument]
    error: type[Exception],
    message: str,
) -> None:
    """Require a nonempty tuple containing only ChannelSettings records."""
    with pytest.raises(error, match=message):
        DisplaySettings(
            channels=channels,
            transfer_function=StubTransferFunction(),
        )


@pytest.mark.parametrize("transfer", [None, object(), "gamma"])
def test_display_settings_require_a_transfer_function(
    channel: ChannelSettings,
    transfer,  # ruff: ignore[missing-type-function-argument]
) -> None:
    """Require a TransferFunction instance rather than a name or unrelated object."""
    with pytest.raises(TypeError, match="must be a TransferFunction"):
        DisplaySettings(
            channels=(channel,),
            transfer_function=transfer,
        )


@pytest.mark.parametrize(
    "maximum",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_display_settings_reject_invalid_transfer_domain(
    channel: ChannelSettings,
    maximum: float,
) -> None:
    """Reject transfer domains whose upper endpoint is nonpositive or nonfinite."""
    with pytest.raises(ValueError, match="finite and positive"):
        DisplaySettings(
            channels=(channel,),
            transfer_function=StubTransferFunction(maximum=maximum),
        )


def test_display_settings_allow_a_transfer_domain_above_one(
    channel: ChannelSettings,
) -> None:
    """Allow transfer mappings to operate over accumulated values above one."""
    settings = DisplaySettings(
        channels=(channel,),
        transfer_function=StubTransferFunction(maximum=8.0),
    )

    assert settings.transfer_function.input_max == 8.0


def test_settings_snapshot_is_immutable(channel: ChannelSettings) -> None:
    """Prevent reassignment of settings and their nested color and limit records."""
    settings = DisplaySettings(
        channels=(channel,),
        transfer_function=StubTransferFunction(),
    )

    with pytest.raises(FrozenInstanceError):
        settings.channels = ()  # ty: ignore[invalid-assignment]

    with pytest.raises(FrozenInstanceError):
        channel.enabled = False  # ty: ignore[invalid-assignment]

    with pytest.raises(FrozenInstanceError):
        channel.color.rgb = 1.0, 0.0, 0.0  # ty: ignore[invalid-assignment]

    with pytest.raises(FrozenInstanceError):
        channel.limits.min = 100  # ty: ignore[invalid-assignment]


def test_display_image_preserves_metadata_and_freezes_its_envelope() -> None:
    """Preserve the source metadata record and prevent replacing its reference."""
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels.flags.writeable = False
    metadata = FrameMetadata(
        sequence_index=17,
        time_index=4,
        z_index=2,
    )

    image = DisplayImage(
        pixels=pixels,
        pixel_format=DisplayPixelFormat.RGB24,
        metadata=metadata,
    )

    assert image.metadata is metadata

    with pytest.raises(FrozenInstanceError):
        image.metadata = FrameMetadata()  # ty: ignore[invalid-assignment]
