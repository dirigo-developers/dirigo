from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import numpy.typing as npt
import pytest

from dirigo.display import (
    ChannelSettings,
    ColorVector,
    Display,
    DisplayPixelFormat,
    DisplaySettings,
    Gamma,
    Linear,
)
from dirigo.frames import FrameArray, FrameMetadata
from dirigo.pipeline.product import Metadata
from dirigo.rendering import FrameRenderer
from dirigo.units import IntRange


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


@pytest.fixture
def red_settings(settings: DisplaySettings) -> DisplaySettings:
    return replace(
        settings,
        channels=(
            replace(
                settings.channels[0],
                color=ColorVector((1.0, 0.0, 0.0)),
            ),
        ),
    )


@pytest.fixture
def display(settings: DisplaySettings) -> Display:
    return Display(settings=settings)


@pytest.mark.parametrize("pixel_format", list(DisplayPixelFormat))
def test_display_initial_state(
    settings: DisplaySettings,
    pixel_format: DisplayPixelFormat,
) -> None:
    """Preserve construction settings and expose no image before submission."""
    display = Display(settings=settings, pixel_format=pixel_format)

    assert display.settings is settings
    assert display.pixel_format is pixel_format
    assert display.latest_image is None


@pytest.mark.parametrize("settings", [None, object(), "gamma"])
def test_display_rejects_invalid_initial_settings(settings: DisplaySettings) -> None:
    """Require a complete DisplaySettings record at construction."""
    with pytest.raises(TypeError, match="settings must be DisplaySettings"):
        Display(settings=settings)


@pytest.mark.parametrize("pixel_format", [None, "rgb24", 0])
def test_display_rejects_invalid_pixel_formats(
    settings: DisplaySettings,
    pixel_format: DisplayPixelFormat,
) -> None:
    """Require a DisplayPixelFormat rather than its name or an unrelated value."""
    with pytest.raises(TypeError, match="pixel_format must be"):
        Display(settings=settings, pixel_format=pixel_format)


def test_settings_can_change_before_the_first_frame(
    display: Display,
    red_settings: DisplaySettings,
) -> None:
    """Accept settings changes without creating an image when no frame is retained."""
    display.set_settings(red_settings)

    assert display.settings is red_settings
    assert display.latest_image is None


@pytest.mark.parametrize(
    ("pixel_format", "expected"),
    [
        (DisplayPixelFormat.RGB24, [128, 0, 0]),
        (DisplayPixelFormat.BGRX32, [0, 0, 128, 255]),
    ],
)
def test_submission_renders_the_selected_pixel_format(
    red_settings: DisplaySettings,
    pixel_format: DisplayPixelFormat,
    expected: list[int],
) -> None:
    """Publish a rendered snapshot with the selected layout and source metadata."""
    display = Display(settings=red_settings, pixel_format=pixel_format)
    metadata = FrameMetadata(sequence_index=17, time_index=4, z_index=2)
    frame = np.full((1, 1, 1), 128, dtype=np.uint16)

    display.submit(frame, metadata=metadata)

    image = display.latest_image
    assert image is not None
    assert image.pixel_format is pixel_format
    assert image.metadata is metadata
    np.testing.assert_array_equal(image.pixels, [[expected]])


def test_omitted_metadata_does_not_reuse_previous_identifiers(
    display: Display,
) -> None:
    """Start a submission with empty metadata when no record is supplied."""
    frame = np.zeros((1, 1, 1), dtype=np.uint16)
    display.submit(frame, metadata=FrameMetadata(sequence_index=17))

    display.submit(frame)

    image = display.latest_image
    assert image is not None
    assert image.metadata == Metadata()


def test_submission_retains_an_owned_scientific_copy(
    display: Display,
    red_settings: DisplaySettings,
) -> None:
    """Redraw from retained scientific values after the caller changes its array."""
    frame = np.full((1, 1, 1), 128, dtype=np.uint16)
    display.submit(frame)
    assert frame.flags.writeable

    frame.fill(0)
    display.set_settings(red_settings)

    image = display.latest_image
    assert image is not None
    np.testing.assert_array_equal(image.pixels, [[[128, 0, 0]]])
    np.testing.assert_array_equal(frame, [[[0]]])


def test_settings_changes_redraw_and_preserve_metadata(
    display: Display,
    red_settings: DisplaySettings,
) -> None:
    """Apply new color and transfer settings to the retained frame automatically."""
    metadata = FrameMetadata(sequence_index=17, time_index=4, z_index=2)
    display.submit(
        np.full((1, 1, 1), 128, dtype=np.uint16),
        metadata=metadata,
    )
    updated = replace(red_settings, transfer_function=Gamma(gamma=2.0))

    display.set_settings(updated)

    image = display.latest_image
    assert image is not None
    assert display.settings is updated
    assert image.metadata is metadata
    np.testing.assert_array_equal(image.pixels, [[[64, 0, 0]]])


def test_previous_image_snapshots_remain_valid(
    display: Display,
    red_settings: DisplaySettings,
) -> None:
    """Preserve earlier pixels across later submissions and settings redraws."""
    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    first = display.latest_image
    assert first is not None

    display.submit(np.full((1, 1, 1), 128, dtype=np.uint16))
    second = display.latest_image
    assert second is not None

    display.set_settings(red_settings)
    third = display.latest_image
    assert third is not None

    assert first is not second
    assert second is not third
    np.testing.assert_array_equal(first.pixels, [[[64, 64, 64]]])
    np.testing.assert_array_equal(second.pixels, [[[128, 128, 128]]])
    np.testing.assert_array_equal(third.pixels, [[[128, 0, 0]]])


def test_image_pixels_are_readonly(display: Display) -> None:
    """Prevent accidental pixel assignment through a returned image snapshot."""
    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    image = display.latest_image
    assert image is not None

    assert not image.pixels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        image.pixels[0, 0, 0] = 0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_display_accepts_new_dimensions_and_supported_dtypes(
    display: Display,
    dtype: npt.DTypeLike,
) -> None:
    """Allow a persistent display to receive a different acquisition layout."""
    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    frame = np.array([[[0], [255]]], dtype=dtype)

    display.submit(frame)

    image = display.latest_image
    assert image is not None
    np.testing.assert_array_equal(image.pixels, [[[0, 0, 0], [255, 255, 255]]])


@pytest.mark.parametrize(
    ("frame", "error"),
    [
        (None, TypeError),
        ([], TypeError),
        (np.zeros((1, 1), dtype=np.uint16), ValueError),
        (np.zeros((0, 1, 1), dtype=np.uint16), ValueError),
        (np.zeros((1, 1, 2), dtype=np.uint16), ValueError),
        (np.zeros((1, 1, 1), dtype=np.float32), ValueError),
    ],
)
def test_failed_submission_preserves_the_retained_frame(
    display: Display,
    settings: DisplaySettings,
    red_settings: DisplaySettings,
    frame,  # ruff: ignore[missing-type-function-argument]
    error: type[Exception],
) -> None:
    """Keep the previous image and scientific frame when a submission fails."""
    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    previous = display.latest_image

    with pytest.raises(error):
        display.submit(frame)

    assert display.latest_image is previous
    assert display.settings is settings

    display.set_settings(red_settings)
    image = display.latest_image
    assert image is not None
    np.testing.assert_array_equal(image.pixels, [[[64, 0, 0]]])


@pytest.mark.parametrize("metadata", [{}, object(), "metadata", True])
def test_submission_rejects_invalid_metadata(
    display: Display,
    metadata,  # ruff: ignore[missing-type-function-argument]
) -> None:
    """Reject unrelated metadata values without replacing the current image."""
    frame = np.full((1, 1, 1), 64, dtype=np.uint16)
    display.submit(frame)
    previous = display.latest_image

    with pytest.raises(TypeError, match="metadata must be"):
        display.submit(frame, metadata=metadata)

    assert display.latest_image is previous


@pytest.mark.parametrize("invalid", [None, {}, "gamma"])
def test_invalid_settings_preserve_the_current_state(
    display: Display,
    settings: DisplaySettings,
    invalid,  # ruff: ignore[missing-type-function-argument]
) -> None:
    """Reject incorrectly typed settings without altering the active configuration."""
    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    previous = display.latest_image

    with pytest.raises(TypeError, match="settings must be DisplaySettings"):
        display.set_settings(invalid)

    assert display.settings is settings
    assert display.latest_image is previous


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("channel_count", "channel count must match"),
        ("transfer", "must return finite values"),
    ],
)
def test_failed_redraw_preserves_settings_and_image(
    display: Display,
    settings: DisplaySettings,
    failure: str,
    message: str,
) -> None:
    """Commit neither settings nor pixels when prepping or rendering an update fails."""

    class BrokenTransfer(Linear):
        def __call__(
            self,
            values: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            return np.full_like(values, np.nan)

    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    previous = display.latest_image

    if failure == "channel_count":
        updated = replace(settings, channels=settings.channels * 2)
    else:
        updated = replace(settings, transfer_function=BrokenTransfer())

    with pytest.raises(ValueError, match=message):
        display.set_settings(updated)

    assert display.settings is settings
    assert display.latest_image is previous

    display.submit(np.full((1, 1, 1), 128, dtype=np.uint16))
    image = display.latest_image
    assert image is not None
    np.testing.assert_array_equal(image.pixels, [[[128, 128, 128]]])


def test_settings_update_during_submission_uses_the_new_frame(
    display: Display,
    red_settings: DisplaySettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialize a settings update behind an in-progress frame submission."""
    display.submit(np.full((1, 1, 1), 64, dtype=np.uint16))
    entered = Event()
    release = Event()
    updating = Event()
    original = FrameRenderer.render_into
    metadata = FrameMetadata(sequence_index=18)

    def blocked_render(
        renderer: FrameRenderer,
        frame: FrameArray,
        output: npt.NDArray[np.uint8],
    ) -> None:
        if not entered.is_set():
            entered.set()
            assert release.wait(5.0)
        original(renderer, frame, output)

    def update_settings() -> None:
        updating.set()
        display.set_settings(red_settings)

    monkeypatch.setattr(FrameRenderer, "render_into", blocked_render)

    with ThreadPoolExecutor(max_workers=2) as executor:
        submission = executor.submit(
            display.submit,
            np.full((1, 1, 1), 128, dtype=np.uint16),
            metadata=metadata,
        )
        try:
            assert entered.wait(5.0)
            update = executor.submit(update_settings)
            assert updating.wait(5.0)
        finally:
            release.set()

        submission.result(timeout=5.0)
        update.result(timeout=5.0)

    image = display.latest_image
    assert image is not None
    assert display.settings is red_settings
    assert image.metadata is metadata
    np.testing.assert_array_equal(image.pixels, [[[128, 0, 0]]])
