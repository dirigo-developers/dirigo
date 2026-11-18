"""Test requested raster geometry and system optical calibration."""

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from dirigo import units
from dirigo.hardware.config import load_system
from dirigo.optics import LaserScanningOptics
from dirigo.raster import (
    RasterFrameAcquisitionSpec,
    resolve_raster_frame_geometry,
)


def _spec() -> RasterFrameAcquisitionSpec:
    """Request an exactly representable image grid."""
    return RasterFrameAcquisitionSpec(
        pixel_width=units.Length("1 um"),
        pixel_height=units.Length("1 um"),
        frame_width=units.Length("128 um"),
        frame_height=units.Length("64 um"),
        bidirectional=True,
        frames_per_acquisition=10,
    )


def _optics() -> LaserScanningOptics:
    """Provide a known conversion from object extent to scanner angle."""
    return LaserScanningOptics(
        objective_focal_length=units.Length("10 mm"),
        relay_magnification=2.0,
    )


def test_exact_grid_and_optical_spans() -> None:
    """Preserve requested dimensions and calculate complete field angles."""
    spec = _spec()
    geometry = resolve_raster_frame_geometry(spec, _optics())

    assert geometry.spec is spec
    assert geometry.shape == (64, 128)
    assert geometry.pixel_width == spec.pixel_width
    assert geometry.pixel_height == spec.pixel_height

    assert float(geometry.fast_angle_span) == pytest.approx(0.0256)
    assert float(geometry.slow_angle_span) == pytest.approx(0.0128)


def test_rounding_reports_effective_spacing_and_preserves_request() -> None:
    """Resolve fractional counts without changing the specification."""
    spec = replace(
        _spec(),
        pixel_width=units.Length("0.3 um"),
        pixel_height=units.Length("0.4 um"),
        frame_width=units.Length("100 um"),
        frame_height=units.Length("50 um"),
    )

    geometry = resolve_raster_frame_geometry(
        spec,
        _optics(),
        row_multiple=2,
    )

    assert geometry.shape == (126, 333)
    assert spec.pixel_width == units.Length("0.3 um")
    assert spec.pixel_height == units.Length("0.4 um")

    assert float(geometry.pixel_width) == pytest.approx(100e-6 / 333)
    assert float(geometry.pixel_height) == pytest.approx(50e-6 / 126)

    assert float(geometry.fast_angle_span) == pytest.approx(0.02)
    assert float(geometry.slow_angle_span) == pytest.approx(0.01)


def test_excessive_pixel_size_adjustment_is_rejected() -> None:
    """Avoid silently producing an image with substantially different spacing."""
    spec = replace(
        _spec(),
        frame_width=units.Length("10 um"),
        pixel_width=units.Length("3 um"),
    )

    with pytest.raises(ValueError, match=r"pixel_width.*adjustment"):
        resolve_raster_frame_geometry(spec, _optics())


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"pixel_width": units.Length(0)}, ValueError),
        ({"frame_height": units.Length(float("nan"))}, ValueError),
        ({"pixel_height": units.Time("1 ms")}, TypeError),
        ({"bidirectional": 1}, TypeError),
        ({"frames_per_acquisition": True}, TypeError),
        ({"frames_per_acquisition": 0}, ValueError),
    ],
)
def test_invalid_specification_is_rejected(
    changes: dict[str, object],
    error: type[Exception],
) -> None:
    """Reject invalid dimensions and acquisition options at construction."""
    with pytest.raises(error):
        replace(_spec(), **changes)


@pytest.mark.parametrize(
    ("multiple", "error"),
    [
        (True, TypeError),
        (2.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_invalid_row_multiple_is_rejected(
    multiple: object,
    error: type[Exception],
) -> None:
    """Require a positive integer grouping constraint."""
    with pytest.raises(error):
        resolve_raster_frame_geometry(
            _spec(),
            _optics(),
            row_multiple=cast(int, multiple),
        )


def test_unlimited_acquisition_has_the_same_frame_geometry() -> None:
    """Keep acquisition length independent of a single frame's geometry."""
    spec = replace(_spec(), frames_per_acquisition=None)

    geometry = resolve_raster_frame_geometry(spec, _optics())

    assert geometry.shape == (64, 128)
    assert geometry.spec.frames_per_acquisition is None


def test_system_loads_optical_calibration(
    tmp_path: Path,
) -> None:
    """Parse calibration before use without requiring device plugins."""
    path = tmp_path / "system.toml"
    path.write_text(
        """
[devices]

[optics]
objective_focal_length = "10 mm"
relay_magnification = 2.0
fast_axis_correction = 1.1
slow_axis_correction = 0.9
""",
        encoding="utf-8",
    )

    with closing(
        load_system(
            path,
            drivers={},
            sync_backends={},
        )
    ) as system:
        optics = system.require_optics()
        geometry = resolve_raster_frame_geometry(_spec(), optics)

        assert float(geometry.fast_angle_span) == pytest.approx(0.02816)
        assert float(geometry.slow_angle_span) == pytest.approx(0.01152)
