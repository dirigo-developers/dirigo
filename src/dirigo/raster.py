"""Requested raster-frame geometry and its resolved pixel grid."""

import math
from dataclasses import dataclass

from dirigo import units
from dirigo.optics import LaserScanningOptics


@dataclass(frozen=True, slots=True, kw_only=True)
class RasterLineAcquisitionSpec:
    """Request finite repeated-line capture in complete publication batches.

    Counts refer to useful image lines. Bidirectional records may contain two
    lines; the selected planner validates pairing and batch divisibility.
    """

    pixel_width: units.Length
    line_width: units.Length
    bidirectional: bool
    lines_per_acquisition: int
    lines_per_batch: int

    def __post_init__(self) -> None:
        """Require positive physical dimensions and positive integer counts."""
        for name in ("pixel_width", "line_width"):
            value = getattr(self, name)
            if not isinstance(value, units.Length):
                raise TypeError(f"{name} must be a Length.")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional must be a bool.")
        for name in ("lines_per_acquisition", "lines_per_batch"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an int.")
            if value < 1:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RasterFrameAcquisitionSpec:
    """Request physical image dimensions, pixel spacing, and acquisition length.

    Pixel spacings are requested values. Geometry resolution preserves frame
    dimensions and calculates effective spacings from integer pixel counts.

    frames_per_acquisition=None requests acquisition until cancellation.
    Particular capture backends may reject unlimited acquisition.
    """

    pixel_width: units.Length
    pixel_height: units.Length
    frame_width: units.Length
    frame_height: units.Length
    bidirectional: bool
    frames_per_acquisition: int | None

    def __post_init__(self) -> None:
        """Require typed, finite, positive dimensions and valid scan options."""
        for name in (
            "pixel_width",
            "pixel_height",
            "frame_width",
            "frame_height",
        ):
            value = getattr(self, name)

            if not isinstance(value, units.Length):
                raise TypeError(f"{name} must be a Length.")

            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive.")

        if type(self.bidirectional) is not bool:
            raise TypeError("bidirectional must be a bool.")

        count = self.frames_per_acquisition
        if count is not None:
            if type(count) is not int:
                raise TypeError("frames_per_acquisition must be an integer or None.")
            if count <= 0:
                raise ValueError("frames_per_acquisition must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RasterFrameGeometry:
    """Calculated image geometry returned by resolve_raster_frame_geometry().

    Angular spans cover the useful field of view. Resonant overscan and
    scanner trajectories are calculated separately during timing planning.
    """

    spec: RasterFrameAcquisitionSpec
    columns: int
    rows: int
    fast_angle_span: units.Angle
    slow_angle_span: units.Angle

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape in row, column order."""
        return self.rows, self.columns

    @property
    def pixel_width(self) -> units.Length:
        """Effective horizontal pixel spacing."""
        return units.Length(float(self.spec.frame_width) / self.columns)

    @property
    def pixel_height(self) -> units.Length:
        """Effective vertical pixel spacing."""
        return units.Length(float(self.spec.frame_height) / self.rows)


def _pixel_count(
    extent: units.Length,
    requested: units.Length,
    *,
    multiple: int,
    limit: float,
    name: str,
) -> int:
    """Round the requested count to a multiple and check spacing adjustment."""
    try:
        desired = float(extent) / float(requested) / multiple
    except OverflowError as failure:
        raise ValueError(f"{name} grid cannot be represented.") from failure

    if not math.isfinite(desired):
        raise ValueError(f"{name} grid cannot be represented.")

    # Nearest multiple, choosing the larger count at an exact halfway point.
    count = multiple * math.floor(desired + 0.5)

    if count == 0:
        raise ValueError(
            f"{name} must produce at least one group of {multiple} pixels."
        )

    effective = float(extent) / count
    error = abs(effective / float(requested) - 1.0)

    if error > limit:
        raise ValueError(
            f"{name} requires a pixel-size adjustment of {error:.3%}; "
            f"the limit is {limit:.3%}."
        )

    return count


def resolve_raster_frame_geometry(
    spec: RasterFrameAcquisitionSpec,
    optics: LaserScanningOptics,
    *,
    row_multiple: int = 1,
    max_pixel_size_adjustment: float = 0.01,
) -> RasterFrameGeometry:
    """Resolve requested dimensions into an integer grid and optical spans.

    Preserve frame_width and frame_height. Round columns to the nearest
    integer and rows to the nearest row_multiple, resolving exact halfway
    points toward the larger count.

    Reject effective pixel spacings whose fractional change from the
    requested spacing exceeds max_pixel_size_adjustment.

    row_multiple expresses a planning constraint. For example, a resonant
    arrangement producing two image rows per record can request two.

    This function performs no hardware access or sample-array allocation.
    """
    if not isinstance(spec, RasterFrameAcquisitionSpec):
        raise TypeError("spec must be a RasterFrameAcquisitionSpec.")

    if not isinstance(optics, LaserScanningOptics):
        raise TypeError("optics must be LaserScanningOptics.")

    if type(row_multiple) is not int:
        raise TypeError("row_multiple must be an integer.")

    if row_multiple <= 0:
        raise ValueError("row_multiple must be positive.")

    if type(max_pixel_size_adjustment) not in (int, float):
        raise TypeError("max_pixel_size_adjustment must be a plain number.")

    limit = float(max_pixel_size_adjustment)
    if not math.isfinite(limit) or not 0 <= limit < 1:
        raise ValueError("max_pixel_size_adjustment must be finite and in [0, 1).")

    columns = _pixel_count(
        spec.frame_width,
        spec.pixel_width,
        multiple=1,
        limit=limit,
        name="pixel_width",
    )
    rows = _pixel_count(
        spec.frame_height,
        spec.pixel_height,
        multiple=row_multiple,
        limit=limit,
        name="pixel_height",
    )

    return RasterFrameGeometry(
        spec=spec,
        columns=columns,
        rows=rows,
        fast_angle_span=optics.object_length_to_scan_angle(
            spec.frame_width,
            axis="fast",
        ),
        slow_angle_span=optics.object_length_to_scan_angle(
            spec.frame_height,
            axis="slow",
        ),
    )
