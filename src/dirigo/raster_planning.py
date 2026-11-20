"""Hardware-independent entry point for raster frame planning."""

import math

from dirigo import units
from dirigo.hardware.scanner import GalvoScannerConfig, ResonantScannerConfig
from dirigo.optics import LaserScanningOptics
from dirigo.raster import (
    RasterFrameAcquisitionSpec,
    RasterLineAcquisitionSpec,
    resolve_raster_frame_geometry,
)
from dirigo.resonant_raster import (
    ResonantLinePlan,
    ResonantRasterPlan,
    ResonantRasterSettings,
    ResonantScanSettings,
    plan_resonant_fast_axis,
    plan_resonant_raster_frame,
)


def plan_raster_line(
    spec: RasterLineAcquisitionSpec,
    *,
    optics: LaserScanningOptics,
    fast_scanner_config: ResonantScannerConfig | GalvoScannerConfig,
    settings: ResonantScanSettings,
    sample_rate: units.SampleRate,
    record_length_minimum: int,
    record_length_step: int,
) -> ResonantLinePlan:
    """Plan finite resonant lines without a slow scanner or frame clock."""
    if not isinstance(spec, RasterLineAcquisitionSpec):
        raise TypeError("spec must be a RasterLineAcquisitionSpec.")
    if not isinstance(optics, LaserScanningOptics):
        raise TypeError("optics must be LaserScanningOptics.")
    if not isinstance(fast_scanner_config, ResonantScannerConfig):
        raise NotImplementedError("Only resonant fast-axis planning is implemented.")
    if not isinstance(settings, ResonantScanSettings):
        raise TypeError("settings must be ResonantScanSettings.")

    ratio = float(spec.line_width) / float(spec.pixel_width)
    if not math.isfinite(ratio):
        raise ValueError("Requested line pixel count must be finite.")
    columns = math.floor(ratio + 0.5)
    if columns < 1:
        raise ValueError("Requested line must contain at least one pixel.")
    pixel_width = float(spec.line_width) / columns
    adjustment = abs(pixel_width / float(spec.pixel_width) - 1)
    if adjustment > settings.max_pixel_size_adjustment:
        raise ValueError("Required pixel_width adjustment exceeds the allowed limit.")

    maximum_rate = units.SampleRate(
        float(fast_scanner_config.nominal_frequency)
        * (1 + fast_scanner_config.frequency_tolerance)
    )
    fast = plan_resonant_fast_axis(
        optics.object_length_to_scan_angle(spec.line_width, axis="fast"),
        fast_scanner_config,
        bidirectional=spec.bidirectional,
        sample_rate=sample_rate,
        max_record_rate=maximum_rate,
        record_length_minimum=record_length_minimum,
        record_length_step=record_length_step,
        line_fill_fraction=settings.line_fill_fraction,
        trigger_phase=settings.trigger_phase,
        capture_rearm_time=settings.capture_rearm_time,
    )
    if spec.lines_per_batch % fast.lines_per_record:
        raise ValueError("lines_per_batch must contain complete sweep pairs.")
    if spec.lines_per_acquisition % spec.lines_per_batch:
        raise ValueError("lines_per_acquisition must be a multiple of lines_per_batch.")

    return ResonantLinePlan(
        spec=spec,
        columns=columns,
        fast_axis=fast,
        records_per_batch=spec.lines_per_batch // fast.lines_per_record,
        batch_count=spec.lines_per_acquisition // spec.lines_per_batch,
    )


def plan_raster_frame(
    spec: RasterFrameAcquisitionSpec,
    *,
    optics: LaserScanningOptics,
    fast_scanner_config: ResonantScannerConfig | GalvoScannerConfig,
    slow_scanner_config: GalvoScannerConfig,
    settings: ResonantRasterSettings,
    sample_rate: units.SampleRate,
    max_record_rate: units.SampleRate,
    record_length_minimum: int,
    record_length_step: int,
) -> ResonantRasterPlan:
    """Resolve geometry and select the currently supported scan calculation.

    Inputs describe devices and constraints; this function accesses no
    hardware. Galvo fast-axis planning is not implemented yet.
    """
    if not isinstance(spec, RasterFrameAcquisitionSpec):
        raise TypeError("spec must be a RasterFrameAcquisitionSpec.")
    if not isinstance(slow_scanner_config, GalvoScannerConfig):
        raise TypeError("slow_scanner_config must be a GalvoScannerConfig.")
    if not isinstance(fast_scanner_config, ResonantScannerConfig):
        raise NotImplementedError("Only resonant fast-axis planning is implemented.")
    if not isinstance(settings, ResonantRasterSettings):
        raise TypeError("settings must be ResonantRasterSettings.")

    geometry = resolve_raster_frame_geometry(
        spec,
        optics,
        row_multiple=2 if spec.bidirectional else 1,
        max_pixel_size_adjustment=settings.max_pixel_size_adjustment,
    )
    plan = plan_resonant_raster_frame(
        geometry,
        fast_scanner_config,
        sample_rate=sample_rate,
        max_record_rate=max_record_rate,
        record_length_minimum=record_length_minimum,
        record_length_step=record_length_step,
        line_fill_fraction=settings.line_fill_fraction,
        trigger_phase=settings.trigger_phase,
        capture_rearm_time=settings.capture_rearm_time,
        slow_sample_rate=settings.slow_sample_rate,
        slow_flyback_time=settings.slow_flyback_time,
        output_rearm_time=settings.output_rearm_time,
    )
    angles = plan.slow_waveform.samples
    bounds = slow_scanner_config.angle_range
    if angles.min() < float(bounds.min) or angles.max() > float(bounds.max):
        raise ValueError("Slow-axis waveform exceeds the configured angle_range.")
    return plan
