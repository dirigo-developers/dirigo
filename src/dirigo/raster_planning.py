"""Hardware-independent entry point for raster frame planning."""

from dirigo import units
from dirigo.hardware.scanner import GalvoScannerConfig, ResonantScannerConfig
from dirigo.optics import LaserScanningOptics
from dirigo.raster import RasterFrameAcquisitionSpec, resolve_raster_frame_geometry
from dirigo.resonant_raster import (
    ResonantRasterPlan,
    ResonantRasterSettings,
    plan_resonant_raster_frame,
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
