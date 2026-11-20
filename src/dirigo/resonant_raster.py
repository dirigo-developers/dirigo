"""Plan nominal resonant raster timing without accessing hardware."""

import math
from dataclasses import dataclass

import numpy as np

from dirigo import units
from dirigo.hardware.scanner import ResonantScannerConfig
from dirigo.hardware.waveform import AngularWaveform
from dirigo.raster import RasterFrameGeometry


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantRasterPlan:
    """Resolved capture geometry and one nominal slow-axis frame waveform.

    Sample ranges are half-open windows at the declared nominal frequency.
    They are relative to each digitizer record trigger with zero capture
    offset. Frequency bounds reserve timing margin; they are not measurements.

    Waveform time zero is the frame-trigger edge. Alignment between that edge
    and acquired-record indices remains a synchronization calibration.
    """

    geometry: RasterFrameGeometry
    sample_rate: units.SampleRate
    nominal_frequency: units.Frequency
    trigger_phase: units.Angle
    line_fill_fraction: float
    fast_amplitude: units.Angle
    record_length: int
    image_records: int
    records_per_frame: int
    nominal_forward_samples: range
    nominal_reverse_samples: range | None
    slow_waveform: AngularWaveform

    @property
    def records_per_batch(self) -> int:
        """Deliver one logical frame, including its blank records, per batch."""
        return self.records_per_frame

    @property
    def batch_count(self) -> int | None:
        """Requested frame count, or None for unlimited acquisition."""
        return self.geometry.spec.frames_per_acquisition

    @property
    def blank_records(self) -> int:
        """Additional record periods reserved for return and timing margins."""
        return self.records_per_frame - self.image_records

    @property
    def nominal_frame_rate(self) -> units.Frequency:
        """Frame rate derived from the declared resonant frequency."""
        return units.Frequency(float(self.nominal_frequency) / self.records_per_frame)


def _positive_quantity(
    value: units.Quantity,
    kind: type[units.Quantity],
    name: str,
    *,
    allow_zero: bool = False,
) -> float:
    """Validate a timing input and return its base-unit value."""
    if not isinstance(value, kind):
        raise TypeError(f"{name} must be a {kind.__name__}.")

    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")

    return number


def _ceil_samples(value: float) -> int:
    """Round up while treating a few floating-point ULPs as integer roundoff."""
    if not math.isfinite(value) or value < 0:
        raise ValueError("Calculated sample count must be finite and nonnegative.")

    nearest = round(value)
    if abs(value - nearest) <= 8 * math.ulp(value):
        return nearest

    return math.ceil(value)


def _sample_window(
    center: float,
    half_width: float,
    samples_per_radian: float,
) -> range:
    """Select nominal sample times within a half-open useful phase window."""
    return range(
        _ceil_samples((center - half_width) * samples_per_radian),
        _ceil_samples((center + half_width) * samples_per_radian),
    )


def plan_resonant_raster_frame(
    geometry: RasterFrameGeometry,
    scanner: ResonantScannerConfig,
    *,
    sample_rate: units.SampleRate,
    max_record_rate: units.SampleRate,
    record_length_minimum: int,
    record_length_step: int,
    line_fill_fraction: float,
    trigger_phase: units.Angle,
    capture_rearm_time: units.Time,
    slow_sample_rate: units.SampleRate,
    slow_flyback_time: units.Time,
    output_rearm_time: units.Time,
) -> ResonantRasterPlan:
    """Plan one record per resonant period and one waveform per frame.

    Define fast-axis angle as:
        -(fast_amplitude / 2) * cos(2*pi*f*t + trigger_phase)

    Thus phase zero is the negative turning point and the forward center
    crossing occurs at pi/2. trigger_phase describes the scanner phase at the
    digitizer's recognized record trigger, including fixed trigger-path delay.

    line_fill_fraction is the useful fraction of each half-period.
    Capture begins at offset zero and includes the complete useful forward
    sweep, plus the following reverse sweep for bidirectional imaging.

    Record lengths are positive multiples of record_length_step and at least
    record_length_minimum. Reject windows that cannot cover the slowest
    scanner while leaving capture rearm time before the fastest next record.

    The slow command is a nominal linear ramp through image-row centers,
    followed by a hold if needed and a quintic return. Its return time is
    quantized upward to the AO sample grid. This does not certify galvo
    tracking or establish measured image coordinates.
    """
    if not isinstance(geometry, RasterFrameGeometry):
        raise TypeError("geometry must be a RasterFrameGeometry.")

    if not isinstance(scanner, ResonantScannerConfig):
        raise TypeError("scanner must be a ResonantScannerConfig.")

    for name, value in (
        ("record_length_minimum", record_length_minimum),
        ("record_length_step", record_length_step),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    if type(line_fill_fraction) not in (int, float):
        raise TypeError("line_fill_fraction must be a plain number.")

    fill = float(line_fill_fraction)
    if not 0 < fill <= 1:
        raise ValueError("line_fill_fraction must be in (0, 1].")

    if not isinstance(trigger_phase, units.Angle):
        raise TypeError("trigger_phase must be an Angle.")

    if not math.isfinite(float(trigger_phase)):
        raise ValueError("trigger_phase must be finite.")

    fs = _positive_quantity(
        sample_rate,
        units.SampleRate,
        "sample_rate",
    )
    upper_rate = _positive_quantity(
        max_record_rate,
        units.SampleRate,
        "max_record_rate",
    )
    ao_rate = _positive_quantity(
        slow_sample_rate,
        units.SampleRate,
        "slow_sample_rate",
    )
    capture_rearm = _positive_quantity(
        capture_rearm_time,
        units.Time,
        "capture_rearm_time",
        allow_zero=True,
    )
    flyback = _positive_quantity(
        slow_flyback_time,
        units.Time,
        "slow_flyback_time",
    )
    output_rearm = _positive_quantity(
        output_rearm_time,
        units.Time,
        "output_rearm_time",
        allow_zero=True,
    )

    frequency = float(scanner.nominal_frequency)
    minimum_rate = frequency * (1 - scanner.frequency_tolerance)
    physical_maximum = frequency * (1 + scanner.frequency_tolerance)

    if not math.isfinite(minimum_rate) or minimum_rate <= 0:
        raise ValueError("The scanner's minimum frequency is invalid.")

    if upper_rate < physical_maximum:
        raise ValueError(
            "max_record_rate must cover the scanner's frequency tolerance."
        )

    lines_per_record = 2 if geometry.spec.bidirectional else 1

    if geometry.rows <= 0 or geometry.rows % lines_per_record:
        raise ValueError(
            "Image rows must be a positive multiple of lines per record; "
            "resolve geometry with the appropriate row_multiple."
        )

    image_records = geometry.rows // lines_per_record
    phase = float(trigger_phase) % math.tau
    forward_center = (math.pi / 2 - phase) % math.tau
    half_window = math.pi * fill / 2

    if forward_center < half_window:
        raise ValueError(
            "The useful forward sweep begins before the record trigger; "
            "zero-offset capture cannot cover this trigger phase."
        )

    reverse_center = forward_center + math.pi if geometry.spec.bidirectional else None
    last_center = forward_center if reverse_center is None else reverse_center
    end_phase = last_center + half_window

    needed = _ceil_samples(end_phase / math.tau / minimum_rate * fs)
    needed = max(record_length_minimum, needed)

    record_length = (
        (needed + record_length_step - 1) // record_length_step
    ) * record_length_step

    if record_length / fs + capture_rearm >= 1 / upper_rate:
        raise ValueError(
            "The record window and capture rearm time do not fit "
            "before the fastest next record trigger."
        )

    nominal_scale = fs / (math.tau * frequency)
    forward_samples = _sample_window(
        forward_center,
        half_window,
        nominal_scale,
    )
    reverse_samples = (
        None
        if reverse_center is None
        else _sample_window(
            reverse_center,
            half_window,
            nominal_scale,
        )
    )

    windows = [forward_samples]
    if reverse_samples is not None:
        windows.append(reverse_samples)

    if any(len(window) == 0 for window in windows):
        raise ValueError("A useful scan window contains no ADC samples.")

    if any(window.stop > record_length for window in windows):
        raise ValueError("A useful scan window exceeds the planned record.")

    amplitude = float(geometry.fast_angle_span) / math.sin(math.pi * fill / 2)
    if (
        not math.isfinite(amplitude)
        or amplitude <= 0
        or amplitude > float(scanner.max_amplitude)
    ):
        raise ValueError(
            "The required resonant amplitude exceeds the scanner's limits."
        )

    line_period = 1 / (frequency * lines_per_record)
    first_center_time = forward_center / (math.tau * frequency)

    ramp_start = first_center_time - line_period / 2
    ramp_duration = geometry.rows * line_period
    ramp_end = ramp_start + ramp_duration

    last_image_end = ((image_records - 1) * math.tau + end_phase) / (
        math.tau * minimum_rate
    )
    flyback_start = max(ramp_end, last_image_end)

    ramp_samples = max(
        1,
        _ceil_samples(flyback_start * ao_rate),
    )
    return_intervals = max(
        1,
        _ceil_samples(flyback * ao_rate),
    )

    times = np.arange(ramp_samples, dtype=np.float64) / ao_rate
    fraction = np.clip(
        (times - ramp_start) / ramp_duration,
        0.0,
        1.0,
    )
    slow_span = float(geometry.slow_angle_span)
    ramp = (fraction - 0.5) * slow_span

    u = np.linspace(0.0, 1.0, return_intervals + 1)
    smooth = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
    np.clip(smooth, 0.0, 1.0, out=smooth)

    returning = slow_span / 2 + (ramp[0] - slow_span / 2) * smooth

    angles = np.concatenate((ramp, returning))
    angles[-1] = angles[0]

    waveform = AngularWaveform(
        angles,
        sample_rate=slow_sample_rate,
    )

    duration = angles.size / ao_rate
    required_period = duration + output_rearm
    required_records = required_period * upper_rate

    if not math.isfinite(required_records):
        raise ValueError("Calculated frame period cannot be represented.")

    records_per_frame = max(
        4,
        image_records,
        math.floor(required_records) + 1,
    )

    if records_per_frame / upper_rate <= required_period:
        records_per_frame += 1

    return ResonantRasterPlan(
        geometry=geometry,
        sample_rate=sample_rate,
        nominal_frequency=scanner.nominal_frequency,
        trigger_phase=units.Angle(phase),
        line_fill_fraction=fill,
        fast_amplitude=units.Angle(amplitude),
        record_length=record_length,
        image_records=image_records,
        records_per_frame=records_per_frame,
        nominal_forward_samples=forward_samples,
        nominal_reverse_samples=reverse_samples,
        slow_waveform=waveform,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantRasterSettings:
    """Installed timing and execution settings for resonant raster acquisition.

    trigger_phase uses the phase convention of plan_resonant_raster_frame().
    read_timeout is a positive polling interval in seconds, not an overall
    acquisition deadline.

    When stop_fast_scanner_on_close is True, a worker that attempted an
    amplitude command requests zero amplitude after closing its run resources.
    """

    line_fill_fraction: float
    trigger_phase: units.Angle
    capture_rearm_time: units.Time
    slow_sample_rate: units.SampleRate
    slow_flyback_time: units.Time
    output_rearm_time: units.Time
    slow_settling_time: units.Time
    read_timeout: float = 0.05
    max_pixel_size_adjustment: float = 0.01
    stop_fast_scanner_on_close: bool = True

    def __post_init__(self) -> None:
        """Validate declared timing values without accessing hardware."""
        for name in (
            "line_fill_fraction",
            "read_timeout",
            "max_pixel_size_adjustment",
        ):
            value = getattr(self, name)

            if type(value) not in (int, float):
                raise TypeError(f"{name} must be a plain number.")

            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if not 0 < self.line_fill_fraction <= 1:
            raise ValueError("line_fill_fraction must be in (0, 1].")

        if self.read_timeout <= 0:
            raise ValueError("read_timeout must be positive.")

        if not 0 <= self.max_pixel_size_adjustment < 1:
            raise ValueError("max_pixel_size_adjustment must be in [0, 1).")

        if not isinstance(self.trigger_phase, units.Angle):
            raise TypeError("trigger_phase must be an Angle.")

        if not math.isfinite(float(self.trigger_phase)):
            raise ValueError("trigger_phase must be finite.")

        _positive_quantity(
            self.slow_sample_rate,
            units.SampleRate,
            "slow_sample_rate",
        )
        _positive_quantity(
            self.slow_flyback_time,
            units.Time,
            "slow_flyback_time",
        )

        for name in (
            "capture_rearm_time",
            "output_rearm_time",
            "slow_settling_time",
        ):
            _positive_quantity(
                getattr(self, name),
                units.Time,
                name,
                allow_zero=True,
            )

        if type(self.stop_fast_scanner_on_close) is not bool:
            raise TypeError("stop_fast_scanner_on_close must be a bool.")
