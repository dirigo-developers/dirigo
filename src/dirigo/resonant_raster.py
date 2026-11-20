"""Plan nominal resonant raster timing without accessing hardware."""

import math
from dataclasses import dataclass

import numpy as np

from dirigo import units
from dirigo.hardware.scanner import ResonantScannerConfig
from dirigo.hardware.waveform import AngularWaveform
from dirigo.raster import RasterFrameGeometry, RasterLineAcquisitionSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantFastAxisPlan:
    """Describe resonant capture independently of frame or slow-axis motion.

    Sample windows are nominal half-open ranges within each record.
    maximum_record_period and latest_useful_sample_end use the slowest
    permitted scanner frequency. All times are relative to record triggers.
    """

    sample_rate: units.SampleRate
    nominal_frequency: units.Frequency
    trigger_phase: units.Angle
    line_fill_fraction: float
    fast_amplitude: units.Angle
    record_length: int
    nominal_forward_samples: range
    nominal_reverse_samples: range | None
    nominal_first_line_center: units.Time
    maximum_record_period: units.Time
    latest_useful_sample_end: units.Time

    @property
    def lines_per_record(self) -> int:
        """Number of useful sweeps captured in each record."""
        return 2 if self.nominal_reverse_samples is not None else 1

    @property
    def nominal_line_period(self) -> units.Time:
        """Nominal separation between useful line centers."""
        return units.Time(1 / (float(self.nominal_frequency) * self.lines_per_record))


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantLinePlan:
    """Combine fast-axis capture with finite line grouping."""

    spec: RasterLineAcquisitionSpec
    columns: int
    fast_axis: ResonantFastAxisPlan
    records_per_batch: int
    batch_count: int

    @property
    def pixel_width(self) -> units.Length:
        """Resolved spacing while preserving the requested line width."""
        return units.Length(float(self.spec.line_width) / self.columns)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantRasterPlan:
    """Combine reusable fast-axis capture with slow-axis frame motion."""

    geometry: RasterFrameGeometry
    fast_axis: ResonantFastAxisPlan
    image_records: int
    records_per_frame: int
    slow_waveform: AngularWaveform

    @property
    def records_per_batch(self) -> int:
        """Number of records in the current one-frame transfer batch."""
        return self.records_per_frame

    @property
    def batch_count(self) -> int | None:
        """Requested number of frame batches, or None for unlimited capture."""
        return self.geometry.spec.frames_per_acquisition

    @property
    def blank_records(self) -> int:
        """Number of records outside the useful image portion of each frame."""
        return self.records_per_frame - self.image_records

    @property
    def nominal_frame_rate(self) -> units.Frequency:
        """Nominal frame cadence before frequency drift."""
        return units.Frequency(
            float(self.fast_axis.nominal_frequency) / self.records_per_frame
        )

    @property
    def sample_rate(self) -> units.SampleRate:
        """Nominal ADC sample rate."""
        return self.fast_axis.sample_rate

    @property
    def nominal_frequency(self) -> units.Frequency:
        """Nominal fast-scanner cycle frequency."""
        return self.fast_axis.nominal_frequency

    @property
    def trigger_phase(self) -> units.Angle:
        """Normalized resonant phase at a record trigger."""
        return self.fast_axis.trigger_phase

    @property
    def line_fill_fraction(self) -> float:
        """Useful fraction of each resonant half-cycle."""
        return self.fast_axis.line_fill_fraction

    @property
    def fast_amplitude(self) -> units.Angle:
        """Required peak-to-peak optical amplitude."""
        return self.fast_axis.fast_amplitude

    @property
    def record_length(self) -> int:
        """Number of ADC samples per record."""
        return self.fast_axis.record_length

    @property
    def nominal_forward_samples(self) -> range:
        """Nominal useful forward-sweep samples."""
        return self.fast_axis.nominal_forward_samples

    @property
    def nominal_reverse_samples(self) -> range | None:
        """Nominal useful reverse-sweep samples, when requested."""
        return self.fast_axis.nominal_reverse_samples


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


def plan_resonant_fast_axis(
    angle_span: units.Angle,
    scanner: ResonantScannerConfig,
    *,
    bidirectional: bool,
    sample_rate: units.SampleRate,
    max_record_rate: units.SampleRate,
    record_length_minimum: int,
    record_length_step: int,
    line_fill_fraction: float,
    trigger_phase: units.Angle,
    capture_rearm_time: units.Time,
) -> ResonantFastAxisPlan:
    """Resolve one repeating capture cycle without accessing hardware.

    Use x(t) = -(amplitude / 2) * cos(2*pi*f*t + trigger_phase).
    A bidirectional record contains both the forward and reverse sweeps.
    Reject records that cannot contain the useful data and leave rearm time.
    """
    if not isinstance(scanner, ResonantScannerConfig):
        raise TypeError("scanner must be a ResonantScannerConfig.")
    if not isinstance(bidirectional, bool):
        raise TypeError("bidirectional must be a bool.")

    span = _positive_quantity(angle_span, units.Angle, "angle_span")
    rate = _positive_quantity(sample_rate, units.SampleRate, "sample_rate")
    maximum_rate = _positive_quantity(
        max_record_rate, units.SampleRate, "max_record_rate"
    )
    rearm = _positive_quantity(
        capture_rearm_time, units.Time, "capture_rearm_time", allow_zero=True
    )

    for name, value in (
        ("record_length_minimum", record_length_minimum),
        ("record_length_step", record_length_step),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int.")
        if value < 1:
            raise ValueError(f"{name} must be positive.")

    if isinstance(line_fill_fraction, (bool, units.Quantity)) or not isinstance(
        line_fill_fraction, (int, float)
    ):
        raise TypeError("line_fill_fraction must be a plain number.")
    if not math.isfinite(line_fill_fraction) or not 0 < line_fill_fraction <= 1:
        raise ValueError("line_fill_fraction must satisfy 0 < value <= 1.")

    if not isinstance(trigger_phase, units.Angle):
        raise TypeError("trigger_phase must be an Angle.")
    if not math.isfinite(float(trigger_phase)):
        raise ValueError("trigger_phase must be finite.")

    frequency = float(scanner.nominal_frequency)
    minimum_frequency = frequency * (1 - scanner.frequency_tolerance)
    maximum_frequency = frequency * (1 + scanner.frequency_tolerance)
    if maximum_rate < maximum_frequency:
        raise ValueError("max_record_rate must cover the scanner frequency range.")

    half_window = math.pi * line_fill_fraction / 2
    amplitude = span / math.sin(half_window)
    if amplitude > float(scanner.max_amplitude):
        raise ValueError("Required fast amplitude exceeds scanner max_amplitude.")

    phase = float(trigger_phase) % math.tau
    forward_center = (math.pi / 2 - phase) % math.tau
    if forward_center - half_window < 0:
        raise ValueError("The useful forward sweep begins before the record trigger.")

    last_center = forward_center + (math.pi if bidirectional else 0)
    last_phase = last_center + half_window
    latest_end = last_phase / (math.tau * minimum_frequency)

    length = max(record_length_minimum, _ceil_samples(latest_end * rate))
    length = (
        (length + record_length_step - 1) // record_length_step * record_length_step
    )
    if length / rate + rearm >= 1 / maximum_rate:
        raise ValueError("record_length leaves insufficient capture rearm time.")

    samples_per_radian = rate / (math.tau * frequency)
    forward = _sample_window(forward_center, half_window, samples_per_radian)
    reverse = (
        _sample_window(forward_center + math.pi, half_window, samples_per_radian)
        if bidirectional
        else None
    )
    for window in (forward, reverse):
        if window is not None and (
            not window or window.start < 0 or window.stop > length
        ):
            raise ValueError("Useful sample windows do not fit the record_length.")

    return ResonantFastAxisPlan(
        sample_rate=sample_rate,
        nominal_frequency=scanner.nominal_frequency,
        trigger_phase=units.Angle(phase),
        line_fill_fraction=line_fill_fraction,
        fast_amplitude=units.Angle(amplitude),
        record_length=length,
        nominal_forward_samples=forward,
        nominal_reverse_samples=reverse,
        nominal_first_line_center=units.Time(forward_center / (math.tau * frequency)),
        maximum_record_period=units.Time(1 / minimum_frequency),
        latest_useful_sample_end=units.Time(latest_end),
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
    """Add slow-axis frame motion around a reusable resonant capture plan."""
    if not isinstance(geometry, RasterFrameGeometry):
        raise TypeError("geometry must be a RasterFrameGeometry.")

    fast = plan_resonant_fast_axis(
        geometry.fast_angle_span,
        scanner,
        bidirectional=geometry.spec.bidirectional,
        sample_rate=sample_rate,
        max_record_rate=max_record_rate,
        record_length_minimum=record_length_minimum,
        record_length_step=record_length_step,
        line_fill_fraction=line_fill_fraction,
        trigger_phase=trigger_phase,
        capture_rearm_time=capture_rearm_time,
    )
    if geometry.rows % fast.lines_per_record:
        raise ValueError(
            f"geometry.rows must be divisible by row_multiple={fast.lines_per_record}."
        )
    image_records = geometry.rows // fast.lines_per_record

    output_rate = _positive_quantity(
        slow_sample_rate, units.SampleRate, "slow_sample_rate"
    )
    flyback_time = _positive_quantity(
        slow_flyback_time, units.Time, "slow_flyback_time"
    )
    output_rearm = _positive_quantity(
        output_rearm_time, units.Time, "output_rearm_time", allow_zero=True
    )

    line_period = float(fast.nominal_line_period)
    ramp_start = float(fast.nominal_first_line_center) - line_period / 2
    ramp_duration = geometry.rows * line_period
    ramp_end = ramp_start + ramp_duration
    last_image_end = (image_records - 1) * float(fast.maximum_record_period) + float(
        fast.latest_useful_sample_end
    )
    flyback_start = max(ramp_end, last_image_end)

    ramp_samples = max(1, _ceil_samples(flyback_start * output_rate))
    return_intervals = max(1, _ceil_samples(flyback_time * output_rate))
    times = np.arange(ramp_samples, dtype=np.float64) / output_rate
    fractions = np.clip((times - ramp_start) / ramp_duration, 0.0, 1.0)
    span = float(geometry.slow_angle_span)
    ramp = (fractions - 0.5) * span

    u = np.linspace(0.0, 1.0, return_intervals + 1)
    smooth = np.clip(u**3 * (10 + u * (-15 + 6 * u)), 0.0, 1.0)
    returned = span / 2 + (float(ramp[0]) - span / 2) * smooth
    samples = np.concatenate((ramp, returned))
    samples[-1] = samples[0]
    waveform = AngularWaveform(samples, sample_rate=slow_sample_rate)

    required_period = samples.size / output_rate + output_rearm
    maximum_rate = float(max_record_rate)
    records_per_frame = max(
        4,
        image_records,
        math.floor(required_period * maximum_rate) + 1,
    )
    if records_per_frame / maximum_rate <= required_period:
        records_per_frame += 1

    return ResonantRasterPlan(
        geometry=geometry,
        fast_axis=fast,
        image_records=image_records,
        records_per_frame=records_per_frame,
        slow_waveform=waveform,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantScanSettings:
    """Settings shared by resonant line and frame acquisition."""

    line_fill_fraction: float
    trigger_phase: units.Angle
    capture_rearm_time: units.Time
    read_timeout: float = 0.05
    max_pixel_size_adjustment: float = 0.01
    stop_fast_scanner_on_close: bool = True

    def __post_init__(self) -> None:
        """Validate useful scan extent, phase, margins, and execution policy."""
        for name in (
            "line_fill_fraction",
            "read_timeout",
            "max_pixel_size_adjustment",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, units.Quantity)) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(f"{name} must be a plain number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")

        if not 0 < self.line_fill_fraction <= 1:
            raise ValueError("line_fill_fraction must satisfy 0 < value <= 1.")
        if self.read_timeout <= 0:
            raise ValueError("read_timeout must be positive.")
        if not 0 <= self.max_pixel_size_adjustment < 1:
            raise ValueError("max_pixel_size_adjustment must satisfy 0 <= value < 1.")
        if not isinstance(self.trigger_phase, units.Angle):
            raise TypeError("trigger_phase must be an Angle.")
        if not math.isfinite(float(self.trigger_phase)):
            raise ValueError("trigger_phase must be finite.")
        _positive_quantity(
            self.capture_rearm_time,
            units.Time,
            "capture_rearm_time",
            allow_zero=True,
        )
        if not isinstance(self.stop_fast_scanner_on_close, bool):
            raise TypeError("stop_fast_scanner_on_close must be a bool.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ResonantRasterSettings(ResonantScanSettings):
    """Extend resonant scan settings with slow-axis frame motion."""

    slow_sample_rate: units.SampleRate
    slow_flyback_time: units.Time
    output_rearm_time: units.Time
    slow_settling_time: units.Time

    def __post_init__(self) -> None:
        """Validate shared settings and the additional frame timing."""
        ResonantScanSettings.__post_init__(self)
        _positive_quantity(self.slow_sample_rate, units.SampleRate, "slow_sample_rate")
        _positive_quantity(self.slow_flyback_time, units.Time, "slow_flyback_time")
        _positive_quantity(
            self.output_rearm_time,
            units.Time,
            "output_rearm_time",
            allow_zero=True,
        )
        _positive_quantity(
            self.slow_settling_time,
            units.Time,
            "slow_settling_time",
            allow_zero=True,
        )
