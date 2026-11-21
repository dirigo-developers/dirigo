"""Time nominal frame reconstruction on a synthetic digitizer batch.

Run from the repository root with python -m benchmarks.resonant_reconstruction.
"""

import argparse
from functools import partial
from time import perf_counter_ns

import numpy as np

from dirigo import units
from dirigo.hardware.capture import CaptureInfo, CaptureLayout
from dirigo.hardware.scanner import GalvoScannerConfig, ResonantScannerConfig
from dirigo.optics import LaserScanningOptics
from dirigo.processors.frame import _reconstruct_frame
from dirigo.raster import RasterFrameAcquisitionSpec
from dirigo.raster_planning import plan_raster_frame
from dirigo.records import RecordBatch
from dirigo.resonant_raster import (
    ResonantLineReconstructor,
    ResonantRasterPlan,
    ResonantRasterSettings,
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def _plan_frame(
    *,
    width: int,
    height: int,
    sample_rate: units.SampleRate,
    frequency: units.Frequency,
) -> ResonantRasterPlan:
    """Plan one bidirectional workload with synthetic scanner settings."""
    pixel_size = units.Length("1 um")
    fast = ResonantScannerConfig(
        nominal_frequency=frequency,
        max_amplitude=units.Angle("20 deg"),
        frequency_tolerance=0.003,
    )
    max_record_rate = units.SampleRate(
        float(frequency) * (1 + fast.frequency_tolerance)
    )
    spec = RasterFrameAcquisitionSpec(
        pixel_width=pixel_size,
        pixel_height=pixel_size,
        frame_width=units.Length(width * float(pixel_size)),
        frame_height=units.Length(height * float(pixel_size)),
        bidirectional=True,
        frames_per_acquisition=None,
    )
    return plan_raster_frame(
        spec,
        optics=LaserScanningOptics(
            objective_focal_length=units.Length("10 mm"), relay_magnification=2.0
        ),
        fast_scanner_config=fast,
        slow_scanner_config=GalvoScannerConfig(
            angle_range=units.AngleRange("-20 deg", "20 deg")
        ),
        settings=ResonantRasterSettings(
            line_fill_fraction=0.8,
            trigger_phase=units.Angle(0),
            capture_rearm_time=units.Time("2 us"),
            slow_sample_rate=units.SampleRate("100 kS/s"),
            slow_flyback_time=units.Time("2 ms"),
            output_rearm_time=units.Time("100 us"),
            slow_settling_time=units.Time(0),
        ),
        sample_rate=sample_rate,
        max_record_rate=max_record_rate,
        record_length_minimum=256,
        record_length_step=32,
    )


def main() -> None:
    """Report first-call and steady-state reconstruction timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=_positive_int, default=1024)
    parser.add_argument("--height", type=_positive_int, default=1024)
    parser.add_argument("--channels", type=_positive_int, default=2)
    parser.add_argument("--sample-rate", type=units.SampleRate, default="125 MS/s")
    parser.add_argument("--frequency", type=units.Frequency, default="8 kHz")
    parser.add_argument("--iterations", type=_positive_int, default=100)
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numpy")
    parser.add_argument("--threads", type=_positive_int, help="Numba worker threads.")
    args = parser.parse_args()

    use_numba = args.backend == "numba"
    backend_label = "NumPy"
    if use_numba:
        from numba import get_num_threads, set_num_threads

        if args.threads is not None:
            try:
                set_num_threads(args.threads)
            except ValueError as exc:
                parser.error(str(exc))
        backend_label = f"Numba ({get_num_threads()} threads)"
    elif args.threads is not None:
        parser.error("--threads requires --backend numba.")

    try:
        plan = _plan_frame(
            width=args.width,
            height=args.height,
            sample_rate=args.sample_rate,
            frequency=args.frequency,
        )
        reconstructor = ResonantLineReconstructor(
            plan.fast_axis, columns=plan.geometry.columns
        )
    except ValueError as exc:
        parser.error(str(exc))

    layout = CaptureLayout(
        shape=(plan.records_per_batch, plan.record_length, args.channels),
        channel_indices=tuple(range(args.channels)),
        dtype=np.dtype(np.uint16),
    )
    sample_code_range = units.IntRange(0, 4095)
    samples = np.random.default_rng(0).integers(
        sample_code_range.min,
        sample_code_range.max + 1,
        size=layout.shape,
        dtype=np.uint16,
    )
    samples.setflags(write=False)
    batch = RecordBatch(
        samples=samples,
        layout=layout,
        info=CaptureInfo(),
        first_record=0,
        records_per_frame=plan.records_per_frame,
    )
    reconstruct = partial(
        _reconstruct_frame,
        batch,
        plan=plan,
        reconstructor=reconstructor,
        sample_code_range=sample_code_range,
    )

    start_ns = perf_counter_ns()
    frame = reconstruct(use_numba=use_numba)
    first_ms = (perf_counter_ns() - start_ns) / 1e6
    frame_shape = frame.shape
    if use_numba:
        np.testing.assert_array_equal(frame, reconstruct())
    del frame
    for _ in range(4):
        reconstruct(use_numba=use_numba)

    elapsed_ns: list[int] = []
    for _ in range(args.iterations):
        start_ns = perf_counter_ns()
        frame = reconstruct(use_numba=use_numba)
        elapsed_ns.append(perf_counter_ns() - start_ns)
        del frame

    nominal_fps = float(plan.nominal_frame_rate)
    period_ms = 1000 / nominal_fps
    elapsed_ms = np.asarray(elapsed_ns, dtype=np.float64) / 1e6
    median_ms = float(np.median(elapsed_ms))
    p95_ms = float(np.percentile(elapsed_ms, 95))
    processing_fps = args.iterations * 1e9 / sum(elapsed_ns)

    print(
        f"ADC {float(plan.sample_rate) / 1e6:g} MS/s, "
        f"scanner {float(plan.nominal_frequency) / 1e3:g} kHz"
    )
    print(
        f"Raw batch {layout.shape} ({samples.nbytes / 2**20:.2f} MiB), "
        f"{plan.image_records} imaging + {plan.blank_records} flyback records"
    )
    print(
        f"Output frame {frame_shape}, nominal {nominal_fps:.2f} frames/s "
        f"({period_ms:.2f} ms per frame)"
    )
    print(f"Backend {backend_label}")
    if use_numba:
        print(f"First call {first_ms:.2f} ms (includes any JIT compile or cache load)")
    else:
        print(f"First call {first_ms:.2f} ms")
    print(
        f"{args.iterations} timed calls, median {median_ms:.2f} ms, p95 {p95_ms:.2f} ms"
    )
    print(
        f"Processing capacity {processing_fps:.2f} frames/s, "
        f"p95 uses {p95_ms / period_ms:.1%} of the nominal frame period"
    )


if __name__ == "__main__":
    main()
