"""Run real TiffWriter capacity and bounded-buffer acquisition checks.

Example: python -m dirigo.benchmarks.tiff --directory D:/bench --width 1024
    --height 1024 --channels 2 --interpretation multichannel --fps 30 --contiguous
"""

import argparse
import os
from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import BinaryIO, Literal

import numpy as np
import tifffile

from dirigo.frames import FrameArray, FrameMetadata
from dirigo.pipeline.exceptions import EndOfStream, OperationCancelled
from dirigo.pipeline.pipeline import Pipeline
from dirigo.pipeline.pool import Lease
from dirigo.pipeline.product import Product
from dirigo.pipeline.worker import Worker
from dirigo.sinks.tiff import TiffWriter

from ._tiff_config import TiffBenchmarkConfig, due_slot

type BenchmarkMode = Literal["capacity", "paced"]


@dataclass(frozen=True, slots=True)
class TiffBenchmarkResult:
    """Measured results for one run; readback is outside all timing windows."""

    mode: BenchmarkMode
    frames: int
    pixel_bytes: int
    file_bytes: int
    streaming_seconds: float
    finalization_seconds: float
    sync_seconds: float
    total_seconds: float
    overruns: int
    scheduling_misses: int
    max_lateness_seconds: float
    peak_buffers: int

    @property
    def fps(self) -> float:
        """Overall rate, including drain, finalization, close, and disk sync."""
        return self.frames / self.total_seconds

    @property
    def mib_per_second(self) -> float:
        """Overall pixel throughput; file overhead is excluded from the numerator."""
        return self.pixel_bytes / self.total_seconds / 2**20

    @property
    def acquisition_status(self) -> str:
        """Scheduling failures prevent an unqualified acquisition pass."""
        if self.mode == "capacity":
            return "not applicable"
        if self.overruns:
            return "overrun"
        if self.scheduling_misses:
            return "inconclusive: source missed frame intervals"
        return "kept up for the tested duration"


class _FrameSource(Worker[None, FrameArray]):
    """Reuse initialized storage, adding only two pixel markers per frame."""

    def __init__(self, config: TiffBenchmarkConfig, mode: BenchmarkMode) -> None:
        super().__init__()
        self.config = config
        self.mode = mode
        self.started_at = 0.0
        self.overruns = 0
        self.scheduling_misses = 0
        self.max_lateness = 0.0
        self.peak_buffers = 0
        self.indices: list[int] = []

    def _prepare(self) -> None:
        self._init_pool(
            capacity=self.config.buffers,
            factory=lambda: np.full(self.config.shape, 7, dtype=self.config.dtype),
        )

    @contextmanager
    def _borrow(self, timeout: float | None) -> Generator[Lease[FrameArray]]:
        """Sample occupancy atomically with acquisition, using the pool's lock.

        This benchmark alone inspects pool internals. Holding the condition
        through context entry prevents a concurrent return from hiding the peak.
        The condition is released before filling or publishing a frame.
        """
        pool = self._pool
        assert pool is not None
        with ExitStack() as resources:
            with pool._condition:
                lease = resources.enter_context(self._acquire(timeout=timeout))
                occupied = self.config.buffers - len(pool._available)
                self.peak_buffers = max(self.peak_buffers, occupied)
            yield lease

    def _wait_until(self, deadline: float) -> None:
        if self._abort_requested.wait(max(0.0, deadline - monotonic())):
            raise OperationCancelled("Benchmark source was aborted.")

    def _send_frame(self, index: int, timeout: float | None) -> None:
        with self._borrow(timeout) as lease:
            marker = index % (256 if self.config.dtype == "uint8" else 65536)
            lease.payload.flat[0] = marker
            lease.payload.flat[-1] = marker
            lease.metadata = FrameMetadata(sequence_index=index, time_index=index)
            # Queue capacity matches the pool. Nonblocking publication cannot
            # silently change the paced acquisition rate.
            self._publish(lease, timeout=0.0)
            self.indices.append(index)

    def _work(self) -> None:
        config = self.config
        self.started_at = monotonic()
        end = self.started_at + config.duration
        index = 0
        while not self._abort_requested.is_set():
            if self.mode == "paced":
                if index >= config.scheduled_frames:
                    self._wait_until(end)
                    return
                self._wait_until(self.started_at + index / config.fps)
                now = monotonic()
                if now >= end:
                    self.scheduling_misses += config.scheduled_frames - index
                    return
                index, skipped = due_slot(
                    index, now - self.started_at, config.fps,
                    config.scheduled_frames,
                )
                self.scheduling_misses += skipped
                if index >= config.scheduled_frames:
                    self._wait_until(end)
                    return
                self.max_lateness = max(
                    self.max_lateness,
                    now - (self.started_at + index / config.fps),
                )
                timeout = 0.0
            else:
                timeout = end - monotonic()
                if timeout <= 0:
                    return
            try:
                self._send_frame(index, timeout)
            except TimeoutError:
                if self.mode == "capacity":
                    return
                self.overruns += 1
            index += 1

    def _close(self) -> None:
        pass


class _MeasuredWriter(TiffWriter):
    """Measure the existing writer and sync after its encoder and file close."""

    def __init__(self, source: _FrameSource, path: Path) -> None:
        config = source.config
        super().__init__(
            upstream=source, path=path, interpretation=config.interpretation,
            contiguous=config.contiguous,
        )
        # Replace this worker's default capacity-one subscription before launch.
        assert self._subscription is not None
        self._subscription.close()
        self._subscription = source._publisher.subscribe(capacity=config.buffers)
        self.frames = 0
        self.last_write_at: float | None = None
        self.finalization_started_at: float | None = None
        self.finalization_seconds = 0.0
        self.sync_seconds = 0.0
        self._sync_fd: int | None = None

    def _open_file(self) -> BinaryIO:
        output = super()._open_file()
        try:
            self._sync_fd = os.dup(output.fileno())
        except BaseException:
            output.close()
            raise
        return output

    def _write_frame(
        self, tif: tifffile.TiffWriter, product: Product[FrameArray], *, first: bool
    ) -> None:
        super()._write_frame(tif, product, first=first)
        self.frames += 1
        self.last_write_at = monotonic()

    def _work(self) -> None:
        try:
            super()._work()
        finally:
            if self.finalization_started_at is not None:
                self.finalization_seconds = monotonic() - self.finalization_started_at

    @contextmanager
    def _receive(
        self, timeout: float | None = None
    ) -> Generator[Product[FrameArray]]:
        try:
            with super()._receive(timeout=timeout) as product:
                yield product
        except EndOfStream:
            self.finalization_started_at = monotonic()
            raise

    def _close(self) -> None:
        # TiffWriter._work has flushed and closed the Python file and encoder.
        # The duplicate keeps the same file open until final synchronization.
        try:
            super()._close()
        finally:
            fd, self._sync_fd = self._sync_fd, None
            if fd is not None:
                try:
                    started = monotonic()
                    os.fsync(fd)
                    self.sync_seconds = monotonic() - started
                finally:
                    os.close(fd)


def _verify(path: Path, source: _FrameSource, frames: int) -> None:
    """Check page counts, all identifiers, and selected pixels without stacking."""
    if frames != len(source.indices):
        raise ValueError("Published and saved frame counts differ.")
    if not frames:
        if path.exists():
            raise ValueError("An empty recording unexpectedly created output.")
        return
    config = source.config
    if source.mode == "paced" and (
        frames + source.overruns + source.scheduling_misses != config.scheduled_frames
    ):
        raise ValueError("Not all scheduled acquisition slots were accounted for.")
    pages_per_frame = config.channels if config.interpretation == "multichannel" else 1
    with tifffile.TiffFile(path) as tif:
        if len(tif.pages) != frames * pages_per_frame:
            raise ValueError("Incorrect TIFF page count.")
        saved = tif.shaped_metadata
        if saved is None:
            raise ValueError("TIFF metadata was not finalized.")
        if config.contiguous:
            if len(saved) != 1 or len(tif.series) != 1:
                raise ValueError("Expected one contiguous series.")
            recording = saved[0]["dirigo"]
            if recording["complete"] is not True:
                raise ValueError("Recording did not finish normally.")
            records = recording["frames"]
            stored_shape = (
                (frames, config.channels, config.height, config.width)
                if config.interpretation == "multichannel"
                else (frames, *config.shape)
            )
            if tuple(saved[0]["shape"]) != stored_shape:
                raise ValueError("Incorrect contiguous recording shape.")
        else:
            if len(saved) != frames or len(tif.series) != frames:
                raise ValueError("Expected one series per frame.")
            records = [entry["dirigo"] for entry in saved]
        if len(records) != frames:
            raise ValueError("Incorrect metadata record count.")
        for record, index in zip(records, source.indices, strict=True):
            expected = {"sequence_index": index, "time_index": index, "z_index": None}
            if record != expected:
                raise ValueError("Incorrect saved frame metadata.")
        for position in sorted({0, frames // 2, frames - 1}):
            marker = source.indices[position] % (
                256 if config.dtype == "uint8" else 65536
            )
            for channel in sorted({0, pages_per_frame - 1}):
                data = tif.pages[position * pages_per_frame + channel].asarray()
                expected = np.full(data.shape, 7, dtype=config.dtype)
                if channel == 0:
                    expected.flat[0] = marker
                if channel == pages_per_frame - 1:
                    expected.flat[-1] = marker
                expected_shape = (
                    config.shape if config.interpretation == "rgb"
                    else (config.height, config.width)
                )
                if data.shape != expected_shape or data.dtype != np.dtype(config.dtype):
                    raise ValueError("Incorrect saved page layout.")
                np.testing.assert_array_equal(data, expected)


def run_benchmark(
    config: TiffBenchmarkConfig,
    *,
    directory: Path,
    mode: BenchmarkMode,
) -> TiffBenchmarkResult:
    """Run one mode on the selected filesystem and remove only its own output.

    Preparation and readback are excluded. Timing starts when the source begins
    producing, and ends after worker cleanup, including final disk sync. A paced
    run drains all successfully published frames even after simulated overruns.
    Writer, verification, synchronization, and cleanup errors propagate.
    """
    if mode not in ("capacity", "paced"):
        raise ValueError("mode must be capacity or paced.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dirigo-tiff-", dir=directory) as temporary:
        path = Path(temporary) / "recording.tif"
        source = _FrameSource(config, mode)
        writer = _MeasuredWriter(source, path)
        pipeline = Pipeline(writer)
        try:
            pipeline.start()
            pipeline.wait()
        except BaseException as exc:
            if not pipeline.done:
                pipeline.abort()
                try:
                    pipeline.wait()
                except BaseException as cleanup:
                    if cleanup is not exc:
                        exc.add_note(f"Benchmark cleanup also failed: {cleanup!r}")
            raise
        ended = monotonic()
        _verify(path, source, writer.frames)
        return TiffBenchmarkResult(
            mode=mode, frames=writer.frames,
            pixel_bytes=writer.frames * config.frame_bytes,
            file_bytes=path.stat().st_size if path.exists() else 0,
            streaming_seconds=(
                writer.finalization_started_at - source.started_at
                if writer.finalization_started_at is not None else 0.0
            ),
            finalization_seconds=writer.finalization_seconds,
            sync_seconds=writer.sync_seconds,
            total_seconds=ended - source.started_at,
            overruns=source.overruns, scheduling_misses=source.scheduling_misses,
            max_lateness_seconds=source.max_lateness, peak_buffers=source.peak_buffers,
        )


def format_result(result: TiffBenchmarkResult, config: TiffBenchmarkConfig) -> str:
    """Format measurements without imposing a universal safety margin."""
    ratio = result.fps / config.fps
    lines = [
        f"{result.mode}: {result.frames} frames; {result.acquisition_status}",
        f"  Overall: {result.fps:.2f} frames/s, {result.mib_per_second:.2f} MiB/s",
        f"  Streaming: {result.streaming_seconds:.3f} s; "
        f"finalization/close: {result.finalization_seconds:.3f} s; "
        f"sync: {result.sync_seconds:.3f} s; total: {result.total_seconds:.3f} s",
        f"  Pixel data: {result.pixel_bytes / 2**20:.2f} MiB; "
        f"file: {result.file_bytes / 2**20:.2f} MiB",
        f"  Peak buffers: {result.peak_buffers}/{config.buffers}; "
        f"overruns: {result.overruns}; scheduling misses: {result.scheduling_misses}",
        "  Maximum serviced-frame lateness: "
        f"{result.max_lateness_seconds * 1000:.3f} ms",
    ]
    if result.mode == "capacity":
        lines.insert(
            2, f"  Capacity / requested rate: {ratio:.2f}x (includes final sync)"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the workload before recording; return nonzero on failed headroom."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--dtype", choices=("uint8", "uint16"), default="uint16")
    parser.add_argument(
        "--interpretation", choices=("monochrome", "multichannel", "rgb"),
        default="monochrome",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--buffers", type=int, default=16)
    parser.add_argument("--contiguous", action="store_true")
    parser.add_argument("--mode", choices=("both", "capacity", "paced"), default="both")
    args = parser.parse_args(argv)
    try:
        config = TiffBenchmarkConfig(
            width=args.width, height=args.height, channels=args.channels,
            dtype=args.dtype, interpretation=args.interpretation, fps=args.fps,
            duration=args.duration, buffers=args.buffers, contiguous=args.contiguous,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Input: {config.shape}, {config.dtype}, {config.fps:g} frames/s; "
        f"required: {config.required_bytes_per_second / 2**20:.2f} MiB/s\n"
        f"Pool: {config.frame_bytes * config.buffers / 2**20:.2f} MiB; "
        f"contiguous={config.contiguous}\n"
        f"Each run produces for {config.duration:g} s, then drains and syncs.\n"
        f"Paced pixel estimate: "
        f"{config.frame_bytes * config.scheduled_frames / 2**30:.3f} GiB.\n"
        "Capacity output size depends on measured speed. "
        "Temporary recordings are removed after each run.",
        flush=True,
    )
    modes: tuple[BenchmarkMode, ...] = (
        ("capacity", "paced") if args.mode == "both" else (args.mode,)
    )
    success = True
    try:
        for mode in modes:
            result = run_benchmark(config, directory=args.directory, mode=mode)
            print(format_result(result, config), flush=True)
            success &= result.frames > 0
            if mode == "capacity":
                success &= result.fps >= config.fps
            else:
                success &= not (result.overruns or result.scheduling_misses)
    except KeyboardInterrupt:
        print("Benchmark interrupted; workers joined and temporary output removed.")
        return 130
    print(
        "Results describe this configuration and duration. Longer runs help expose "
        "cache effects; final sync alone does not establish indefinite throughput."
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
