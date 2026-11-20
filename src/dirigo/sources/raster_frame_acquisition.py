"""Acquire raw raster batches using a specification and configured System."""

from time import monotonic
from typing import Never

import numpy as np

from dirigo import units
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.raster_frame_sync import (
    RasterFrameSync,
    RasterFrameSyncSession,
)
from dirigo.hardware.scanner import BeamScanner, GalvoScanner, ResonantScanner
from dirigo.pipeline.product import Metadata, Product
from dirigo.pipeline.worker import Worker
from dirigo.raster import (
    RasterFrameAcquisitionSpec,
    resolve_raster_frame_geometry,
)
from dirigo.records import RecordBatch
from dirigo.resonant_raster import (
    ResonantRasterPlan,
    ResonantRasterSettings,
    plan_resonant_raster_frame,
)
from dirigo.system import System


class RasterFrameAcquisition(Worker[Never, RecordBatch]):
    """Publish raw record batches from a specification-driven raster run.

    The initial supported arrangement is a resonant fast scanner and a galvo
    slow scanner. The System supplies devices, optics, synchronization, and
    timing settings.

    Construction stores dependencies. Preparation resolves constraints and
    configures participants, including static scanner commands. Execution
    waits for declared settling before starting synchronized acquisition.

    Each published batch owns an independent read-only sample array and
    contains one logical frame including blank records. The immutable plan is
    available after preparation for downstream reconstruction.
    """

    def __init__(
        self,
        spec: RasterFrameAcquisitionSpec,
        *,
        system: System,
        name: str | None = None,
    ) -> None:
        """Retain the request and borrowed system without hardware I/O."""
        if not isinstance(spec, RasterFrameAcquisitionSpec):
            raise TypeError("spec must be a RasterFrameAcquisitionSpec.")

        if not isinstance(system, System):
            raise TypeError("system must be a System.")

        super().__init__(name=name)
        self._spec = spec
        self._system = system

        self._capture: DigitizerCaptureSession | None = None
        self._sync: RasterFrameSyncSession | None = None
        self._fast_scanner: ResonantScanner | None = None
        self._settings: ResonantRasterSettings | None = None
        self._plan: ResonantRasterPlan | None = None
        self._prepared = False
        self._settled_at = 0.0
        self._fast_command_attempted = False

    @property
    def spec(self) -> RasterFrameAcquisitionSpec:
        """Original immutable acquisition request."""
        return self._spec

    @property
    def plan(self) -> ResonantRasterPlan:
        """Prepared immutable plan, retained after acquisition cleanup."""
        plan = self._plan

        if not self._prepared or plan is None:
            raise RuntimeError("The raster plan requires successful preparation.")

        return plan

    def _prepare(self) -> None:
        """Plan and prepare the run, retaining ownership after partial failure."""
        system = self._system
        settings = system.require_resonant_raster()
        optics = system.require_optics()

        fast = system.require_device("fast_scanner", BeamScanner)
        if not isinstance(fast, ResonantScanner):
            raise NotImplementedError(
                "RasterFrameAcquisition currently requires a resonant fast scanner."
            )

        slow = system.require_device("slow_scanner", GalvoScanner)
        digitizer = system.require_device("digitizer", Digitizer)
        backend = system.require_sync(
            "raster_frame_sync",
            RasterFrameSync,
        )

        self._settings = settings
        self._fast_scanner = fast

        capture = digitizer.create_capture()
        self._capture = capture

        sync = backend.create_session(
            fast_scanner=fast,
            slow_scanner=slow,
            capture=capture,
        )
        self._sync = sync

        geometry = resolve_raster_frame_geometry(
            self._spec,
            optics,
            row_multiple=2 if self._spec.bidirectional else 1,
            max_pixel_size_adjustment=settings.max_pixel_size_adjustment,
        )
        minimum, step = capture.resolve_record_length_constraints()

        plan = plan_resonant_raster_frame(
            geometry,
            fast.config,
            sample_rate=capture.requested_sample_rate,
            max_record_rate=sync.max_record_rate,
            record_length_minimum=minimum,
            record_length_step=step,
            line_fill_fraction=settings.line_fill_fraction,
            trigger_phase=settings.trigger_phase,
            capture_rearm_time=settings.capture_rearm_time,
            slow_sample_rate=settings.slow_sample_rate,
            slow_flyback_time=settings.slow_flyback_time,
            output_rearm_time=settings.output_rearm_time,
        )

        capture.configure(
            record_length=plan.record_length,
            records_per_batch=plan.records_per_batch,
            batch_count=plan.batch_count,
        )
        sync.configure(
            records_per_frame=plan.records_per_frame,
            frame_count=plan.batch_count,
            waveforms={slow: plan.slow_waveform},
        )

        # Static commands precede buffered output reservations.
        slow.angle = units.Angle(float(plan.slow_waveform.samples[0]))
        slow_settled_at = monotonic() + float(settings.slow_settling_time)

        self._fast_command_attempted = True
        fast.amplitude = plan.fast_amplitude
        fast_settled_at = monotonic() + float(fast.config.settling_time)

        self._settled_at = max(
            slow_settled_at,
            fast_settled_at,
        )
        sync.prepare()

        layout = capture.layout
        if (
            layout.records_per_batch != plan.records_per_batch
            or layout.record_length != plan.record_length
        ):
            raise ValueError("Prepared capture layout differs from the raster plan.")

        self._plan = plan
        self._prepared = True

    def _work(self) -> None:
        """Wait for settling, start synchronization, and publish capture data."""
        capture = self._capture
        sync = self._sync
        settings = self._settings
        assert capture is not None
        assert sync is not None
        assert settings is not None
        plan = self.plan

        remaining = max(
            0.0,
            self._settled_at - monotonic(),
        )
        if self._abort_requested.wait(remaining):
            return

        sync.start()
        layout = capture.layout
        batch_index = 0
        batch_count = plan.batch_count

        while (
            batch_count is None or batch_index < batch_count
        ) and not self._abort_requested.is_set():
            samples = np.empty(
                layout.shape,
                dtype=layout.dtype,
            )

            while not self._abort_requested.is_set():
                try:
                    info = capture.read_into(
                        samples,
                        timeout=settings.read_timeout,
                    )
                except TimeoutError:
                    if capture.state is CaptureState.CLOSED:
                        raise
                    continue

                if self._abort_requested.is_set():
                    return

                if info is None:
                    if batch_count is None:
                        raise RuntimeError("Unlimited capture ended unexpectedly.")

                    raise RuntimeError(
                        "Capture ended before the requested frame count."
                    )

                samples.setflags(write=False)
                self._publish(
                    Product(
                        RecordBatch(
                            samples=samples,
                            layout=layout,
                            info=info,
                            first_record=(batch_index * layout.records_per_batch),
                            records_per_frame=plan.records_per_frame,
                        ),
                        metadata=Metadata(
                            sequence_index=batch_index,
                        ),
                    )
                )
                batch_index += 1
                break

    def _close(self) -> None:
        """Close synchronization and capture, then apply scanner idle policy."""
        sync, self._sync = self._sync, None
        capture, self._capture = self._capture, None
        fast, self._fast_scanner = self._fast_scanner, None
        commanded, self._fast_command_attempted = (
            self._fast_command_attempted,
            False,
        )
        settings = self._settings
        failures: list[BaseException] = []

        for resource in (sync, capture):
            if resource is None:
                continue

            try:
                resource.close()
            except BaseException as exc:
                failures.append(exc)

        if (
            commanded
            and fast is not None
            and settings is not None
            and settings.stop_fast_scanner_on_close
        ):
            try:
                fast.amplitude = units.Angle(0)
            except BaseException as exc:
                failures.append(exc)

        if failures:
            raise BaseExceptionGroup(
                "Raster acquisition cleanup failed.",
                failures,
            )
