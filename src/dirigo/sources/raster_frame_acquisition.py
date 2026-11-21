"""Raster frame acquisition using explicitly supplied system resources."""

from time import monotonic

from dirigo import units
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.raster_frame_sync import RasterFrameSync
from dirigo.hardware.scanner import BeamScanner, GalvoScanner, ResonantScanner
from dirigo.raster import RasterFrameAcquisitionSpec
from dirigo.raster_planning import plan_raster_frame
from dirigo.resonant_raster import ResonantRasterPlan, ResonantRasterSettings
from dirigo.sources.raster_acquisition import RasterAcquisition
from dirigo.system import System


class RasterFrameAcquisition(RasterAcquisition):
    """Acquire raster frames using a prepared capture and synchronization plan."""

    def __init__(
        self,
        spec: RasterFrameAcquisitionSpec,
        *,
        system: System,
        name: str | None = None,
    ) -> None:
        """Store the specification without accessing hardware."""
        if not isinstance(spec, RasterFrameAcquisitionSpec):
            raise TypeError("spec must be RasterFrameAcquisitionSpec.")

        super().__init__(system=system, name=name)
        self._spec = spec

        self._settings: ResonantRasterSettings | None = None
        self._fast_scanner: ResonantScanner | None = None
        self._plan: ResonantRasterPlan | None = None
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
        self._read_timeout = settings.read_timeout
        self._fast_scanner = fast

        capture = digitizer.create_capture()
        self._capture = capture

        sync = backend.create_session(
            fast_scanner=fast,
            slow_scanner=slow,
            capture=capture,
        )
        self._sync = sync

        minimum, step = capture.resolve_record_length_constraints()

        plan = plan_raster_frame(
            self._spec,
            optics=optics,
            fast_scanner_config=fast.config,
            slow_scanner_config=slow.config,
            settings=settings,
            sample_rate=capture.requested_sample_rate,
            max_record_rate=sync.max_record_rate,
            record_length_minimum=minimum,
            record_length_step=step,
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

        self._capture_layout = layout
        self._sample_code_range = digitizer.data_range
        self._records_per_frame = plan.records_per_frame
        self._plan = plan
        self._prepared = True

    def _close_scanners(self) -> None:
        """Apply the resonant amplitude shutdown policy after session closure."""
        fast, self._fast_scanner = self._fast_scanner, None
        commanded, self._fast_command_attempted = (
            self._fast_command_attempted,
            False,
        )
        settings = self._settings

        if (
            commanded
            and fast is not None
            and settings is not None
            and settings.stop_fast_scanner_on_close
        ):
            fast.amplitude = units.Angle(0)
