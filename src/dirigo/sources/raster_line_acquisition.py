"""Finite repeated-line acquisition using the shared raster execution loop."""

from time import monotonic

from dirigo import units
from dirigo.hardware.capture_sync import CaptureOnlySyncSession
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.scanner import BeamScanner, ResonantScanner
from dirigo.raster import RasterLineAcquisitionSpec
from dirigo.raster_planning import plan_raster_line
from dirigo.resonant_raster import ResonantLinePlan, ResonantScanSettings
from dirigo.sources.raster_acquisition import RasterAcquisition
from dirigo.system import System


class RasterLineAcquisition(RasterAcquisition):
    """Acquire finite repeated resonant lines without slow-axis frame motion."""

    def __init__(
        self,
        spec: RasterLineAcquisitionSpec,
        *,
        system: System,
        name: str | None = None,
    ) -> None:
        """Store the requested line acquisition without accessing hardware."""
        if not isinstance(spec, RasterLineAcquisitionSpec):
            raise TypeError("spec must be a RasterLineAcquisitionSpec.")
        super().__init__(system=system, name=name)
        self._spec = spec
        self._settings: ResonantScanSettings | None = None
        self._fast_scanner: ResonantScanner | None = None
        self._fast_command_attempted = False
        self._plan: ResonantLinePlan | None = None

    @property
    def spec(self) -> RasterLineAcquisitionSpec:
        """Original immutable acquisition request."""
        return self._spec

    @property
    def plan(self) -> ResonantLinePlan:
        """Resolved plan after successful preparation, retained after closure."""
        if not self._prepared or self._plan is None:
            raise RuntimeError("The line plan requires successful preparation.")
        return self._plan

    def _prepare(self) -> None:
        """Plan capture, command the scanner, and prepare capture resources."""
        settings = self._system.require_resonant_scan()
        optics = self._system.require_optics()
        fast = self._system.require_device("fast_scanner", BeamScanner)
        if not isinstance(fast, ResonantScanner):
            raise NotImplementedError("Only resonant line acquisition is implemented.")
        digitizer = self._system.require_device("digitizer", Digitizer)
        self._settings = settings
        self._read_timeout = settings.read_timeout
        self._fast_scanner = fast

        capture = digitizer.create_capture()
        self._capture = capture
        sync = CaptureOnlySyncSession(capture)
        self._sync = sync
        minimum, step = capture.resolve_record_length_constraints()

        plan = plan_raster_line(
            self._spec,
            optics=optics,
            fast_scanner_config=fast.config,
            settings=settings,
            sample_rate=capture.requested_sample_rate,
            record_length_minimum=minimum,
            record_length_step=step,
        )
        capture.configure(
            record_length=plan.fast_axis.record_length,
            records_per_batch=plan.records_per_batch,
            batch_count=plan.batch_count,
        )
        if capture.settings.record_trigger is None:
            raise ValueError("Resonant line acquisition requires a record_trigger.")

        self._fast_command_attempted = True
        fast.amplitude = plan.fast_axis.fast_amplitude
        self._settled_at = monotonic() + float(fast.config.settling_time)

        sync.prepare()
        layout = capture.layout
        if (
            layout.record_length != plan.fast_axis.record_length
            or layout.records_per_batch != plan.records_per_batch
        ):
            raise ValueError("Prepared capture layout does not match the line plan.")

        self._plan = plan
        self._prepared = True

    def _close_scanners(self) -> None:
        """Apply the fast-scanner shutdown policy after capture closure."""
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
