"""Simulated devices for exercising hardware APIs without physical equipment."""

import math
import time
from collections.abc import Mapping

from dirigo import units
from dirigo.hardware.output import GalvoOutputSession
from dirigo.hardware.scanner import (
    GalvoScanner,
    GalvoScannerConfig,
    ResonantScanner,
    ResonantScannerConfig,
)
from dirigo.hardware.timing import ExternalSampleClock
from dirigo.hardware.waveform import AngularWaveform


class SimulatedResonantScanner(ResonantScanner):
    """Accept amplitude commands without hardware I/O or simulated motion.

    The inherited amplitude property records successful commands. No physical
    motion, settling delay, or waveform generation is simulated.
    """

    config_type = ResonantScannerConfig

    def _write_amplitude(self, value: units.Angle) -> None:
        """Accept the command; the base class records its value."""
        return

    def _close(self) -> None:
        """Complete closure without owning any external resources."""
        return


class SimulatedGalvoScanner(GalvoScanner):
    """Record static galvo commands without hardware or motion simulation."""

    config_type = GalvoScannerConfig

    def _write_angle(self, value: units.Angle) -> None:
        """Accept a validated static angle without producing output."""
        return

    def _close(self) -> None:
        """Release no resources."""
        return


class SimulatedGalvoOutputSession(GalvoOutputSession):
    """Simulate one finite run using elapsed monotonic time.

    Supports simulated galvo axes with internal timing only. External sample
    clocks are rejected because no incoming-edge simulation is provided.
    No background thread is created.

    wait() observes completion after sample_count / sample_rate seconds;
    it does not simulate mirror motion or update static-angle history.
    """

    def __init__(
        self,
        waveforms: Mapping[GalvoScanner, AngularWaveform],
        *,
        external_clock: ExternalSampleClock | None = None,
    ) -> None:
        """Require simulated axes and reject unsupported external timing."""
        super().__init__(waveforms, external_clock=external_clock)
        if any(not isinstance(axis, SimulatedGalvoScanner) for axis in self._waveforms):
            raise TypeError("Simulation requires SimulatedGalvoScanner axes.")
        if self.external_clock is not None:
            raise NotImplementedError(
                "Elapsed-time simulation does not support external sample clocks."
            )
        self._deadline: float | None = None

    def _prepare_output(self) -> None:
        """Allocate no hardware resources."""
        return

    def _start_output(self) -> None:
        """Set the nominal completion time for this finite run."""
        deadline = time.monotonic() + (self._sample_count / float(self._sample_rate))
        if not math.isfinite(deadline):
            raise ValueError("Simulated run duration is too large.")
        self._deadline = deadline

    def _wait_output(self, timeout: float | None) -> bool:
        """Wait for the deadline, bounded by the requested timeout."""
        assert self._deadline is not None
        remaining = max(0.0, self._deadline - time.monotonic())
        delay = remaining if timeout is None else min(remaining, timeout)
        if delay > 0:
            time.sleep(delay)
        return time.monotonic() >= self._deadline

    def _stop_output(self) -> None:
        """Stop tracking output without commanding an angle."""
        return

    def _close_output(self) -> None:
        """Discard the simulated timing resource."""
        self._deadline = None
