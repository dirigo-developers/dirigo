"""Simulated devices for exercising hardware APIs without physical equipment."""

from dirigo import units
from dirigo.hardware.scanner import ResonantScanner, ResonantScannerConfig


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
