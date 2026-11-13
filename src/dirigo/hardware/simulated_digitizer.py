"""Finite, unpaced integer capture for exercising acquisition data flow."""

from dataclasses import dataclass, field

import numpy as np

from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.capture_session import DigitizerCaptureSession
from dirigo.hardware.digitizer import Digitizer


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulatedDigitizerConfig:
    """Declare the synthetic device's available input channels."""

    channel_count: int = field(
        default=2,
        metadata={"description": "Number of synthetic input channels."},
    )

    def __post_init__(self) -> None:
        """Require a positive integer channel count."""
        if not isinstance(self.channel_count, int) or isinstance(
            self.channel_count, bool
        ):
            raise TypeError("channel_count must be an integer.")
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive.")


class SimulatedDigitizer(Digitizer):
    """Provide deterministic int16 records without elapsed-time pacing.

    Reads generate complete batches immediately. sample_rate defines ideal
    sample times; this device does not simulate ADC timing or mirror motion.

    Only finite, internally clocked capture without a start trigger is
    supported.
    """

    config_type = SimulatedDigitizerConfig

    def __init__(self, config: SimulatedDigitizerConfig) -> None:
        """Store capabilities without starting acquisition or reserving resources."""
        if not isinstance(config, SimulatedDigitizerConfig):
            raise TypeError("config must be a SimulatedDigitizerConfig.")

        super().__init__()
        self._config = config

    @property
    def config(self) -> SimulatedDigitizerConfig:
        """Immutable synthetic device configuration."""
        return self._config

    @property
    def channel_count(self) -> int:
        """Number of physical input indices exposed by this device."""
        return self._config.channel_count

    @property
    def bit_depth(self) -> int:
        """Resolution of the generated signed sample codes."""
        return 16

    @property
    def data_range(self) -> units.IntRange:
        """Inclusive range of generated signed int16 sample codes."""
        return units.IntRange(-32768, 32767)

    def _create_capture(self, settings: CaptureSettings) -> DigitizerCaptureSession:
        """Reject unsupported timing modes and construct a finite session."""
        if settings.batch_count is None:
            raise NotImplementedError("Simulation requires finite batch_count.")
        if settings.external_clock is not None:
            raise NotImplementedError("Simulation does not support external clocks.")
        if settings.start_trigger is not None:
            raise NotImplementedError("Simulation does not support start triggers.")
        if settings.record_trigger is not None:
            raise NotImplementedError("Simulation does not support record triggers.")

        return _SimulatedCaptureSession(self, settings)

    def _close(self) -> None:
        """Release no external resources."""
        return

    def _snapshot_settings(self) -> dict[str, object]:
        """Identify synthetic storage and execution behavior in saved metadata."""
        return {
            "channel_count": self.channel_count,
            "bit_depth": self.bit_depth,
            "sample_dtype": "int16",
            "simulation_mode": "unpaced",
        }


class _SimulatedCaptureSession(DigitizerCaptureSession):
    """Generate consecutive sample windows from a validated simulator request."""

    def __init__(
        self, digitizer: SimulatedDigitizer, settings: CaptureSettings
    ) -> None:
        super().__init__(
            digitizer,
            batch_count=settings.batch_count,
            timestamps_enabled=settings.timestamps_enabled,
        )
        self._settings = settings

    def _prepare_capture(self) -> CaptureLayout:
        """Describe complete batches without allocating SDK transfer resources."""
        settings = self._settings
        return CaptureLayout(
            shape=(
                settings.records_per_batch,
                settings.record_length,
                len(settings.channel_indices),
            ),
            channel_indices=settings.channel_indices,
            dtype=np.dtype(np.int16),
        )

    def _start_capture(self) -> None:
        """Make synthetic data available without starting a background thread."""
        return

    def _read_into(
        self, destination: SampleArray, *, timeout: float | None
    ) -> CaptureInfo:
        """Fill deterministic codes and optional ideal record timestamps.

        For session-wide sample index n and physical channel c, the code is
        ((n + 257*c) % 65536) - 32768. Each new session restarts n at zero.

        Data is immediately available even when timeout is zero.
        """
        settings = self._settings
        records = settings.records_per_batch
        length = settings.record_length
        first_record = self.batches_acquired * records
        first_sample = (first_record * length) % 65536

        offsets = np.arange(records * length, dtype=np.int64).reshape(
            records, length, 1
        )
        channels = np.array(
            [index % 65536 for index in settings.channel_indices],
            dtype=np.int64,
        )
        codes = ((first_sample + offsets + 257 * channels) % 65536) - 32768
        np.copyto(destination, codes, casting="unsafe")

        if not self.timestamps_enabled:
            return CaptureInfo()

        with np.errstate(over="raise", invalid="raise"):
            times = (first_record + np.arange(records, dtype=np.float64)) * (
                length / float(settings.sample_rate)
            )
        return CaptureInfo(times)

    def _stop_capture(self) -> None:
        """Stop no background work."""
        return

    def _close_capture(self) -> None:
        """Release no persistent per-run resources."""
        return
