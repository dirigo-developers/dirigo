"""Test resonant amplitude control and the simulated driver's loading path."""

from pathlib import Path

import pytest

from dirigo import units
from dirigo.hardware.config import load_hardware
from dirigo.hardware.scanner import ResonantScanner, ResonantScannerConfig
from dirigo.hardware.simulated import SimulatedResonantScanner


class RecordingScanner(SimulatedResonantScanner):
    """Record driver calls and optionally fail amplitude output or cleanup."""

    def __init__(self, config: ResonantScannerConfig) -> None:
        super().__init__(config)
        self.commands: list[units.Angle] = []
        self.close_calls = 0
        self.write_failure: Exception | None = None
        self.close_failure: Exception | None = None

    def _write_amplitude(self, value: units.Angle) -> None:
        if self.write_failure is not None:
            raise self.write_failure
        self.commands.append(value)

    def _close(self) -> None:
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


@pytest.fixture
def config() -> ResonantScannerConfig:
    """Provide physical parameters without selecting a control backend."""
    return ResonantScannerConfig(
        nominal_frequency=units.Frequency("7.91 kHz"),
        max_amplitude=units.Angle("26 deg"),
        settling_time=units.Time("100 ms"),
        frequency_tolerance=0.003,
    )


@pytest.fixture
def scanner(config: ResonantScannerConfig) -> RecordingScanner:
    """Provide a scanner whose protected driver calls can be observed."""
    return RecordingScanner(config)


def test_construction_does_not_command_amplitude(
    scanner: RecordingScanner,
    config: ResonantScannerConfig,
) -> None:
    """Keep the initial amplitude unknown without touching the output."""
    assert scanner.config is config
    assert scanner.amplitude is None
    assert scanner.commands == []
    assert scanner.close_calls == 0


def test_config_property_is_read_only(
    scanner: RecordingScanner,
    config: ResonantScannerConfig,
) -> None:
    """Prevent replacement of the physical configuration through the property."""
    with pytest.raises(AttributeError):
        scanner.config = config  # ty: ignore[invalid-assignment]


def test_constructor_requires_scanner_configuration() -> None:
    """Reject an inappropriate initialization object."""
    with pytest.raises(TypeError, match="config"):
        SimulatedResonantScanner(object())  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("value", ["0 deg", "5 deg", "26 deg"])
def test_applies_amplitude_including_both_bounds(
    scanner: RecordingScanner,
    value: str,
) -> None:
    """Accept zero, intermediate, and maximum peak-to-peak optical amplitudes."""
    amplitude = units.Angle(value)
    scanner.amplitude = amplitude

    assert scanner.amplitude == amplitude
    assert scanner.commands == [amplitude]


@pytest.mark.parametrize("value", [None, True, 1.0, "1 deg", units.Time("1 s")])
def test_rejects_untyped_or_incompatible_amplitudes(
    scanner: RecordingScanner,
    value: object,
) -> None:
    """Reject inappropriate types without sending a driver command."""
    with pytest.raises(TypeError, match="amplitude"):
        scanner.amplitude = value  # ty: ignore[invalid-assignment]

    assert scanner.amplitude is None
    assert scanner.commands == []


@pytest.mark.parametrize(
    "value",
    ["-1 deg", "27 deg", float("nan"), float("inf"), float("-inf")],
)
def test_invalid_values_preserve_the_previous_command(
    scanner: RecordingScanner,
    value: str | float,
) -> None:
    """Reject nonfinite or out-of-range values before accessing the driver."""
    previous = units.Angle("3 deg")
    scanner.amplitude = previous

    with pytest.raises(ValueError, match="amplitude"):
        scanner.amplitude = units.Angle(value)

    assert scanner.amplitude == previous
    assert scanner.commands == [previous]


@pytest.mark.parametrize("initial", [None, "3 deg"])
def test_driver_failure_preserves_command_history(
    scanner: RecordingScanner,
    initial: str | None,
) -> None:
    """Keep unknown or previously commanded amplitude when output fails."""
    previous = None if initial is None else units.Angle(initial)
    if previous is not None:
        scanner.amplitude = previous

    commands = scanner.commands.copy()
    failure = RuntimeError("Output failed.")
    scanner.write_failure = failure

    with pytest.raises(RuntimeError) as caught:
        scanner.amplitude = units.Angle("6 deg")

    assert caught.value is failure
    assert scanner.amplitude == previous
    assert scanner.commands == commands


@pytest.mark.parametrize("initial", [None, "3 deg"])
def test_close_is_idempotent_and_does_not_command_amplitude(
    scanner: RecordingScanner,
    config: ResonantScannerConfig,
    initial: str | None,
) -> None:
    """Retain cached information while permanently disabling amplitude commands."""
    previous = None if initial is None else units.Angle(initial)
    if previous is not None:
        scanner.amplitude = previous
    commands = scanner.commands.copy()

    scanner.close()
    scanner.close()

    assert scanner.close_calls == 1
    assert scanner.commands == commands
    assert scanner.amplitude == previous
    assert scanner.config is config

    with pytest.raises(RuntimeError, match="closed"):
        scanner.amplitude = units.Angle("4 deg")

    assert scanner.commands == commands


def test_cleanup_failure_still_permanently_closes_scanner(
    scanner: RecordingScanner,
) -> None:
    """Propagate cleanup failure without allowing retries or new commands."""
    failure = RuntimeError("Cleanup failed.")
    scanner.close_failure = failure

    with pytest.raises(RuntimeError) as caught:
        scanner.close()

    assert caught.value is failure
    scanner.close()
    assert scanner.close_calls == 1

    with pytest.raises(RuntimeError, match="closed"):
        scanner.amplitude = units.Angle("4 deg")


def test_loads_simulated_scanner_and_owns_its_lifetime(tmp_path: Path) -> None:
    """Load typed physical parameters and close the scanner with its collection."""
    path = tmp_path / "system_config.toml"
    path.write_text(
        """
        [devices.fast_scanner]
        driver = "simulated_resonant"

        [devices.fast_scanner.config]
        nominal_frequency = "7.91 kHz"
        max_amplitude = "26 deg"
        settling_time = "100 ms"
        frequency_tolerance = 0.003
        """,
        encoding="utf-8",
    )
    hardware = load_hardware(
        path,
        drivers={"simulated_resonant": SimulatedResonantScanner},
    )

    try:
        scanner = hardware.require("fast_scanner", ResonantScanner)
        assert isinstance(scanner, SimulatedResonantScanner)
        assert scanner.amplitude is None
        assert scanner.config.nominal_frequency == units.Frequency("7910 Hz")

        scanner.amplitude = units.Angle("5 deg")
        assert scanner.amplitude == units.Angle("5 deg")
    finally:
        hardware.close()

    with pytest.raises(RuntimeError, match="closed"):
        scanner.amplitude = units.Angle("6 deg")
