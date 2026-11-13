"""Test galvo specifications, static commands, and lifecycle behavior."""

from typing import cast

import pytest

from dirigo import units
from dirigo.hardware.config import parse_config
from dirigo.hardware.scanner import GalvoScanner, GalvoScannerConfig
from dirigo.hardware.simulated import SimulatedGalvoScanner
from dirigo.hardware.waveform import AngularWaveform


class RecordingGalvoScanner(GalvoScanner):
    """Record driver-hook calls and allow injected failures."""

    def __init__(self, config: GalvoScannerConfig) -> None:
        """Store specifications and initialize the hook records."""
        super().__init__(config)
        self.writes: list[units.Angle] = []
        self.close_calls = 0
        self.write_failure: RuntimeError | None = None
        self.close_failure: RuntimeError | None = None

    def _write_angle(self, value: units.Angle) -> None:
        """Record the attempted command, then optionally fail."""
        self.writes.append(value)
        if self.write_failure is not None:
            raise self.write_failure

    def _close(self) -> None:
        """Count cleanup attempts, then optionally fail."""
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


@pytest.fixture
def config() -> GalvoScannerConfig:
    """Provide asymmetric optical limits to avoid assuming a centered range."""
    return GalvoScannerConfig(angle_range=units.AngleRange("-10 deg", "15 deg"))


@pytest.fixture
def scanner(config: GalvoScannerConfig) -> RecordingGalvoScanner:
    """Provide a fresh scanner whose driver hooks can be inspected."""
    return RecordingGalvoScanner(config)


def test_config_parses_optical_range() -> None:
    """Parse a vendor range without requiring it to contain zero.

    A positive-only optical coordinate range is valid. Construction must not
    impose a centered range or require an implicit zero-angle command.
    """
    config = parse_config(
        GalvoScannerConfig,
        {"angle_range": {"min": "2 deg", "max": "12 deg"}},
    )
    assert config.angle_range == units.AngleRange("2 deg", "12 deg")

    scanner = SimulatedGalvoScanner(config)
    assert scanner.angle is None
    scanner.close()


@pytest.mark.parametrize(
    ("limits", "error_type"),
    [
        (object(), TypeError),
        (units.AngleRange(float("-inf"), 1.0), ValueError),
        (units.AngleRange(-1.0, float("inf")), ValueError),
    ],
)
def test_config_rejects_invalid_ranges(
    limits: object,
    error_type: type[Exception],
) -> None:
    """Require an AngleRange with finite lower and upper endpoints."""
    with pytest.raises(error_type, match="angle_range"):
        GalvoScannerConfig(angle_range=cast(units.AngleRange, limits))


def test_constructor_rejects_incorrect_config_type() -> None:
    """Reject objects outside the galvo configuration family."""
    with pytest.raises(TypeError, match="config must be a GalvoScannerConfig"):
        SimulatedGalvoScanner(cast(GalvoScannerConfig, object()))


def test_construction_does_not_command_output(
    config: GalvoScannerConfig,
    scanner: RecordingGalvoScanner,
) -> None:
    """Store configuration with unknown command history and no driver calls."""
    assert scanner.config is config
    assert scanner.angle is None
    assert scanner.writes == []
    assert scanner.close_calls == 0


@pytest.mark.parametrize("degrees", [-10, -3, 0, 15])
def test_static_commands_include_both_endpoints(
    scanner: RecordingGalvoScanner,
    degrees: int,
) -> None:
    """Accept signed optical angles, including both inclusive range bounds."""
    value = units.Angle(f"{degrees} deg")

    scanner.angle = value

    assert scanner.angle == value
    assert scanner.writes == [value]


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        (0.0, TypeError, "must be an Angle"),
        ("1 deg", TypeError, "must be an Angle"),
        (units.Voltage("1 V"), TypeError, "must be an Angle"),
        (units.Angle(float("nan")), ValueError, "must be finite"),
        (units.Angle(float("inf")), ValueError, "must be finite"),
        (units.Angle(float("-inf")), ValueError, "must be finite"),
        (units.Angle("-11 deg"), ValueError, "within angle_range"),
        (units.Angle("16 deg"), ValueError, "within angle_range"),
    ],
)
def test_invalid_commands_leave_history_and_driver_untouched(
    scanner: RecordingGalvoScanner,
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Validate commands before invoking the driver or changing history.

    Establish a successful command, then submit an invalid type, nonfinite
    angle, or out-of-range angle. Only the original valid command may
    appear in the hook record or successful-command history.
    """
    previous = units.Angle("1 deg")
    scanner.angle = previous

    with pytest.raises(error_type, match=message):
        scanner.angle = cast(units.Angle, value)

    assert scanner.angle == previous
    assert scanner.writes == [previous]


def test_driver_failure_preserves_last_successful_static_angle(
    scanner: RecordingGalvoScanner,
) -> None:
    """Propagate driver failure without recording the attempted angle as applied.

    The hook records its attempted write before raising. That attempt must
    not replace the previous successful static command.
    """
    previous = units.Angle("1 deg")
    attempted = units.Angle("2 deg")
    scanner.angle = previous
    failure = RuntimeError("Output is unavailable.")
    scanner.write_failure = failure

    with pytest.raises(RuntimeError) as caught:
        scanner.angle = attempted

    assert caught.value is failure
    assert scanner.angle == previous
    assert scanner.writes == [previous, attempted]


def test_close_is_idempotent_and_does_not_park(
    scanner: RecordingGalvoScanner,
) -> None:
    """Clean up once, preserve readable history, and reject later commands.

    Closing after a successful command must not add a zero-angle or other
    parking write. Repeated closure must not repeat cleanup.
    """
    previous = units.Angle("3 deg")
    scanner.angle = previous

    scanner.close()
    scanner.close()

    with pytest.raises(RuntimeError, match="closed"):
        scanner.angle = units.Angle("0 deg")

    assert scanner.close_calls == 1
    assert scanner.writes == [previous]
    assert scanner.angle == previous


def test_closed_scanner_cannot_be_reserved_for_waveform_output(
    scanner: RecordingGalvoScanner,
) -> None:
    """Reject waveform-output reservation after permanent scanner closure."""
    scanner.close()

    with pytest.raises(RuntimeError, match="Scanner is closed"):
        scanner._reserve_output(object())

    assert scanner.close_calls == 1
    assert scanner.writes == []


def test_cleanup_failure_still_permanently_closes_scanner(
    scanner: RecordingGalvoScanner,
) -> None:
    """Preserve cleanup failure while permanently disabling later commands.

    The first close raises the driver's original error. A repeated close
    does not retry cleanup, and a later angle command cannot reach the driver.
    """
    failure = RuntimeError("Cleanup failed.")
    scanner.close_failure = failure

    with pytest.raises(RuntimeError) as caught:
        scanner.close()

    assert caught.value is failure
    scanner.close()

    with pytest.raises(RuntimeError, match="closed"):
        scanner.angle = units.Angle("0 deg")

    assert scanner.close_calls == 1
    assert scanner.writes == []


def test_simulated_snapshot_records_static_history(
    config: GalvoScannerConfig,
) -> None:
    """Record optical limits and static command history in base units.

    Before a scalar command, history is unknown. After a command, the
    simulator records the requested optical angle without simulating motion.
    The metadata key explicitly identifies this as static-command history.
    """
    scanner = SimulatedGalvoScanner(config)
    try:
        assert scanner.snapshot()["settings"]["last_static_angle"] is None

        value = units.Angle("3 deg")
        scanner.angle = value
        settings = scanner.snapshot()["settings"]

        assert settings["angle_range"] == {
            "min": {"value": float(config.angle_range.min), "unit": "rad"},
            "max": {"value": float(config.angle_range.max), "unit": "rad"},
        }
        assert settings["last_static_angle"] == {
            "value": float(value),
            "unit": "rad",
        }
    finally:
        scanner.close()


@pytest.mark.parametrize("closed", [False, True])
def test_waveform_validation_is_read_only(
    config: GalvoScannerConfig,
    scanner: RecordingGalvoScanner,
    closed: bool,
) -> None:
    """Accept inclusive optical bounds without commanding or reserving the axis.

    Establish a static command, then validate a waveform containing both
    range endpoints and an interior sample. Validation must preserve driver
    hook records and static history, whether the scanner is open or closed.
    """
    previous = units.Angle("1 deg")
    scanner.angle = previous
    if closed:
        scanner.close()

    waveform = AngularWaveform(
        [
            float(config.angle_range.min),
            0.0,
            float(config.angle_range.max),
        ],
        sample_rate=units.SampleRate("10 kS/s"),
    )

    assert scanner.validate_waveform(waveform) is None
    assert scanner.angle == previous
    assert scanner.writes == [previous]
    assert scanner.close_calls == int(closed)


@pytest.mark.parametrize("overshoot", ["-11 deg", "16 deg"])
def test_waveform_validation_checks_interior_excursions(
    scanner: RecordingGalvoScanner,
    overshoot: str,
) -> None:
    """Reject an out-of-range interior sample even when both endpoints are valid.

    A turnaround or flyback excursion can violate limits inside a waveform.
    Checking only its first and last samples would incorrectly accept this
    buffer. Rejection must leave the driver and static history untouched.
    """
    waveform = AngularWaveform(
        [0.0, float(units.Angle(overshoot)), 0.0],
        sample_rate=units.SampleRate("10 kS/s"),
    )

    with pytest.raises(ValueError, match="within angle_range"):
        scanner.validate_waveform(waveform)

    assert scanner.angle is None
    assert scanner.writes == []
    assert scanner.close_calls == 0


def test_waveform_validation_rejects_other_objects(
    scanner: RecordingGalvoScanner,
) -> None:
    """Require a validated AngularWaveform rather than accepting a raw buffer."""
    with pytest.raises(TypeError, match="waveform must be an AngularWaveform"):
        scanner.validate_waveform(cast(AngularWaveform, object()))

    assert scanner.angle is None
    assert scanner.writes == []
