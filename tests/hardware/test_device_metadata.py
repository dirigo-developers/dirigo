"""Test detached scientific metadata and explicit quantity serialization."""

import json

import pytest

from dirigo import units
from dirigo.hardware.device import Device
from dirigo.hardware.metadata import DeviceIdentity, DriverIdentity, encode_metadata
from dirigo.hardware.scanner import ResonantScannerConfig
from dirigo.hardware.simulated import SimulatedResonantScanner


class MetadataDevice(Device):
    """Expose selected mutable settings without owning external resources."""

    def __init__(self) -> None:
        self.settings: dict[str, object] = {}

    def _snapshot_settings(self) -> dict[str, object]:
        return self.settings

    def close(self) -> None:
        return


@pytest.fixture
def scanner() -> SimulatedResonantScanner:
    """Provide a scanner with declared physical parameters."""
    return SimulatedResonantScanner(
        ResonantScannerConfig(
            nominal_frequency=units.Frequency("7.91 kHz"),
            max_amplitude=units.Angle("26 deg"),
            settling_time=units.Time("100 ms"),
            frequency_tolerance=0.003,
        )
    )


def test_default_identity_does_not_invent_hardware_information() -> None:
    """Report unknown hardware fields and the concrete Python implementation."""
    device = MetadataDevice()
    snapshot = device.snapshot()

    assert snapshot["identity"] == {
        "manufacturer": None,
        "model": None,
        "serial_number": None,
    }
    assert snapshot["driver"] == {
        "implementation": f"{MetadataDevice.__module__}.{MetadataDevice.__qualname__}",
        "distribution": None,
        "version": None,
    }
    assert snapshot["settings"] == {}


def test_driver_can_supply_physical_and_software_identity() -> None:
    """Keep the controlled equipment distinct from its driver distribution."""

    class IdentifiedDevice(MetadataDevice):
        @property
        def identity(self) -> DeviceIdentity:
            return DeviceIdentity(
                manufacturer="Example Instruments",
                model="RS-1",
                serial_number="12345",
            )

        @property
        def driver_identity(self) -> DriverIdentity:
            return DriverIdentity(
                implementation="example_driver.Scanner",
                distribution="example-driver",
                version="1.2.3",
            )

    snapshot = IdentifiedDevice().snapshot()

    assert snapshot["identity"] == {
        "manufacturer": "Example Instruments",
        "model": "RS-1",
        "serial_number": "12345",
    }
    assert snapshot["driver"] == {
        "implementation": "example_driver.Scanner",
        "distribution": "example-driver",
        "version": "1.2.3",
    }


def test_nested_settings_are_detached_in_both_directions() -> None:
    """Copy nested containers rather than exposing device-owned objects."""
    device = MetadataDevice()
    channels = [{"enabled": True}]
    device.settings = {"channels": channels}

    snapshot = device.snapshot()
    channels[0]["enabled"] = False

    assert snapshot["settings"]["channels"] == [{"enabled": True}]

    copied_channels = snapshot["settings"]["channels"]
    assert isinstance(copied_channels, list)
    copied_channels.append(None)

    assert channels == [{"enabled": False}]


def test_encoder_preserves_scalars_and_encodes_nested_quantities() -> None:
    """Preserve scalar types while representing quantities with explicit units."""
    encoded = encode_metadata(
        {
            "enabled": True,
            "count": 3,
            "label": "channel A",
            "unknown": None,
            "gain": 1.25,
            "ranges": (units.Voltage("-2 V"), units.Voltage("2 V")),
        }
    )

    assert encoded == {
        "enabled": True,
        "count": 3,
        "label": "channel A",
        "unknown": None,
        "gain": 1.25,
        "ranges": [
            {"value": -2.0, "unit": "V"},
            {"value": 2.0, "unit": "V"},
        ],
    }
    assert encoded["enabled"] is True


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), units.Angle(float("inf"))],
)
def test_nonfinite_metadata_is_rejected(value: object) -> None:
    """Reject numeric values that cannot be represented in strict JSON."""
    device = MetadataDevice()
    device.settings["invalid"] = value

    with pytest.raises(ValueError, match="finite"):
        device.snapshot()


def test_unsupported_metadata_is_rejected() -> None:
    """Require drivers to explicitly describe objects they want recorded."""
    device = MetadataDevice()
    device.settings["invalid"] = object()

    with pytest.raises(TypeError, match="Unsupported metadata value type"):
        device.snapshot()


def test_metadata_keys_must_be_strings() -> None:
    """Reject nonstring keys instead of silently changing their meaning."""
    with pytest.raises(TypeError, match="field names must be strings"):
        encode_metadata({1: "invalid"})  # ty: ignore[invalid-argument-type]


def test_scanner_snapshot_records_declared_parameters_and_unknown_amplitude(
    scanner: SimulatedResonantScanner,
) -> None:
    """Record selected scientific fields without issuing an amplitude command."""
    settings = scanner.snapshot()["settings"]
    config = scanner.config

    assert settings == {
        "nominal_frequency": {"value": float(config.nominal_frequency), "unit": "Hz"},
        "max_amplitude": {"value": float(config.max_amplitude), "unit": "rad"},
        "settling_time": {"value": float(config.settling_time), "unit": "s"},
        "frequency_tolerance": config.frequency_tolerance,
        "commanded_amplitude": None,
    }
    assert scanner.amplitude is None


def test_scanner_snapshot_preserves_precision_and_command_history(
    scanner: SimulatedResonantScanner,
) -> None:
    """Keep earlier snapshots independent of later commands and closure."""
    amplitude = units.Angle("12.3456789012345 deg")
    scanner.amplitude = amplitude
    snapshot = scanner.snapshot()

    assert snapshot["settings"]["commanded_amplitude"] == {
        "value": float(amplitude),
        "unit": "rad",
    }
    assert json.loads(json.dumps(snapshot, allow_nan=False)) == snapshot

    scanner.amplitude = units.Angle("3 deg")
    scanner.close()

    assert snapshot["settings"]["commanded_amplitude"] == {
        "value": float(amplitude),
        "unit": "rad",
    }
    assert scanner.snapshot()["settings"]["commanded_amplitude"] == {
        "value": float(units.Angle("3 deg")),
        "unit": "rad",
    }


def test_device_snapshot_defaults_to_empty_settings() -> None:
    """Use the default empty settings when a driver selects no fields."""

    class MinimalDevice(Device):
        def close(self) -> None:
            """No external resources require cleanup."""
            return

    device = MinimalDevice()

    assert device.snapshot()["settings"] == {}
