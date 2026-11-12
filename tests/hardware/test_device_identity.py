from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from dirigo.hardware import config as config_module
from dirigo.hardware.config import load_hardware
from dirigo.hardware.device import Device
from dirigo.hardware.metadata import DeviceIdentity


class FakeDevice(Device):
    """A device that needs no base constructor or hardware."""

    def close(self) -> None:
        """Release no resources."""
        return


class ReportingDevice(FakeDevice):
    """Report a partial identity supplied by a driver."""

    @property
    def reported_identity(self) -> DeviceIdentity:
        """Return known physical-device fields."""
        return DeviceIdentity(
            manufacturer="Reported manufacturer",
            serial_number="reported-123",
        )


@dataclass(frozen=True)
class EmptyConfig:
    """A driver configuration with no fields."""


class FakeDriver:
    """Record constructed devices."""

    config_type = EmptyConfig

    def __init__(self) -> None:
        """Initialize the construction record."""
        self.created: list[FakeDevice] = []

    def __call__(self, config: EmptyConfig, /) -> Device:
        """Construct and record a device with no hardware resources."""
        device = FakeDevice()
        self.created.append(device)
        return device


def test_identity_defaults_to_unknown() -> None:
    """Provide unknown identities without requiring a base constructor."""
    device = FakeDevice()

    assert device.configured_identity == DeviceIdentity()
    assert device.reported_identity == DeviceIdentity()
    assert device.identity == DeviceIdentity()


def test_configured_identity_is_per_instance_and_in_snapshot() -> None:
    """Keep configured identity local to its device and include it in snapshots.

    Configure one of two otherwise identical devices. Only that device's
    effective identity changes, and its snapshot retains the unknown serial
    number as None. Clearing its configured fields restores unknown identity.
    """
    first = FakeDevice()
    second = FakeDevice()
    identity = DeviceIdentity(
        manufacturer="Cambridge Technologies",
        model="CRS8K",
    )

    first.configure_identity(identity)

    assert first.configured_identity == identity
    assert first.identity == identity
    assert second.identity == DeviceIdentity()
    assert first.snapshot()["identity"] == {
        "manufacturer": "Cambridge Technologies",
        "model": "CRS8K",
        "serial_number": None,
    }

    first.configure_identity(DeviceIdentity())
    assert first.identity == DeviceIdentity()


def test_reported_fields_take_precedence() -> None:
    """Merge identity field by field while preserving configured declarations.

    The driver reports a manufacturer and serial number that conflict with
    configured values, but leaves the model unknown. Effective identity uses
    the reported fields and configured model. Clearing configuration removes
    that fallback without changing the driver's reported identity.
    """
    device = ReportingDevice()
    configured = DeviceIdentity(
        manufacturer="Configured manufacturer",
        model="CRS8K",
        serial_number="configured-456",
    )
    device.configure_identity(configured)

    assert device.identity == DeviceIdentity(
        manufacturer="Reported manufacturer",
        model="CRS8K",
        serial_number="reported-123",
    )
    assert device.configured_identity == configured

    device.configure_identity(DeviceIdentity())
    assert device.identity == device.reported_identity


def test_identity_rejects_non_string_fields() -> None:
    """Reject a numeric identity field during runtime validation."""
    with pytest.raises(TypeError, match="model must be a string or None"):
        DeviceIdentity(model=cast(str, 123))


def test_configure_identity_rejects_other_objects() -> None:
    """Reject identity-like mappings without changing the device's identity.

    A dictionary containing a valid field name is still not a DeviceIdentity.
    The rejected assignment leaves the device's default identity intact.
    """
    device = FakeDevice()

    with pytest.raises(TypeError, match="identity must be a DeviceIdentity"):
        device.configure_identity(cast(DeviceIdentity, {"model": "CRS8K"}))

    assert device.identity == DeviceIdentity()


def test_loader_applies_identity_outside_driver_config(
    tmp_path: Path,
) -> None:
    """Attach common identity metadata after selecting a categorized driver.

    The explicit scanners registry supplies a driver whose configuration
    accepts no fields. Successful loading therefore demonstrates that identity
    fields remain outside driver configuration. The returned device exposes
    those fields through both its effective identity and scientific snapshot.
    """
    path = tmp_path / "system_config.toml"
    path.write_text(
        """
[devices.fast_scanner]
category = "scanners"
driver = "fake"

[devices.fast_scanner.identity]
manufacturer = "Cambridge Technologies"
model = "CRS8K"
""",
        encoding="utf-8",
    )
    driver = FakeDriver()

    hardware = load_hardware(
        path,
        drivers={"scanners": {"fake": driver}},
    )
    try:
        device = hardware.require("fast_scanner", FakeDevice)

        assert device is driver.created[0]
        assert device.identity == DeviceIdentity(
            manufacturer="Cambridge Technologies",
            model="CRS8K",
        )
        assert device.snapshot()["identity"]["model"] == "CRS8K"
    finally:
        hardware.close()


@pytest.mark.parametrize(
    ("identity_value", "error_type"),
    [
        ('"not a table"', TypeError),
        ('{modle = "CRS8K"}', ValueError),
        ("{model = 123}", TypeError),
        ("{model = false}", TypeError),
    ],
)
def test_invalid_later_identity_prevents_all_construction(
    tmp_path: Path,
    identity_value: str,
    error_type: type[Exception],
) -> None:
    """Validate every identity before constructing any device.

    Both declarations select a valid driver from the scanners category.
    The first declaration is valid, but the second supplies a malformed
    identity table, an unknown field, or a non-string value. Each case must
    raise the expected error before the driver constructs even the first
    device.
    """
    path = tmp_path / "system_config.toml"
    path.write_text(
        f"""
[devices.first]
category = "scanners"
driver = "fake"

[devices.second]
category = "scanners"
driver = "fake"
identity = {identity_value}
""",
        encoding="utf-8",
    )
    driver = FakeDriver()

    with pytest.raises(error_type):
        load_hardware(
            path,
            drivers={"scanners": {"fake": driver}},
        )

    assert driver.created == []


def test_identity_parser_rejects_non_string_field_names() -> None:
    """Reject non-string keys in a Python identity mapping.

    TOML keys are always strings, so a configuration file cannot exercise
    this defensive check. Call the parser directly with an integer key.
    """
    with pytest.raises(
        TypeError,
        match="Identity field names must be strings",
    ):
        config_module._parse_identity({123: "CRS8K"})
