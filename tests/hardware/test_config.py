"""Test categorized hardware loading, typed configuration, and rollback."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from dirigo import units
from dirigo.hardware.config import DeviceDriver, load_hardware
from dirigo.hardware.device import Device

_DEFAULT_TIMEOUT = units.Time("100 ms")


@dataclass(frozen=True, kw_only=True)
class FakeConfig:
    """Provide defaults, quantity parsing, and a physical constraint."""

    channel: str = "default"
    gain: float = 1.0
    timeout: units.Time = _DEFAULT_TIMEOUT

    def __post_init__(self) -> None:
        """Reject negative timeout values after parsing."""
        if float(self.timeout) < 0:
            raise ValueError("timeout must be nonnegative.")


class FakeDevice(Device):
    """Accept typed configuration and count cleanup attempts."""

    config_type = FakeConfig

    def __init__(self, config: FakeConfig) -> None:
        """Store the parsed model and initialize the cleanup counter."""
        self.config = config
        self.close_calls = 0

    def close(self) -> None:
        """Record a cleanup attempt without accessing hardware."""
        self.close_calls += 1


class FixedDriver:
    """Return a predetermined device and record the configuration received."""

    config_type = FakeConfig

    def __init__(self, device: Device) -> None:
        """Store the result to return when the driver is called."""
        self.device = device
        self.received: FakeConfig | None = None

    def __call__(self, config: FakeConfig) -> Device:
        """Record the parsed model and return the supplied device."""
        self.received = config
        return self.device


def _write_config(tmp_path: Path, text: str) -> Path:
    """Write a configuration document in the test's temporary directory."""
    path = tmp_path / "system_config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_named_instances_with_typed_configuration(
    tmp_path: Path,
) -> None:
    """Construct distinct named devices from parsed models and apply defaults.

    Both declarations select the same driver in the scanners category.
    One overrides every field, including a quantity string; the other
    supplies only a channel and receives the remaining model defaults.
    Closing the collection cleans up each constructed device once.
    """
    path = _write_config(
        tmp_path,
        """
[devices.x]
category = "scanners"
driver = "demo"
config = {channel = "ao0", gain = 2.5, timeout = "25 ms"}

[devices.y]
category = "scanners"
driver = "demo"
config = {channel = "ao1"}
""",
    )
    hardware = load_hardware(path, drivers={"scanners": {"demo": FakeDevice}})

    try:
        x = hardware.require("x", FakeDevice)
        y = hardware.require("y", FakeDevice)
        assert hardware.names == ("x", "y")
        assert x is not y
        assert x.config == FakeConfig(
            channel="ao0", gain=2.5, timeout=units.Time("25 ms")
        )
        assert y.config == FakeConfig(channel="ao1")
    finally:
        hardware.close()

    assert x.close_calls == y.close_calls == 1


def test_config_is_optional_and_string_paths_are_accepted(
    tmp_path: Path,
) -> None:
    """Accept a string path and use model defaults when config is omitted."""
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "demo"
""",
    )
    hardware = load_hardware(str(path), drivers={"scanners": {"demo": FakeDevice}})

    try:
        assert hardware.require("board", FakeDevice).config == FakeConfig()
    finally:
        hardware.close()


def test_registry_accepts_callable_factory_objects(
    tmp_path: Path,
) -> None:
    """Pass a parsed model to a factory object and own its returned device.

    The categorized registry contains a callable instance rather than a
    driver class. The loader returns the exact device supplied by that
    factory and closes it when the hardware collection is closed.
    """
    device = FakeDevice(FakeConfig())
    driver = FixedDriver(device)
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "demo"
""",
    )
    hardware = load_hardware(path, drivers={"scanners": {"demo": driver}})

    try:
        assert hardware.require("board", FakeDevice) is device
        assert driver.received == FakeConfig()
    finally:
        hardware.close()

    assert device.close_calls == 1


@pytest.mark.parametrize(
    ("invalid_config", "error", "field_name"),
    [
        ('gain = "wrong"', TypeError, "gain"),
        ('timeout = "1 V"', ValueError, "timeout"),
        ('timeout = "-1 ms"', ValueError, "timeout"),
        ("typo = 1", ValueError, "typo"),
    ],
)
def test_validates_all_configs_before_constructing_devices(
    tmp_path: Path,
    invalid_config: str,
    error: type[Exception],
    field_name: str,
) -> None:
    """Reject a later invalid config before constructing either device.

    The first declaration is valid. The second exercises an invalid scalar
    type, incompatible units, a physical constraint, or an unknown field.
    No constructor may run, and the error must identify the field plus the
    affected device, driver, and category.
    """
    created: list[FakeDevice] = []

    class RecordingDevice(FakeDevice):
        """Record whether construction was reached."""

        def __init__(self, config: FakeConfig) -> None:
            """Store configuration and record this constructed device."""
            super().__init__(config)
            created.append(self)

    path = _write_config(
        tmp_path,
        f"""
[devices.good]
category = "scanners"
driver = "demo"

[devices.bad]
category = "scanners"
driver = "demo"

[devices.bad.config]
{invalid_config}
""",
    )

    with pytest.raises(error, match=field_name) as caught:
        load_hardware(path, drivers={"scanners": {"demo": RecordingDevice}})

    assert created == []
    assert any(
        all(value in note for value in ("bad", "demo", "scanners"))
        for note in caught.value.__notes__
    )


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[other]", "must contain"),
        ("devices = 3", "must contain"),
        ("[devices]\nboard = 3", "must be a table"),
        (
            '[devices.""]\ncategory = "scanners"\ndriver = "demo"',
            "names must not be empty",
        ),
        (
            '[devices.board]\ndriver = "demo"',
            "requires a nonempty string category",
        ),
        (
            '[devices.board]\ncategory = 3\ndriver = "demo"',
            "requires a nonempty string category",
        ),
        (
            '[devices.board]\ncategory = ""\ndriver = "demo"',
            "requires a nonempty string category",
        ),
        (
            '[devices.board]\ncategory = "scanners"',
            "requires a nonempty string driver name",
        ),
        (
            '[devices.board]\ncategory = "scanners"\ndriver = 3',
            "requires a nonempty string driver name",
        ),
        (
            '[devices.board]\ncategory = "scanners"\ndriver = ""',
            "requires a nonempty string driver name",
        ),
        (
            '[devices.board]\ncategory = "scanners"\ndriver = "demo"\nconfig = 3',
            "config must be a table",
        ),
        (
            '[devices.board]\ncategory = "scanners"\ndriver = "demo"\nconfgi = {}',
            "unsupported fields",
        ),
        (
            '[devices.board]\ncategory = "scanners"\ndriver = "unknown"',
            "unknown driver",
        ),
    ],
)
def test_invalid_definitions_are_rejected(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    """Reject malformed documents and invalid device-selection fields.

    Cases isolate missing or malformed devices tables, invalid device names,
    missing or invalid category and driver names, malformed config tables,
    unsupported declaration fields, and unavailable driver selections.
    Other required fields are supplied where needed so each case reaches
    its intended validation.
    """
    path = _write_config(tmp_path, document)

    with pytest.raises(ValueError, match=message):
        load_hardware(path, drivers={"scanners": {"demo": FakeDevice}})


@pytest.mark.parametrize("driver", [object(), int])
def test_invalid_driver_declarations_are_rejected(
    tmp_path: Path,
    driver: object,
) -> None:
    """Require a callable driver with a declared dataclass configuration.

    An object instance fails callability validation. The int class is
    callable but lacks config_type. Both failures must identify the
    selected driver.
    """
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "demo"
""",
    )

    with pytest.raises(TypeError, match="demo"):
        load_hardware(
            path,
            drivers={
                "scanners": {
                    "demo": cast(DeviceDriver[Any], driver),
                },
            },
        )


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_construction_failure_rolls_back_in_reverse_order(
    tmp_path: Path,
    cleanup_fails: bool,
) -> None:
    """Close earlier devices in reverse order and preserve constructor failure.

    Two devices are returned successfully before the third constructor raises.
    Rollback must close the second and then the first, once each. When closing
    the second also raises, cleanup must still reach the first, and the
    original constructor exception must remain primary with cleanup details
    attached as an error note.
    """
    events: list[str] = []
    created: list[FakeDevice] = []
    failure = RuntimeError("Connection failed.")

    class RecordingDevice(FakeDevice):
        """Record construction and closure, with configurable failures."""

        def __init__(self, config: FakeConfig) -> None:
            """Raise for the failing channel or record a returned device."""
            if config.channel == "fail":
                raise failure
            super().__init__(config)
            created.append(self)

        def close(self) -> None:
            """Record cleanup before optionally reporting a disconnect error."""
            super().close()
            events.append(self.config.channel)
            if cleanup_fails and self.config.channel == "second":
                raise RuntimeError("Disconnect failed.")

    path = _write_config(
        tmp_path,
        """
[devices.first]
category = "scanners"
driver = "demo"
config = {channel = "first"}

[devices.second]
category = "scanners"
driver = "demo"
config = {channel = "second"}

[devices.broken]
category = "scanners"
driver = "demo"
config = {channel = "fail"}
""",
    )

    with pytest.raises(RuntimeError) as caught:
        load_hardware(path, drivers={"scanners": {"demo": RecordingDevice}})

    assert caught.value is failure
    assert events == ["second", "first"]
    assert len(created) == 2
    assert all(device.close_calls == 1 for device in created)
    assert any("broken" in note for note in failure.__notes__)

    if cleanup_fails:
        assert any("Disconnect failed" in note for note in failure.__notes__)


def test_invalid_factory_result_rolls_back_previous_devices(
    tmp_path: Path,
) -> None:
    """Reject a non-Device factory result and close earlier returned devices.

    The first factory returns a valid device. The second declares a valid
    configuration model but returns an ordinary object. Result validation
    must identify the bad device and roll back ownership of the good one.
    """
    device = FakeDevice(FakeConfig())
    path = _write_config(
        tmp_path,
        """
[devices.good]
category = "scanners"
driver = "good"

[devices.bad]
category = "scanners"
driver = "bad"
""",
    )

    with pytest.raises(TypeError, match="bad"):
        load_hardware(
            path,
            drivers={
                "scanners": {
                    "good": FixedDriver(device),
                    "bad": FixedDriver(cast(Device, object())),
                },
            },
        )

    assert device.close_calls == 1


def test_empty_devices_table_is_valid(tmp_path: Path) -> None:
    """Allow an empty devices table with an explicit empty registry."""
    hardware = load_hardware(
        _write_config(tmp_path, "[devices]"),
        drivers={},
    )

    assert hardware.names == ()
    hardware.close()


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """Propagate FileNotFoundError when the configuration path is absent."""
    with pytest.raises(FileNotFoundError):
        load_hardware(tmp_path / "missing.toml", drivers={})


def test_registry_category_requires_a_mapping(tmp_path: Path) -> None:
    """Reject a category whose registry value is not a driver mapping.

    The configuration requests no devices. Registry validation must still
    reject the malformed category rather than accepting it because no
    driver lookup is needed.
    """
    path = _write_config(tmp_path, "[devices]")
    invalid_members = cast(dict[str, DeviceDriver[Any]], object())

    with pytest.raises(
        TypeError,
        match="Driver category 'scanners' must contain a mapping",
    ):
        load_hardware(
            path,
            drivers={"scanners": invalid_members},
        )
