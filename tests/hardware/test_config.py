"""Test typed hardware initialization and cleanup after construction failure."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from dirigo import units
from dirigo.hardware import Device
from dirigo.hardware.config import DeviceDriver, load_hardware

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
        self.config = config
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class FixedDriver:
    """A callable factory object declaring the configuration it accepts."""

    config_type = FakeConfig

    def __init__(self, device: Device) -> None:
        self.device = device
        self.received: FakeConfig | None = None

    def __call__(self, config: FakeConfig) -> Device:
        self.received = config
        return self.device


def _write_config(tmp_path: Path, text: str) -> Path:
    """Write a configuration document in the test's temporary directory."""
    path = tmp_path / "system_config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_named_instances_with_typed_configuration(tmp_path: Path) -> None:
    """Convert quantities and apply defaults before passing models to drivers."""
    path = _write_config(
        tmp_path,
        """
        [devices.x]
        driver = "demo"
        config = {channel = "ao0", gain = 2.5, timeout = "25 ms"}

        [devices.y]
        driver = "demo"
        config = {channel = "ao1"}
        """,
    )
    hardware = load_hardware(path, drivers={"demo": FakeDevice})

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


def test_config_is_optional_and_string_paths_are_accepted(tmp_path: Path) -> None:
    """An omitted table constructs the model using its defaults."""
    path = _write_config(tmp_path, '[devices.board]\ndriver = "demo"')
    hardware = load_hardware(str(path), drivers={"demo": FakeDevice})

    try:
        assert hardware.require("board", FakeDevice).config == FakeConfig()
    finally:
        hardware.close()


def test_registry_accepts_callable_factory_objects(tmp_path: Path) -> None:
    """Factory objects receive the same parsed model as driver constructors."""
    device = FakeDevice(FakeConfig())
    driver = FixedDriver(device)
    path = _write_config(tmp_path, '[devices.board]\ndriver = "demo"')
    hardware = load_hardware(path, drivers={"demo": driver})

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
    """A later parsing or model-validation failure prevents all construction."""
    created: list[FakeDevice] = []

    class RecordingDevice(FakeDevice):
        def __init__(self, config: FakeConfig) -> None:
            super().__init__(config)
            created.append(self)

    path = _write_config(
        tmp_path,
        f"""
        [devices.good]
        driver = "demo"

        [devices.bad]
        driver = "demo"
        [devices.bad.config]
        {invalid_config}
        """,
    )

    with pytest.raises(error, match=field_name) as caught:
        load_hardware(path, drivers={"demo": RecordingDevice})

    assert created == []
    assert any("bad" in note and "demo" in note for note in caught.value.__notes__)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("[other]", "must contain"),
        ("devices = 3", "must contain"),
        ("[devices]\nboard = 3", "must be a table"),
        ('[devices.""]\ndriver = "demo"', "names must not be empty"),
        ("[devices.board]", "requires a nonempty string"),
        ("[devices.board]\ndriver = 3", "requires a nonempty string"),
        ('[devices.board]\ndriver = ""', "requires a nonempty string"),
        ('[devices.board]\ndriver = "demo"\nconfig = 3', "config must be a table"),
        ('[devices.board]\ndriver = "demo"\nconfgi = {}', "unsupported fields"),
        ('[devices.board]\ndriver = "unknown"', "unknown driver"),
    ],
)
def test_invalid_definitions_are_rejected(
    tmp_path: Path,
    document: str,
    message: str,
) -> None:
    """Retain structural validation of the configuration document."""
    path = _write_config(tmp_path, document)

    with pytest.raises(ValueError, match=message):
        load_hardware(path, drivers={"demo": FakeDevice})


@pytest.mark.parametrize("driver", [object(), int])
def test_invalid_driver_declarations_are_rejected(
    tmp_path: Path,
    driver: object,
) -> None:
    """Require both a callable and a declared initialization model."""
    path = _write_config(tmp_path, '[devices.board]\ndriver = "demo"')

    with pytest.raises(TypeError, match="demo"):
        load_hardware(path, drivers={"demo": cast(DeviceDriver[Any], driver)})


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_construction_failure_rolls_back_in_reverse_order(
    tmp_path: Path,
    cleanup_fails: bool,
) -> None:
    """Attempt all cleanup and preserve the original construction exception."""
    events: list[str] = []
    created: list[FakeDevice] = []
    failure = RuntimeError("Connection failed.")

    class RecordingDevice(FakeDevice):
        def __init__(self, config: FakeConfig) -> None:
            if config.channel == "fail":
                raise failure
            super().__init__(config)
            created.append(self)

        def close(self) -> None:
            super().close()
            events.append(self.config.channel)
            if cleanup_fails and self.config.channel == "second":
                raise RuntimeError("Disconnect failed.")

    path = _write_config(
        tmp_path,
        """
        [devices.first]
        driver = "demo"
        config = {channel = "first"}

        [devices.second]
        driver = "demo"
        config = {channel = "second"}

        [devices.broken]
        driver = "demo"
        config = {channel = "fail"}
        """,
    )

    with pytest.raises(RuntimeError) as caught:
        load_hardware(path, drivers={"demo": RecordingDevice})

    assert caught.value is failure
    assert events == ["second", "first"]
    assert len(created) == 2
    assert all(device.close_calls == 1 for device in created)
    assert any("broken" in note for note in failure.__notes__)

    if cleanup_fails:
        assert any("Disconnect failed" in note for note in failure.__notes__)


def test_invalid_factory_result_rolls_back_previous_devices(tmp_path: Path) -> None:
    """Reject a non-Device result while closing earlier devices."""
    device = FakeDevice(FakeConfig())
    path = _write_config(
        tmp_path,
        """
        [devices.good]
        driver = "good"

        [devices.bad]
        driver = "bad"
        """,
    )

    with pytest.raises(TypeError, match="bad"):
        load_hardware(
            path,
            drivers={
                "good": FixedDriver(device),
                "bad": FixedDriver(cast(Device, object())),
            },
        )

    assert device.close_calls == 1


def test_empty_devices_table_is_valid(tmp_path: Path) -> None:
    """Allow a configuration with no devices."""
    hardware = load_hardware(_write_config(tmp_path, "[devices]"), drivers={})

    assert hardware.names == ()
    hardware.close()


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """Preserve the file-opening exception."""
    with pytest.raises(FileNotFoundError):
        load_hardware(tmp_path / "missing.toml", drivers={})
