"""Test driver discovery and lazy loading of installed entry points."""

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

import pytest

from dirigo.hardware import config as config_module
from dirigo.hardware.config import discover_drivers, load_hardware
from dirigo.hardware.device import Device


@dataclass(frozen=True)
class DemoConfig:
    """Minimal initialization model for a discovered driver."""

    channel: str = "default"


class DemoDevice(Device):
    """Store parsed configuration and record closure."""

    config_type = DemoConfig

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def registrations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[importlib_metadata.EntryPoint]:
    """Replace installed metadata with a controlled list of registrations."""
    entries: list[importlib_metadata.EntryPoint] = []

    def entry_points(*, group: str) -> importlib_metadata.EntryPoints:
        assert group == "dirigo.devices"
        return importlib_metadata.EntryPoints(entries)

    monkeypatch.setattr(config_module.importlib_metadata, "entry_points", entry_points)
    return entries


def _entry(
    name: str,
    target: str = "example_driver:Device",
) -> importlib_metadata.EntryPoint:
    """Build an unloaded registration without requiring an installed package."""
    return importlib_metadata.EntryPoint(
        name=name,
        value=target,
        group="dirigo.devices",
    )


def _write_config(tmp_path: Path, document: str) -> Path:
    """Write a configuration document for the loader."""
    path = tmp_path / "system_config.toml"
    path.write_text(document, encoding="utf-8")
    return path


def test_discovery_does_not_load_driver_modules(
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose names and targets without importing their implementations."""
    entries = [_entry("demo"), _entry("unused", "missing_sdk:Device")]
    registrations.extend(entries)

    def forbidden_load(entry: importlib_metadata.EntryPoint) -> object:
        raise AssertionError(f"Unexpected load: {entry.name}")

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", forbidden_load)

    assert discover_drivers() == {"demo": entries[0], "unused": entries[1]}


def test_empty_discovery_is_valid(
    registrations: list[importlib_metadata.EntryPoint],
) -> None:
    """Allow an environment without installed drivers."""
    assert registrations == []
    assert discover_drivers() == {}


def test_duplicate_names_are_rejected(
    registrations: list[importlib_metadata.EntryPoint],
) -> None:
    """Reject ambiguity and identify both registered targets."""
    registrations.extend(
        [_entry("demo", "first:Device"), _entry("demo", "second:Device")]
    )

    with pytest.raises(ValueError, match="demo") as caught:
        discover_drivers()

    assert "first:Device" in str(caught.value)
    assert "second:Device" in str(caught.value)


def test_loads_only_selected_driver_and_loads_it_once(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one loaded class to construct multiple independently named devices."""
    registrations.extend([_entry("demo"), _entry("unused", "missing_sdk:Device")])
    loaded: list[str] = []

    def resolve(entry: importlib_metadata.EntryPoint) -> object:
        loaded.append(entry.name)
        assert entry.name == "demo"
        return DemoDevice

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", resolve)
    path = _write_config(
        tmp_path,
        """
        [devices.first]
        driver = "demo"
        config = {channel = "ao0"}

        [devices.second]
        driver = "demo"
        config = {channel = "ao1"}
        """,
    )
    hardware = load_hardware(path)

    try:
        first = hardware.require("first", DemoDevice)
        second = hardware.require("second", DemoDevice)
        assert first is not second
        assert first.config.channel == "ao0"
        assert second.config.channel == "ao1"
        assert loaded == ["demo"]
    finally:
        hardware.close()

    assert first.closed
    assert second.closed


def test_explicit_registry_including_empty_bypasses_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exactly the supplied registry without consulting installed plugins."""

    def forbidden_discovery() -> dict[str, importlib_metadata.EntryPoint]:
        raise AssertionError("Discovery must be bypassed.")

    monkeypatch.setattr(config_module, "discover_drivers", forbidden_discovery)
    path = _write_config(tmp_path, '[devices.board]\ndriver = "demo"')

    hardware = load_hardware(path, drivers={"demo": DemoDevice})
    hardware.close()

    with pytest.raises(ValueError, match="unknown driver"):
        load_hardware(path, drivers={})


def test_missing_driver_reports_available_registrations(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
) -> None:
    """Identify the unavailable driver without loading unrelated registrations."""
    registrations.append(_entry("demo"))
    path = _write_config(tmp_path, '[devices.board]\ndriver = "missing"')

    with pytest.raises(ValueError, match="unknown driver") as caught:
        load_hardware(path)

    assert "Available drivers: demo" in str(caught.value)


def test_driver_import_failure_preserves_exception_and_device_context(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate a selected plugin's import failure with useful context."""
    registrations.append(_entry("demo"))
    failure = ImportError("Required SDK is unavailable.")

    def fail(entry: importlib_metadata.EntryPoint) -> object:
        raise failure

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", fail)
    path = _write_config(tmp_path, '[devices.board]\ndriver = "demo"')

    with pytest.raises(ImportError) as caught:
        load_hardware(path)

    assert caught.value is failure
    assert any("demo" in note and "board" in note for note in failure.__notes__)


@pytest.mark.parametrize("loaded", [object(), int])
def test_loaded_registration_must_satisfy_driver_contract(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
    loaded: object,
) -> None:
    """Apply the same driver checks to discovered and explicit registry entries."""
    registrations.append(_entry("demo"))

    def resolve(entry: importlib_metadata.EntryPoint) -> object:
        return loaded

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", resolve)
    path = _write_config(tmp_path, '[devices.board]\ndriver = "demo"')

    with pytest.raises(TypeError, match="demo"):
        load_hardware(path)


def test_discovered_configs_are_all_validated_before_construction(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing a driver does not bypass configuration prevalidation."""
    registrations.append(_entry("demo"))
    created: list[DemoDevice] = []

    class RecordingDevice(DemoDevice):
        def __init__(self, config: DemoConfig) -> None:
            super().__init__(config)
            created.append(self)

    def resolve(entry: importlib_metadata.EntryPoint) -> object:
        return RecordingDevice

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", resolve)
    path = _write_config(
        tmp_path,
        """
        [devices.good]
        driver = "demo"

        [devices.bad]
        driver = "demo"
        config = {channel = 3}
        """,
    )

    with pytest.raises(TypeError, match="channel"):
        load_hardware(path)

    assert created == []
