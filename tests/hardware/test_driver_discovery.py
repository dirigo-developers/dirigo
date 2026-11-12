"""Test category-aware discovery and lazy loading of device entry points.

Controlled metadata and fake drivers keep these tests independent of installed
plugins, vendor SDKs, and physical hardware.
"""

from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import cast

import pytest

from dirigo.hardware import config as config_module
from dirigo.hardware.config import (
    discover_device_categories,
    discover_drivers,
    load_hardware,
)
from dirigo.hardware.device import Device


@dataclass(frozen=True)
class DemoConfig:
    """Minimal initialization model for a discovered driver."""

    channel: str = "default"


class DemoDevice(Device):
    """Store parsed configuration and record closure."""

    config_type = DemoConfig

    def __init__(self, config: DemoConfig) -> None:
        """Store the initialization model and start with an open device."""
        self.config = config
        self.closed = False

    def close(self) -> None:
        """Record closure without accessing hardware."""
        self.closed = True


@pytest.fixture
def registrations(
    monkeypatch: pytest.MonkeyPatch,
) -> list[importlib_metadata.EntryPoint]:
    """Replace installed metadata with registrations controlled by each test."""
    entries: list[importlib_metadata.EntryPoint] = []

    def entry_points() -> importlib_metadata.EntryPoints:
        """Return the current controlled registrations."""
        return importlib_metadata.EntryPoints(entries)

    monkeypatch.setattr(
        config_module.importlib_metadata,
        "entry_points",
        entry_points,
    )
    return entries


def _entry(
    name: str,
    target: str = "example_driver:Device",
    *,
    group: str = "dirigo.devices.scanners",
) -> importlib_metadata.EntryPoint:
    """Build an unloaded registration in the requested entry-point group."""
    return importlib_metadata.EntryPoint(
        name=name,
        value=target,
        group=group,
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
    """List categories and their registrations without importing drivers.

    Scanner and digitizer groups both contain a driver named demo, which
    must remain independently selectable. Legacy and unrelated groups are
    excluded. Forbidding EntryPoint.load also proves that listing categories
    and inspecting a category do not import any implementations.
    """
    scanner = _entry("demo")
    unused = _entry("unused", "missing_sdk:Device")
    digitizer = _entry(
        "demo",
        "other_driver:Device",
        group="dirigo.devices.digitizers",
    )
    registrations.extend(
        [
            scanner,
            unused,
            digitizer,
            _entry("legacy", group="dirigo.devices"),
            _entry("unrelated", group="other.plugins"),
        ]
    )

    def forbidden_load(entry: importlib_metadata.EntryPoint) -> object:
        """Fail if metadata discovery attempts to import an implementation."""
        raise AssertionError(f"Unexpected load: {entry.name}")

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", forbidden_load)

    assert discover_device_categories() == ("digitizers", "scanners")
    assert discover_drivers("scanners") == {
        "demo": scanner,
        "unused": unused,
    }
    assert discover_drivers("digitizers") == {"demo": digitizer}
    assert discover_drivers("missing") == {}


def test_empty_discovery_is_valid(
    registrations: list[importlib_metadata.EntryPoint],
) -> None:
    """Return empty results when no device entry points are installed."""
    assert registrations == []
    assert discover_device_categories() == ()
    assert discover_drivers("scanners") == {}


def test_duplicate_names_are_rejected(
    registrations: list[importlib_metadata.EntryPoint],
) -> None:
    """Reject duplicate names only within the selected category.

    Two scanner registrations share a name but point to different targets.
    The error must identify both targets. An independently registered
    digitizer with that name remains discoverable despite the scanner
    category's ambiguity.
    """
    digitizer = _entry("demo", group="dirigo.devices.digitizers")
    registrations.extend(
        [
            _entry("demo", "first:Device"),
            _entry("demo", "second:Device"),
            digitizer,
        ]
    )

    with pytest.raises(ValueError, match="demo") as caught:
        discover_drivers("scanners")

    assert "first:Device" in str(caught.value)
    assert "second:Device" in str(caught.value)
    assert discover_drivers("digitizers") == {"demo": digitizer}


def test_loads_only_selected_drivers_once_per_category(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache imports by category and name while constructing distinct devices.

    Two scanner declarations share one registration and receive separate
    parsed configurations. A third declaration selects the same driver name
    from the digitizer category, requiring a separate entry-point load.
    An unused scanner registration must never be loaded, and all three
    returned devices must be closed with the collection.
    """
    registrations.extend(
        [
            _entry("demo"),
            _entry("unused", "missing_sdk:Device"),
            _entry("demo", group="dirigo.devices.digitizers"),
        ]
    )
    loaded: list[tuple[str, str]] = []

    def resolve(entry: importlib_metadata.EntryPoint) -> object:
        """Record the selected registration and substitute a fake driver."""
        loaded.append((entry.group, entry.name))
        assert entry.name == "demo"
        return DemoDevice

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", resolve)
    path = _write_config(
        tmp_path,
        """
[devices.first]
category = "scanners"
driver = "demo"
config = {channel = "ao0"}

[devices.second]
category = "scanners"
driver = "demo"
config = {channel = "ao1"}

[devices.third]
category = "digitizers"
driver = "demo"
config = {channel = "ai0"}
""",
    )
    hardware = load_hardware(path)

    try:
        first = hardware.require("first", DemoDevice)
        second = hardware.require("second", DemoDevice)
        third = hardware.require("third", DemoDevice)
        assert first is not second
        assert first is not third
        assert second is not third
        assert first.config.channel == "ao0"
        assert second.config.channel == "ao1"
        assert third.config.channel == "ai0"
        assert loaded == [
            ("dirigo.devices.scanners", "demo"),
            ("dirigo.devices.digitizers", "demo"),
        ]
    finally:
        hardware.close()

    assert first.closed
    assert second.closed
    assert third.closed


def test_explicit_registry_including_empty_bypasses_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exactly the supplied categorized registry without discovery.

    A populated registry supplies the selected driver directly. An explicit
    empty registry must report an unknown driver instead of falling back
    to installed entry points. Both paths run with discovery forbidden.
    """

    def forbidden_discovery(
        category: str,
    ) -> dict[str, importlib_metadata.EntryPoint]:
        """Fail if an explicit registry triggers driver discovery."""
        raise AssertionError("Discovery must be bypassed.")

    monkeypatch.setattr(config_module, "discover_drivers", forbidden_discovery)
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "demo"
""",
    )

    hardware = load_hardware(path, drivers={"scanners": {"demo": DemoDevice}})
    hardware.close()

    with pytest.raises(ValueError, match="unknown driver"):
        load_hardware(path, drivers={})


def test_missing_driver_reports_available_registrations(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report available drivers from the requested category without imports.

    The requested scanner driver is absent. Another scanner is available,
    and a driver exists in the digitizer category. Only the available
    scanner belongs in the diagnostic; no registered implementation needs
    to be loaded to produce that message.
    """
    registrations.extend(
        [
            _entry("demo"),
            _entry(
                "digitizer_only",
                group="dirigo.devices.digitizers",
            ),
        ]
    )

    def forbidden_load(entry: importlib_metadata.EntryPoint) -> object:
        """Fail if reporting available registrations imports a driver."""
        raise AssertionError(f"Unexpected load: {entry.name}")

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", forbidden_load)
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "missing"
""",
    )

    with pytest.raises(ValueError, match="unknown driver") as caught:
        load_hardware(path)

    assert "category 'scanners'" in str(caught.value)
    assert "Available drivers: demo" in str(caught.value)
    assert "digitizer_only" not in str(caught.value)


def test_driver_import_failure_preserves_exception_and_device_context(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve an import failure and identify the selected device and driver.

    Loading the selected registration raises a prepared ImportError.
    The exact exception object must propagate, with a note identifying
    the device, driver name, and category that required the import.
    """
    registrations.append(_entry("demo"))
    failure = ImportError("Required SDK is unavailable.")

    def fail(entry: importlib_metadata.EntryPoint) -> object:
        """Raise the prepared SDK import failure."""
        raise failure

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", fail)
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "demo"
""",
    )

    with pytest.raises(ImportError) as caught:
        load_hardware(path)

    assert caught.value is failure
    assert any(
        all(value in note for value in ("demo", "board", "scanners"))
        for note in failure.__notes__
    )


@pytest.mark.parametrize("loaded", [object(), int])
def test_loaded_registration_must_satisfy_driver_contract(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
    loaded: object,
) -> None:
    """Validate discovered objects before treating them as device drivers.

    A non-callable object fails callability validation. The int class is
    callable but lacks a configuration model. Errors in either case must
    identify the selected driver and category.
    """
    registrations.append(_entry("demo"))

    def resolve(entry: importlib_metadata.EntryPoint) -> object:
        """Return the deliberately invalid driver declaration."""
        return loaded

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", resolve)
    path = _write_config(
        tmp_path,
        """
[devices.board]
category = "scanners"
driver = "demo"
""",
    )

    with pytest.raises(TypeError, match="demo") as caught:
        load_hardware(path)

    assert "scanners" in str(caught.value)


def test_discovered_configs_are_all_validated_before_construction(
    tmp_path: Path,
    registrations: list[importlib_metadata.EntryPoint],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish configuration prevalidation before calling discovered drivers.

    The entry point resolves successfully to a recording driver class.
    The first device has a valid default configuration, but the second
    supplies an integer for a string field. The parser must reject that
    field before either constructor runs.
    """
    registrations.append(_entry("demo"))
    created: list[DemoDevice] = []

    class RecordingDevice(DemoDevice):
        """Record whether construction was reached."""

        def __init__(self, config: DemoConfig) -> None:
            """Store the configuration and record the constructed device."""
            super().__init__(config)
            created.append(self)

    def resolve(entry: importlib_metadata.EntryPoint) -> object:
        """Resolve the selected entry point to the recording driver."""
        return RecordingDevice

    monkeypatch.setattr(importlib_metadata.EntryPoint, "load", resolve)
    path = _write_config(
        tmp_path,
        """
[devices.good]
category = "scanners"
driver = "demo"

[devices.bad]
category = "scanners"
driver = "demo"
config = {channel = 3}
""",
    )

    with pytest.raises(TypeError, match="channel"):
        load_hardware(path)

    assert created == []


@pytest.mark.parametrize(
    ("category", "error_type", "message"),
    [
        (123, TypeError, "category must be a string"),
        ("", ValueError, "category must not be empty"),
    ],
)
def test_discovery_rejects_invalid_categories(
    category: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Reject non-string and empty categories through the public discovery API.

    The cast permits deliberately invalid input past static checks without
    changing its runtime value.
    """
    with pytest.raises(error_type, match=message):
        discover_drivers(cast(str, category))
