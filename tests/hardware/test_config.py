"""Test loading named hardware devices from TOML configuration.

Verify driver selection, keyword configuration, defaults, and validation before
device construction. Registry entries may be classes or factory functions.

Construction failures must close previously returned devices in reverse order.
Rollback failures must not replace the original construction exception.
"""

from pathlib import Path
from typing import cast

import pytest

from dirigo.hardware import Device
from dirigo.hardware.config import DeviceFactory, load_hardware


class FakeDevice(Device):
    """Store constructor configuration and count cleanup calls."""

    def __init__(
        self,
        *,
        channel: str = "default",
        gain: float = 1.0,
    ) -> None:
        self.channel = channel
        self.gain = gain
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _write_config(tmp_path: Path, text: str) -> Path:
    """Write a UTF-8 configuration file in the test's temporary directory."""
    path = tmp_path / "system_config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_named_instances_and_passes_configuration(
    tmp_path: Path,
) -> None:
    """Construct distinct named devices using their configured keyword arguments.

    Both entries select the same driver but supply different channels. Omitted
    constructor arguments must retain their defaults, and closing the returned
    collection must close both devices.
    """
    path = _write_config(
        tmp_path,
        """
        [devices.x]
        driver = "demo"
        [devices.x.config]
        channel = "ao0"
        gain = 2.5

        [devices.y]
        driver = "demo"
        [devices.y.config]
        channel = "ao1"
        """,
    )

    hardware = load_hardware(path, drivers={"demo": FakeDevice})
    x = hardware.require("x", FakeDevice)
    y = hardware.require("y", FakeDevice)

    try:
        assert hardware.names == ("x", "y")
        assert x is not y
        assert (x.channel, x.gain) == ("ao0", 2.5)
        assert (y.channel, y.gain) == ("ao1", 1.0)
        assert x.close_calls == y.close_calls == 0
    finally:
        hardware.close()

    assert x.close_calls == y.close_calls == 1


def test_config_is_optional_and_string_paths_are_accepted(
    tmp_path: Path,
) -> None:
    """A string path and an omitted config table use the driver's defaults."""
    path = _write_config(
        tmp_path,
        """
        [devices.board]
        driver = "demo"
        """,
    )

    hardware = load_hardware(
        str(path),
        drivers={"demo": FakeDevice},
    )

    try:
        assert hardware.require("board", FakeDevice).channel == "default"
    finally:
        hardware.close()


def test_registry_accepts_factory_functions(tmp_path: Path) -> None:
    """A registered function can supply the device owned by the collection."""
    device = FakeDevice()

    def factory() -> Device:
        return device

    path = _write_config(
        tmp_path,
        """
        [devices.board]
        driver = "demo"
        """,
    )

    hardware = load_hardware(path, drivers={"demo": factory})

    try:
        assert hardware.require("board", FakeDevice) is device
    finally:
        hardware.close()

    assert device.close_calls == 1


def test_empty_devices_table_is_valid(tmp_path: Path) -> None:
    """An empty devices table produces a valid empty Hardware collection."""
    path = _write_config(tmp_path, "[devices]")
    hardware = load_hardware(path, drivers={})

    try:
        assert hardware.names == ()
    finally:
        hardware.close()


@pytest.mark.parametrize(
    "document",
    [
        pytest.param(
            "[other]",
            id="missing-devices-table",
        ),
        pytest.param(
            "devices = 3",
            id="devices-is-not-a-table",
        ),
        pytest.param(
            "[devices]\nboard = 3",
            id="device-is-not-a-table",
        ),
        pytest.param(
            '[devices.""]\ndriver = "demo"',
            id="empty-device-name",
        ),
        pytest.param(
            "[devices.board]",
            id="missing-driver",
        ),
        pytest.param(
            "[devices.board]\ndriver = 3",
            id="nonstring-driver",
        ),
        pytest.param(
            '[devices.board]\ndriver = ""',
            id="empty-driver",
        ),
        pytest.param(
            '[devices.board]\ndriver = "demo"\nconfig = 3',
            id="config-is-not-a-table",
        ),
        pytest.param(
            '[devices.board]\ndriver = "demo"\nconfgi = {}',
            id="unknown-field",
        ),
        pytest.param(
            '[devices.board]\ndriver = "unknown"',
            id="unregistered-driver",
        ),
    ],
)
def test_invalid_definitions_are_rejected(
    tmp_path: Path,
    document: str,
) -> None:
    """Reject malformed device definitions and unregistered driver selections."""
    path = _write_config(tmp_path, document)

    with pytest.raises(ValueError):
        load_hardware(path, drivers={"demo": FakeDevice})


def test_validates_all_entries_before_constructing_any_device(
    tmp_path: Path,
) -> None:
    """An invalid later entry prevents construction of an earlier valid device.

    Record factory calls for the first entry, then select an unknown driver in
    the second. Validation must reject the document before invoking the factory.
    """
    calls: list[str] = []

    def factory() -> Device:
        calls.append("constructed")
        return FakeDevice()

    path = _write_config(
        tmp_path,
        """
        [devices.good]
        driver = "demo"

        [devices.bad]
        driver = "unknown"
        """,
    )

    with pytest.raises(ValueError, match="bad.*unknown"):
        load_hardware(path, drivers={"demo": factory})

    assert calls == []


def test_noncallable_driver_is_rejected_before_construction(
    tmp_path: Path,
) -> None:
    """Selecting a registry entry that is not callable raises a driver-specific error."""
    path = _write_config(
        tmp_path,
        """
        [devices.board]
        driver = "demo"
        """,
    )
    invalid_factory = cast(DeviceFactory, object())

    with pytest.raises(TypeError, match="demo"):
        load_hardware(path, drivers={"demo": invalid_factory})


def test_constructor_failure_rolls_back_in_reverse_order(
    tmp_path: Path,
) -> None:
    """A construction failure closes previously created devices in reverse order.

    Construct two devices successfully before a third factory raises. Both
    earlier devices must be closed exactly once, and the original exception
    must identify the device whose construction failed through an added note.
    """
    events: list[str] = []
    created: list[FakeDevice] = []
    failure = RuntimeError("Connection failed.")

    class RecordingDevice(FakeDevice):
        def close(self) -> None:
            super().close()
            events.append(self.channel)

    def factory(*, channel: str) -> Device:
        device = RecordingDevice(channel=channel)
        created.append(device)
        return device

    def fail() -> Device:
        raise failure

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
        driver = "fail"
        """,
    )

    with pytest.raises(RuntimeError) as caught:
        load_hardware(
            path,
            drivers={"demo": factory, "fail": fail},
        )

    assert caught.value is failure
    assert events == ["second", "first"]
    assert all(device.close_calls == 1 for device in created)
    assert any("broken" in note for note in failure.__notes__)


def test_invalid_factory_result_rolls_back_previous_devices(
    tmp_path: Path,
) -> None:
    """Returning a non-Device fails construction and closes earlier devices.

    A callable registry entry is not sufficient: its result must implement
    Device. Reject the invalid result while releasing the device already
    returned by the preceding factory.
    """
    device = FakeDevice()

    def good() -> Device:
        return device

    def bad() -> Device:
        return cast(Device, object())

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
        load_hardware(path, drivers={"good": good, "bad": bad})

    assert device.close_calls == 1


def test_rollback_failure_does_not_hide_construction_failure(
    tmp_path: Path,
) -> None:
    """Rollback continues after cleanup failure and preserves the construction error.

    After two devices are created, a third factory fails. The first rollback
    cleanup also raises, but the remaining healthy device must still be closed.

    The original construction exception must be re-raised with a note describing
    the additional cleanup failure.
    """
    failure = RuntimeError("Connection failed.")

    class FailingCloseDevice(FakeDevice):
        def close(self) -> None:
            super().close()
            raise RuntimeError("Disconnect failed.")

    healthy = FakeDevice()
    broken = FailingCloseDevice()

    def fail() -> Device:
        raise failure

    path = _write_config(
        tmp_path,
        """
        [devices.healthy]
        driver = "healthy"

        [devices.broken_close]
        driver = "broken_close"

        [devices.broken_constructor]
        driver = "fail"
        """,
    )

    with pytest.raises(RuntimeError) as caught:
        load_hardware(
            path,
            drivers={
                "healthy": lambda: healthy,
                "broken_close": lambda: broken,
                "fail": fail,
            },
        )

    assert caught.value is failure
    assert healthy.close_calls == broken.close_calls == 1
    assert any(
        "Disconnect failed" in note
        for note in failure.__notes__
    )


def test_missing_file_is_reported(tmp_path: Path) -> None:
    """Loading a nonexistent configuration file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_hardware(tmp_path / "missing.toml", drivers={})
