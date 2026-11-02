from pathlib import Path
from typing import cast

import pytest

from dirigo.hardware import Device
from dirigo.hardware_config import DeviceFactory, load_hardware


class FakeDevice(Device):
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


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "system_config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_named_instances_and_passes_configuration(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, """
        [devices.x]
        driver = "demo"
        [devices.x.config]
        channel = "ao0"
        gain = 2.5

        [devices.y]
        driver = "demo"
        [devices.y.config]
        channel = "ao1"
    """)

    hardware = load_hardware(path, drivers={"demo": FakeDevice})
    x = hardware.require("x", FakeDevice)
    y = hardware.require("y", FakeDevice)

    assert hardware.names == ("x", "y")
    assert x is not y
    assert (x.channel, x.gain) == ("ao0", 2.5)
    assert (y.channel, y.gain) == ("ao1", 1.0)
    assert x.close_calls == y.close_calls == 0

    hardware.close()

    assert x.close_calls == y.close_calls == 1


def test_config_is_optional_and_string_paths_are_accepted(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, """
        [devices.board]
        driver = "demo"
    """)

    hardware = load_hardware(
        str(path),
        drivers={"demo": FakeDevice},
    )

    assert hardware.require("board", FakeDevice).channel == "default"
    hardware.close()


def test_registry_accepts_factory_functions(tmp_path: Path) -> None:
    device = FakeDevice()

    def factory() -> Device:
        return device

    path = write_config(tmp_path, """
        [devices.board]
        driver = "demo"
    """)

    hardware = load_hardware(path, drivers={"demo": factory})

    assert hardware.require("board", FakeDevice) is device
    hardware.close()


def test_empty_devices_table_is_valid(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[devices]")
    hardware = load_hardware(path, drivers={})

    assert hardware.names == ()
    hardware.close()


def test_invalid_definitions_are_rejected(tmp_path: Path) -> None:
    invalid_documents = [
        "[other]",
        "devices = 3",
        "[devices]\nboard = 3",
        '[devices.""]\ndriver = "demo"',
        "[devices.board]",
        "[devices.board]\ndriver = 3",
        '[devices.board]\ndriver = ""',
        '[devices.board]\ndriver = "demo"\nconfig = 3',
        '[devices.board]\ndriver = "demo"\nconfgi = {}',
        '[devices.board]\ndriver = "unknown"',
    ]

    for document in invalid_documents:
        path = write_config(tmp_path, document)

        with pytest.raises(ValueError):
            load_hardware(path, drivers={"demo": FakeDevice})


def test_validates_all_entries_before_constructing_any_device(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def factory() -> Device:
        calls.append("constructed")
        return FakeDevice()

    path = write_config(tmp_path, """
        [devices.good]
        driver = "demo"

        [devices.bad]
        driver = "unknown"
    """)

    with pytest.raises(ValueError, match="bad.*unknown"):
        load_hardware(path, drivers={"demo": factory})

    assert calls == []


def test_noncallable_driver_is_rejected_before_construction(
    tmp_path: Path,
) -> None:
    path = write_config(tmp_path, """
        [devices.board]
        driver = "demo"
    """)
    invalid_factory = cast(DeviceFactory, object())

    with pytest.raises(TypeError, match="demo"):
        load_hardware(path, drivers={"demo": invalid_factory})


def test_constructor_failure_rolls_back_in_reverse_order(
    tmp_path: Path,
) -> None:
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

    path = write_config(tmp_path, """
        [devices.first]
        driver = "demo"
        config = {channel = "first"}

        [devices.second]
        driver = "demo"
        config = {channel = "second"}

        [devices.broken]
        driver = "fail"
    """)

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
    device = FakeDevice()

    def good() -> Device:
        return device

    def bad() -> Device:
        return cast(Device, object())

    path = write_config(tmp_path, """
        [devices.good]
        driver = "good"

        [devices.bad]
        driver = "bad"
    """)

    with pytest.raises(TypeError, match="bad"):
        load_hardware(path, drivers={"good": good, "bad": bad})

    assert device.close_calls == 1


def test_rollback_failure_does_not_hide_construction_failure(
    tmp_path: Path,
) -> None:
    failure = RuntimeError("Connection failed.")

    class FailingCloseDevice(FakeDevice):
        def close(self) -> None:
            super().close()
            raise RuntimeError("Disconnect failed.")

    healthy = FakeDevice()
    broken = FailingCloseDevice()

    def fail() -> Device:
        raise failure

    path = write_config(tmp_path, """
        [devices.healthy]
        driver = "healthy"

        [devices.broken_close]
        driver = "broken_close"

        [devices.broken_constructor]
        driver = "fail"
    """)

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
    with pytest.raises(FileNotFoundError):
        load_hardware(tmp_path / "missing.toml", drivers={})
