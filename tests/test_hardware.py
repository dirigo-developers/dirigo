from collections.abc import Mapping

import pytest

from dirigo.hardware import Device, Hardware


class FakeDevice(Device):
    def __init__(
        self,
        label: str = "",
        *,
        events: list[str] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.label = label
        self.events = events
        self.failure = failure
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

        if self.events is not None:
            self.events.append(self.label)

        if self.failure is not None:
            raise self.failure


class FakeScanner(FakeDevice):
    pass


class FakeDigitizer(FakeDevice):
    pass


def test_empty_collection_is_valid() -> None:
    hardware = Hardware({})

    assert hardware.names == ()

    hardware.close()
    hardware.close()


def test_names_preserve_insertion_order() -> None:
    hardware = Hardware({
        "y_galvo": FakeScanner(),
        "digitizer": FakeDigitizer(),
        "x_galvo": FakeScanner(),
    })

    assert hardware.names == ("y_galvo", "digitizer", "x_galvo")


def test_constructor_copies_the_mapping() -> None:
    original = FakeScanner()
    devices: dict[str, Device] = {"scanner": original}
    hardware = Hardware(devices)

    devices["scanner"] = FakeDigitizer()
    devices["other"] = FakeScanner()

    assert hardware.names == ("scanner",)
    assert hardware.require("scanner", FakeScanner) is original


@pytest.mark.parametrize(
    ("devices", "expected_error"),
    [
        ({"": FakeDevice()}, ValueError),
        ({123: FakeDevice()}, TypeError),
        ({"scanner": object()}, TypeError),
    ],
)
def test_constructor_rejects_invalid_entries(
    devices: Mapping[str, Device],
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error):
        Hardware(devices)


def test_failed_construction_does_not_close_supplied_devices() -> None:
    device = FakeScanner()

    with pytest.raises(ValueError):
        Hardware({
            "scanner": device,
            "": FakeDevice(),
        })

    assert device.close_calls == 0


def test_require_selects_named_instance_among_devices_of_same_type() -> None:
    x_galvo = FakeScanner()
    y_galvo = FakeScanner()
    hardware = Hardware({
        "x_galvo": x_galvo,
        "y_galvo": y_galvo,
    })

    assert hardware.require("x_galvo", FakeScanner) is x_galvo
    assert hardware.require("y_galvo", FakeScanner) is y_galvo


def test_repeated_lookup_returns_same_object() -> None:
    scanner = FakeScanner()
    hardware = Hardware({"scanner": scanner})

    first = hardware.require("scanner", FakeScanner)
    second = hardware.require("scanner", FakeScanner)

    assert first is second is scanner
    assert scanner.close_calls == 0


def test_require_accepts_a_base_interface() -> None:
    scanner = FakeScanner()
    hardware = Hardware({"scanner": scanner})

    assert hardware.require("scanner", Device) is scanner


def test_require_missing_name_raises_key_error() -> None:
    hardware = Hardware({"scanner": FakeScanner()})

    with pytest.raises(KeyError) as caught:
        hardware.require("missing", FakeScanner)

    assert "missing" in str(caught.value)


def test_require_wrong_interface_reports_name_and_types() -> None:
    hardware = Hardware({"galvo": FakeDigitizer()})

    with pytest.raises(TypeError) as caught:
        hardware.require("galvo", FakeScanner)

    message = str(caught.value)

    assert "galvo" in message
    assert "FakeScanner" in message
    assert "FakeDigitizer" in message


def test_close_uses_reverse_insertion_order() -> None:
    events: list[str] = []
    hardware = Hardware({
        "controller": FakeDevice("controller", events=events),
        "scanner": FakeScanner("scanner", events=events),
        "digitizer": FakeDigitizer("digitizer", events=events),
    })

    hardware.close()

    assert events == ["digitizer", "scanner", "controller"]


def test_aliases_resolve_to_same_device_and_close_it_once() -> None:
    scanner = FakeScanner()
    hardware = Hardware({
        "x_galvo": scanner,
        "scan_axis": scanner,
    })

    assert hardware.require("x_galvo", FakeScanner) is scanner
    assert hardware.require("scan_axis", FakeScanner) is scanner

    hardware.close()

    assert scanner.close_calls == 1


def test_repeated_close_does_not_close_devices_again() -> None:
    scanner = FakeScanner()
    digitizer = FakeDigitizer()
    hardware = Hardware({
        "scanner": scanner,
        "digitizer": digitizer,
    })

    hardware.close()
    hardware.close()

    assert scanner.close_calls == 1
    assert digitizer.close_calls == 1


@pytest.mark.parametrize("name", ["scanner", "missing"])
def test_require_after_close_raises_runtime_error(name: str) -> None:
    hardware = Hardware({"scanner": FakeScanner()})
    hardware.close()

    with pytest.raises(RuntimeError):
        hardware.require(name, FakeScanner)


def test_names_remain_available_after_close() -> None:
    hardware = Hardware({
        "scanner": FakeScanner(),
        "digitizer": FakeDigitizer(),
    })

    hardware.close()

    assert hardware.names == ("scanner", "digitizer")


@pytest.mark.parametrize("failure_count", [1, 2])
def test_close_attempts_every_device_and_collects_failures(
    failure_count: int,
) -> None:
    events: list[str] = []
    failures = [
        RuntimeError(f"Device {index} failed to close.")
        for index in range(failure_count)
    ]

    devices: dict[str, FakeDevice] = {
        "healthy_before": FakeDevice("healthy_before", events=events),
    }

    for index, failure in enumerate(failures):
        name = f"failing_{index}"
        devices[name] = FakeDevice(
            name,
            events=events,
            failure=failure,
        )

    devices["healthy_after"] = FakeDevice(
        "healthy_after",
        events=events,
    )
    hardware = Hardware(devices)

    with pytest.raises(ExceptionGroup) as caught:
        hardware.close()

    assert events == list(reversed(devices))
    assert all(device.close_calls == 1 for device in devices.values())
    assert caught.value.exceptions == tuple(reversed(failures))


def test_failed_close_still_permanently_closes_collection() -> None:
    scanner = FakeScanner(failure=RuntimeError("Cleanup failed."))
    hardware = Hardware({"scanner": scanner})

    with pytest.raises(ExceptionGroup):
        hardware.close()

    with pytest.raises(RuntimeError):
        hardware.require("scanner", FakeScanner)

    assert hardware.names == ("scanner",)

    # A second close neither retries cleanup nor repeats the exception.
    hardware.close()

    assert scanner.close_calls == 1
