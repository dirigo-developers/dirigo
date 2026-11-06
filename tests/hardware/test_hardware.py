"""Test the Hardware collection's device lookup and ownership rules.

Verify construction, named lookup with interface checking, and aliases referring
to one device. Closing must attempt every distinct device in reverse insertion
order and collect failures without repeating cleanup.

The collection remains closed after cleanup failure, while configured names
remain available for inspection. Configuration loading is tested separately.
"""

from collections.abc import Mapping

import pytest

from dirigo.hardware import Device, Hardware


class FakeDevice(Device):
    """Record cleanup attempts and optionally raise a supplied exception."""

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
    """Provide a distinct scanner interface for lookup tests."""

    pass


class FakeDigitizer(FakeDevice):
    """Provide a distinct digitizer interface for lookup tests."""

    pass


def test_empty_collection_is_valid() -> None:
    """An empty collection has no names and can be closed repeatedly."""
    hardware = Hardware({})

    assert hardware.names == ()

    hardware.close()
    hardware.close()


def test_names_preserve_insertion_order() -> None:
    """Configured names retain their mapping insertion order."""
    hardware = Hardware(
        {
            "y_galvo": FakeScanner(),
            "digitizer": FakeDigitizer(),
            "x_galvo": FakeScanner(),
        }
    )

    assert hardware.names == ("y_galvo", "digitizer", "x_galvo")


def test_constructor_copies_the_mapping() -> None:
    """Later changes to the supplied mapping do not alter the collection.

    Replace an existing entry and add another after construction. The collection
    must retain its original names and the original device instance.
    """
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
    devices: Mapping[object, object],
    expected_error: type[Exception],
) -> None:
    """Reject empty names, nonstring names, and values without a Device interface."""
    with pytest.raises(expected_error):
        Hardware(devices)  # ty: ignore[invalid-argument-type]


def test_failed_construction_does_not_close_supplied_devices() -> None:
    """Failed validation leaves cleanup responsibility with the caller.

    Supply a valid device before an invalid entry. Encountering the invalid
    entry must not close the device supplied earlier in the mapping.
    """
    device = FakeScanner()

    with pytest.raises(ValueError, match="names must not be empty"):
        Hardware(
            {
                "scanner": device,
                "": FakeDevice(),
            }
        )

    assert device.close_calls == 0


def test_require_selects_named_instance_among_devices_of_same_type() -> None:
    """Names distinguish separate devices implementing the same interface."""
    x_galvo = FakeScanner()
    y_galvo = FakeScanner()
    hardware = Hardware(
        {
            "x_galvo": x_galvo,
            "y_galvo": y_galvo,
        }
    )

    assert hardware.require("x_galvo", FakeScanner) is x_galvo
    assert hardware.require("y_galvo", FakeScanner) is y_galvo


def test_repeated_lookup_returns_same_object() -> None:
    """Repeated lookup returns the owned device without copying or closing it."""
    scanner = FakeScanner()
    hardware = Hardware({"scanner": scanner})

    first = hardware.require("scanner", FakeScanner)
    second = hardware.require("scanner", FakeScanner)

    assert first is second is scanner
    assert scanner.close_calls == 0


def test_require_accepts_a_base_interface() -> None:
    """A device can be requested through a base interface that it implements."""
    scanner = FakeScanner()
    hardware = Hardware({"scanner": scanner})

    assert hardware.require("scanner", Device) is scanner


def test_require_missing_name_raises_key_error() -> None:
    """Looking up an unconfigured device raises KeyError identifying its name."""
    hardware = Hardware({"scanner": FakeScanner()})

    with pytest.raises(KeyError) as caught:
        hardware.require("missing", FakeScanner)

    assert "missing" in str(caught.value)


def test_require_wrong_interface_reports_name_and_types() -> None:
    """An interface mismatch identifies the device and expected and actual types."""
    hardware = Hardware({"galvo": FakeDigitizer()})

    with pytest.raises(TypeError) as caught:
        hardware.require("galvo", FakeScanner)

    message = str(caught.value)

    assert "galvo" in message
    assert "FakeScanner" in message
    assert "FakeDigitizer" in message


def test_close_uses_reverse_insertion_order() -> None:
    """Device cleanup proceeds in reverse mapping insertion order."""
    events: list[str] = []
    hardware = Hardware(
        {
            "controller": FakeDevice("controller", events=events),
            "scanner": FakeScanner("scanner", events=events),
            "digitizer": FakeDigitizer("digitizer", events=events),
        }
    )

    hardware.close()

    assert events == ["digitizer", "scanner", "controller"]


def test_aliases_resolve_to_same_device_and_close_it_once() -> None:
    """Multiple names can share one device without causing repeated cleanup."""
    scanner = FakeScanner()
    hardware = Hardware(
        {
            "x_galvo": scanner,
            "scan_axis": scanner,
        }
    )

    assert hardware.require("x_galvo", FakeScanner) is scanner
    assert hardware.require("scan_axis", FakeScanner) is scanner

    hardware.close()

    assert scanner.close_calls == 1


def test_repeated_close_does_not_close_devices_again() -> None:
    """Closing an already closed collection does not repeat device cleanup."""
    scanner = FakeScanner()
    digitizer = FakeDigitizer()
    hardware = Hardware(
        {
            "scanner": scanner,
            "digitizer": digitizer,
        }
    )

    hardware.close()
    hardware.close()

    assert scanner.close_calls == 1
    assert digitizer.close_calls == 1


@pytest.mark.parametrize("name", ["scanner", "missing"])
def test_require_after_close_raises_runtime_error(name: str) -> None:
    """Closure prevents lookup before checking whether the requested name exists."""
    hardware = Hardware({"scanner": FakeScanner()})

    hardware.close()

    with pytest.raises(RuntimeError, match="closed"):
        hardware.require(name, FakeScanner)


def test_names_remain_available_after_close() -> None:
    """Configured names remain inspectable after device access has ended."""
    hardware = Hardware(
        {
            "scanner": FakeScanner(),
            "digitizer": FakeDigitizer(),
        }
    )

    hardware.close()

    assert hardware.names == ("scanner", "digitizer")


@pytest.mark.parametrize("failure_count", [1, 2])
def test_close_attempts_every_device_and_collects_failures(
    failure_count: int,
) -> None:
    """Cleanup attempts every device before reporting an ExceptionGroup.

    Place failing devices between healthy devices so cleanup must continue
    after encountering an error. Every device must be attempted exactly once
    in reverse insertion order.

    The exception group must contain the original failures in cleanup order.
    """
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
    """Cleanup failure leaves the collection closed and does not permit retries.

    After the initial ExceptionGroup, lookup must remain forbidden while names
    remain inspectable. Another close call must neither retry device cleanup
    nor raise the previous failure again.
    """
    scanner = FakeScanner(failure=RuntimeError("Cleanup failed."))
    hardware = Hardware({"scanner": scanner})

    with pytest.raises(ExceptionGroup):
        hardware.close()

    with pytest.raises(RuntimeError, match="closed"):
        hardware.require("scanner", FakeScanner)

    assert hardware.names == ("scanner",)

    hardware.close()

    assert scanner.close_calls == 1
