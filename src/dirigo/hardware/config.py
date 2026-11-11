import tomllib
from collections.abc import Callable, Mapping
from os import PathLike
from pathlib import Path

from .hardware import Device, Hardware

type DeviceFactory = Callable[..., Device]


def load_hardware(
    path: str | PathLike[str],
    *,
    drivers: Mapping[str, DeviceFactory],
) -> Hardware:
    """Construct an owned hardware collection from a TOML devices table.

    Each entry in [devices] selects a driver from the supplied registry.
    Its optional config table is passed as keyword arguments to that driver's
    factory.

    Validate the structure of every device definition and resolve every driver
    before constructing any devices. Driver-specific configuration values are
    currently validated by the factories during construction.

    Construct devices in TOML table order. If construction fails, close
    previously returned devices in reverse order and re-raise the original
    error. Any rollback failure is recorded in an exception note.

    A factory that fails before returning a device must release its own
    partially acquired resources.

    Args:
        path: Path to the TOML configuration file.
        drivers: Registry mapping driver names to callable device factories.

    Returns:
        A Hardware collection owning the constructed devices.

    Raises:
        OSError: The configuration file cannot be read.
        tomllib.TOMLDecodeError: The file is not valid TOML.
        ValueError: The device definitions are invalid or select an unknown
            driver.
        TypeError: A selected factory is not callable or returns a value
            that is not a Device.

    Exceptions raised by factories propagate with a note identifying the
    device being constructed.
    """
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)

    definitions = document.get("devices")
    if not isinstance(definitions, dict):
        raise ValueError("Configuration must contain a [devices] table.")

    registry = dict(drivers)
    plan: list[tuple[str, DeviceFactory, dict[str, object]]] = []

    for name, definition in definitions.items():
        if not name:
            raise ValueError("Device names must not be empty.")

        if not isinstance(definition, dict):
            raise ValueError(f"Device {name!r} must be a table.")

        unknown = definition.keys() - {"driver", "config"}
        if unknown:
            raise ValueError(
                f"Device {name!r} has unsupported fields: {', '.join(sorted(unknown))}."
            )

        driver_name = definition.get("driver")
        if not isinstance(driver_name, str) or not driver_name:
            raise ValueError(f"Device {name!r} requires a nonempty string driver name.")

        config = definition.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"Device {name!r} config must be a table.")

        if driver_name not in registry:
            raise ValueError(f"Device {name!r} selects unknown driver {driver_name!r}.")

        factory = registry[driver_name]
        if not callable(factory):
            raise TypeError(f"Driver {driver_name!r} must be callable.")

        plan.append((name, factory, config))

    devices: dict[str, Device] = {}

    try:
        for name, factory, config in plan:
            try:
                device = factory(**config)

                if not isinstance(device, Device):
                    raise TypeError(
                        f"Factory for device {name!r} must return "
                        f"a Device; got {type(device).__name__}."
                    )

            except BaseException as exc:
                exc.add_note(f"While constructing device {name!r}.")
                raise

            devices[name] = device

        return Hardware(devices)

    except BaseException as failure:
        try:
            Hardware(devices).close()
        except BaseException as cleanup_failure:
            failure.add_note(f"Hardware rollback also failed: {cleanup_failure!r}")
        raise
