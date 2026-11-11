import tomllib
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from importlib import metadata as importlib_metadata
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, get_type_hints

from dirigo import units
from dirigo.hardware.device import Device
from dirigo.hardware.hardware import Hardware


class DeviceDriver[ConfigT](Protocol):
    """A callable that constructs a device from its declared configuration.

    Concrete driver classes can satisfy this protocol by declaring config_type
    as a class attribute and accepting that configuration in their constructor.
    Callable factory objects may implement the same contract.
    """

    @property
    def config_type(self) -> type[ConfigT]:
        """Initialization dataclass, inspectable without constructing hardware."""
        ...

    def __call__(self, config: ConfigT, /) -> Device:
        """Construct a device from validated configuration.

        Release partially acquired resources if construction raises.
        """
        ...


def discover_drivers() -> dict[str, importlib_metadata.EntryPoint]:
    """Discover installed registrations in the dirigo.devices entry-point group.

    Return unloaded entry points keyed by their case-sensitive names. Inspecting
    this mapping does not import driver modules or construct devices. Call
    an entry point's load() method to inspect its driver's config_type.

    Do not cache results globally; each call inspects installed registrations.

    Raises:
        ValueError: Multiple registrations use the same driver name.
    """
    discovered: dict[str, importlib_metadata.EntryPoint] = {}

    for entry in importlib_metadata.entry_points(group="dirigo.devices"):
        if entry.name in discovered:
            previous = discovered[entry.name]
            raise ValueError(
                f"Duplicate driver registration {entry.name!r}: "
                f"{previous.value!r} and {entry.value!r}."
            )

        discovered[entry.name] = entry

    return discovered


def load_hardware(
    path: str | PathLike[str],
    *,
    drivers: Mapping[str, DeviceDriver[Any]] | None = None,
) -> Hardware:
    """Construct an owned hardware collection from a TOML devices table.

    Each device selects a registered driver and optionally supplies a config
    table. The driver declares its initialization dataclass through config_type.

    Parse and validate every configuration before constructing any devices.
    Configuration validation must not acquire hardware resources. Construct
    devices in table order, passing each its validated configuration object.

    If construction fails, close previously returned devices in reverse order
    and re-raise the original failure. A driver that fails before returning a
    device must clean up its own partially acquired resources.

    When drivers is None, discover installed entry points and load only selected
    driver implementations, once per driver name during this call. Driver imports
    must not initialize hardware. An explicit registry, including an empty mapping,
    bypasses discovery.

    Args:
        path: Path to the TOML configuration file.
        drivers: Explicit registry of driver classes or callable factory objects.
            None selects installed registrations in the dirigo.devices group.

    Returns:
        A Hardware collection owning the constructed devices.

    Raises:
        OSError: The configuration file cannot be read.
        tomllib.TOMLDecodeError: The document is not valid TOML.
        ValueError: A device definition or configuration is invalid.
        TypeError: A driver declaration, configuration value, or returned
            device has an invalid type.

    Model validation and driver construction exceptions propagate with a note
    identifying the device. Rollback failures are attached as additional notes.
    """
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)

    definitions = document.get("devices")
    if not isinstance(definitions, dict):
        raise ValueError("Configuration must contain a [devices] table.")

    registry: dict[str, DeviceDriver[Any]] = {} if drivers is None else dict(drivers)
    registrations: dict[str, importlib_metadata.EntryPoint] = (
        discover_drivers() if drivers is None else {}
    )
    plan: list[tuple[str, DeviceDriver[Any], object]] = []

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

        raw_config = definition.get("config", {})
        if not isinstance(raw_config, dict):
            raise ValueError(f"Device {name!r} config must be a table.")

        if driver_name not in registry:
            if driver_name not in registrations:
                available = ", ".join(sorted(registry.keys() | registrations.keys()))
                raise ValueError(
                    f"Device {name!r} selects unknown driver {driver_name!r}. "
                    f"Available drivers: {available or 'none'}."
                )

            try:
                registry[driver_name] = registrations[driver_name].load()
            except BaseException as exc:
                exc.add_note(
                    f"While loading driver {driver_name!r} for device {name!r}."
                )
                raise

        driver = registry[driver_name]

        if not callable(driver):
            raise TypeError(f"Driver {driver_name!r} must be callable.")

        config_type = getattr(driver, "config_type", None)
        if not isinstance(config_type, type) or not is_dataclass(config_type):
            raise TypeError(
                f"Driver {driver_name!r} must declare a dataclass config_type."
            )

        try:
            config = parse_config(config_type, raw_config)
        except BaseException as exc:
            exc.add_note(
                f"While validating device {name!r} with driver {driver_name!r}."
            )
            raise

        plan.append((name, driver, config))

    devices: dict[str, Device] = {}

    try:
        for name, driver, config in plan:
            try:
                device = driver(config)
                if not isinstance(device, Device):
                    raise TypeError(
                        f"Driver for device {name!r} must return "
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


def parse_config[T](model_type: type[T], values: Mapping[str, object]) -> T:
    """Parse a configuration table into a validated dataclass instance.

    Support initialization fields annotated as str, bool, int, float, or a
    Quantity subclass. Quantity inputs must be strings containing units.
    Float fields accept plain integers or floats; other scalar fields require
    their exact type. Booleans are not accepted as numbers.

    Omitted fields retain their declared defaults or default factories.
    Fields excluded from dataclass initialization cannot be supplied.

    Construction runs the model's own validation. This function does not
    modify the input mapping or perform hardware access.

    Args:
        model_type: Dataclass type accepting its initialization fields as
            keyword arguments.
        values: Configuration field names mapped to unparsed values.

    Returns:
        An instance of model_type containing parsed values.

    Raises:
        TypeError: The arguments have invalid types, a field annotation is
            unsupported, or an input value has an inappropriate type.
        ValueError: A field is unknown, a required field is missing, or a
            value cannot be converted.

    Exceptions from model construction and validation propagate unchanged.
    """
    if not isinstance(model_type, type) or not is_dataclass(model_type):
        raise TypeError("model_type must be a dataclass type.")

    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping.")

    if any(not isinstance(name, str) for name in values):
        raise TypeError("Configuration field names must be strings.")

    definitions = {item.name: item for item in fields(model_type) if item.init}

    unknown = values.keys() - definitions.keys()
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}.")

    missing = [
        name
        for name, item in definitions.items()
        if name not in values
        and item.default is MISSING
        and item.default_factory is MISSING
    ]
    if missing:
        raise ValueError(
            f"Missing required configuration fields: {', '.join(missing)}."
        )

    annotations = get_type_hints(model_type)
    parsed: dict[str, object] = {}

    for name in definitions:
        expected_type = annotations[name]
        if not isinstance(expected_type, type) or not (
            expected_type in (str, bool, int, float)
            or issubclass(expected_type, units.Quantity)
        ):
            raise TypeError(f"Unsupported annotation for configuration field {name!r}.")

        if name not in values:
            continue

        try:
            parsed[name] = _parse_config_value(expected_type, values[name])
        except TypeError as exc:
            raise TypeError(f"Configuration field {name!r}: {exc}") from exc
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Configuration field {name!r}: {exc}") from exc

    return model_type(**parsed)


def _parse_config_value(expected_type: type[object], value: object) -> object:
    """Convert one value after its field annotation has been checked."""
    if issubclass(expected_type, units.Quantity):
        if not isinstance(value, str):
            raise TypeError("Expected a quantity string containing units.")
        return expected_type(value)

    if expected_type is float:
        if type(value) is int or type(value) is float:
            return float(value)
        raise TypeError("Expected a plain integer or float.")

    if type(value) is not expected_type:
        raise TypeError(f"Expected {expected_type.__name__}.")

    return value
