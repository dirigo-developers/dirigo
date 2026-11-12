import tomllib
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from importlib import metadata as importlib_metadata
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, cast, get_type_hints

from dirigo import units
from dirigo.hardware.device import Device
from dirigo.hardware.hardware import Hardware
from dirigo.hardware.metadata import DeviceIdentity


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


def _driver_group(category: str) -> str:
    """Return the entry-point group for a nonempty category."""
    if not isinstance(category, str):
        raise TypeError("category must be a string.")
    if not category:
        raise ValueError("category must not be empty.")
    return f"dirigo.devices.{category}"


def discover_device_categories() -> tuple[str, ...]:
    """Return sorted installed device categories without loading drivers."""
    prefix = "dirigo.devices."
    return tuple(
        sorted(
            group[len(prefix) :]
            for group in importlib_metadata.entry_points().groups
            if group.startswith(prefix) and group != prefix
        )
    )


def discover_drivers(
    category: str,
) -> dict[str, importlib_metadata.EntryPoint]:
    """Discover unloaded drivers registered in one device category.

    Names must be unique within the selected category. Registrations in
    other categories are independent.

    Raises:
        TypeError: category is not a string.
        ValueError: category is empty or the selected group contains
            duplicate driver names.
    """
    group = _driver_group(category)
    drivers: dict[str, importlib_metadata.EntryPoint] = {}

    for entry in importlib_metadata.entry_points().select(group=group):
        previous = drivers.get(entry.name)
        if previous is not None:
            raise ValueError(
                f"Duplicate device driver {entry.name!r} in category "
                f"{category!r}: {previous.value!r} and {entry.value!r}."
            )
        drivers[entry.name] = entry

    return drivers


def load_hardware(
    path: str | PathLike[str],
    *,
    drivers: Mapping[str, Mapping[str, DeviceDriver[Any]]] | None = None,
) -> Hardware:
    """Construct named devices selected by category and driver.

    Each device requires category and driver strings, and may supply config
    and identity tables. Explicit registries map categories to driver maps;
    any supplied registry, including an empty one, bypasses discovery.

    Parse every configuration and identity before constructing devices.
    Discovered implementations are loaded only when selected and cached
    per (category, driver name) for this invocation.

    On construction failure, close previously returned devices and preserve
    the original exception. Drivers must clean up resources themselves if
    their constructors fail before returning a Device.
    """
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)

    definitions = document.get("devices")
    if not isinstance(definitions, dict):
        raise ValueError("Configuration must contain a [devices] table.")

    registry: dict[str, dict[str, DeviceDriver[Any]]] | None = None
    if drivers is not None:
        registry = {}
        for category, members in drivers.items():
            _driver_group(category)
            if not isinstance(members, Mapping):
                raise TypeError(f"Driver category {category!r} must contain a mapping.")
            registry[category] = dict(members)

    discovered: dict[str, dict[str, importlib_metadata.EntryPoint]] = {}
    resolved: dict[tuple[str, str], DeviceDriver[Any]] = {}
    plan: list[tuple[str, DeviceDriver[Any], Any, DeviceIdentity]] = []

    for name, definition in definitions.items():
        if not name:
            raise ValueError("Device names must not be empty.")
        if not isinstance(definition, dict):
            raise ValueError(f"Device {name!r} must be a table.")

        unknown = definition.keys() - {"category", "driver", "config", "identity"}
        if unknown:
            raise ValueError(
                f"Device {name!r} has unsupported fields: {', '.join(sorted(unknown))}."
            )

        category = definition.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"Device {name!r} requires a nonempty string category.")

        driver_name = definition.get("driver")
        if not isinstance(driver_name, str) or not driver_name:
            raise ValueError(f"Device {name!r} requires a nonempty string driver name.")

        values = definition.get("config", {})
        if not isinstance(values, dict):
            raise ValueError(f"Device {name!r} config must be a table.")

        key = (category, driver_name)
        if key not in resolved:
            choices: Mapping[str, object]
            if registry is None:
                if category not in discovered:
                    discovered[category] = discover_drivers(category)
                choices = discovered[category]
            else:
                choices = registry.get(category, {})

            if driver_name not in choices:
                available = ", ".join(sorted(choices)) or "(none)"
                raise ValueError(
                    f"Device {name!r} selects unknown driver "
                    f"{driver_name!r} in category {category!r}. "
                    f"Available drivers: {available}"
                )

            candidate = choices[driver_name]
            if registry is None:
                entry = cast(importlib_metadata.EntryPoint, candidate)
                try:
                    candidate = entry.load()
                except BaseException as exc:
                    exc.add_note(
                        f"While loading driver {driver_name!r} in category "
                        f"{category!r} for device {name!r}."
                    )
                    raise

            if not callable(candidate):
                raise TypeError(
                    f"Driver {driver_name!r} in category {category!r} must be callable."
                )

            model_type = getattr(candidate, "config_type", None)
            if not isinstance(model_type, type) or not is_dataclass(model_type):
                raise TypeError(
                    f"Driver {driver_name!r} in category {category!r} "
                    "config_type must be a dataclass type."
                )

            resolved[key] = cast(DeviceDriver[Any], candidate)

        driver = resolved[key]
        try:
            config = parse_config(driver.config_type, values)
            identity = _parse_identity(definition.get("identity", {}))
        except (TypeError, ValueError) as exc:
            exc.add_note(
                f"While parsing device {name!r} using driver {driver_name!r} "
                f"in category {category!r}."
            )
            raise

        plan.append((name, driver, config, identity))

    devices: dict[str, Device] = {}
    try:
        for name, driver, config, identity in plan:
            try:
                device = driver(config)
                if not isinstance(device, Device):
                    raise TypeError(
                        f"Driver for device {name!r} must return a Device; "
                        f"got {type(device).__name__}."
                    )

                devices[name] = device
                device.configure_identity(identity)
            except BaseException as exc:
                exc.add_note(f"While constructing device {name!r}.")
                raise

        return Hardware(devices)
    except BaseException as failure:
        try:
            Hardware(devices).close()
        except BaseException as cleanup_failure:
            failure.add_note(f"Hardware rollback also failed: {cleanup_failure!r}")
        raise


def parse_config[T](model_type: type[T], values: Mapping[str, object]) -> T:
    """Parse a configuration table into a validated dataclass instance.

    Support initialization fields annotated as str, bool, int, float, a Quantity
    subclass, or a concrete QuantityRange subclass.

    Quantity inputs must be strings containing units. Range inputs must be tables
    containing exactly min and max, each a string containing units. Float fields
    accept plain integers or floats; other scalar fields require their exact type.
    Booleans are not accepted as numbers.

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
            or issubclass(expected_type, (units.Quantity, units.QuantityRange))
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
    if issubclass(expected_type, units.QuantityRange):
        return _parse_quantity_range(expected_type, value)

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


def _parse_quantity_range(
    range_type: type[units.QuantityRange[Any]],
    value: object,
) -> units.QuantityRange[Any]:
    """Parse a range table through its concrete quantity-range constructor."""
    if not isinstance(value, Mapping):
        raise TypeError("Expected a range table containing min and max.")

    if value.keys() != {"min", "max"}:
        raise ValueError("Range table must contain exactly min and max.")

    minimum = value["min"]
    maximum = value["max"]

    if not isinstance(minimum, str):
        raise TypeError("Range min must be a quantity string containing units.")
    if not isinstance(maximum, str):
        raise TypeError("Range max must be a quantity string containing units.")

    return range_type(minimum, maximum)


def _parse_identity(value: object) -> DeviceIdentity:
    """Parse an identity table independently of driver configuration."""
    if not isinstance(value, Mapping):
        raise TypeError("Device identity must be a table.")

    allowed = {"manufacturer", "model", "serial_number"}
    values: dict[str, str] = {}

    for name, field_value in value.items():
        if not isinstance(name, str):
            raise TypeError("Identity field names must be strings.")
        if name not in allowed:
            raise ValueError(f"Unsupported identity field: {name!r}.")
        if not isinstance(field_value, str):
            raise TypeError(f"Identity field {name!r} must be a string.")
        values[name] = field_value

    return DeviceIdentity(**values)
