"""Device identity records and conversion of selected metadata to JSON values."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from dirigo import units

type MetadataValue = (
    bool | int | float | str | list[MetadataValue] | dict[str, MetadataValue] | None
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceIdentity:
    """Driver-declared physical identity; None means unknown.

    Values may come from configuration or hardware readback. Their source
    should be documented by the driver.
    """

    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DriverIdentity:
    """Identify the implementation separately from the controlled equipment.

    implementation is the qualified Python class name. distribution and
    version identify the installed driver distribution when known.
    """

    implementation: str
    distribution: str | None = None
    version: str | None = None


class DeviceSnapshot(TypedDict):
    """A detached JSON-compatible record of identity and selected settings."""

    identity: dict[str, MetadataValue]
    driver: dict[str, MetadataValue]
    settings: dict[str, MetadataValue]


def encode_metadata(values: Mapping[str, object]) -> dict[str, MetadataValue]:
    """Copy selected metadata into JSON-compatible values.

    Encode quantities as {"value": float(quantity), "unit": quantity.base_unit}.
    Preserve ordinary scalar values, recursively copy mappings, and convert
    lists and tuples to fresh lists. Do not inspect arbitrary object attributes.

    Args:
        values: A mapping with string keys and supported, acyclic values.

    Returns:
        A detached dictionary containing JSON-compatible values.

    Raises:
        TypeError: A key is not a string or a value has an unsupported type.
        ValueError: A number or quantity is nonfinite.
    """
    result: dict[str, MetadataValue] = {}

    for name, value in values.items():
        if not isinstance(name, str):
            raise TypeError("Metadata field names must be strings.")

        try:
            result[name] = _encode_value(value)
        except (TypeError, ValueError) as exc:
            exc.add_note(f"While encoding metadata field {name!r}.")
            raise

    return result


def _encode_value(value: object) -> MetadataValue:
    """Encode one value, preserving units before handling ordinary floats."""
    if isinstance(value, units.Quantity):
        return {
            "value": _encode_value(float(value)),
            "unit": value.base_unit,
        }

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata numbers must be finite.")
        return float(value)

    if isinstance(value, Mapping):
        return encode_metadata(value)

    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]

    raise TypeError(f"Unsupported metadata value type: {type(value).__name__}.")
