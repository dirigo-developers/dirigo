"""Parse and validate stored digitizer profiles."""

import math
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from enum import StrEnum
from os import PathLike
from pathlib import Path

from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.digitizer import (
    ChannelCoupling,
    ChannelProfile,
    DigitizerProfile,
    ExternalTriggerCoupling,
    ExternalTriggerRange,
    ImpedanceMode,
    SampleClockEdge,
    SampleClockProfile,
    SampleClockSource,
    TriggerProfile,
    TriggerSlope,
    TriggerSource,
)
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    ExternalSampleClock,
    SignalEdge,
)


def _parse_fields[T](
    value: object,
    model: type[T],
    converters: Mapping[str, Callable[[object], object]],
) -> T:
    """Parse one profile table, rejecting unknown fields."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{model.__name__} must be a table.")

    if any(not isinstance(name, str) for name in value):
        raise TypeError("Profile field names must be strings.")

    if not is_dataclass(model):
        raise TypeError("model must be a dataclass type.")

    allowed = {item.name for item in fields(model)}
    unknown = value.keys() - allowed

    if unknown:
        raise ValueError(
            f"Unknown {model.__name__} fields: {', '.join(sorted(unknown))}."
        )

    parsed: dict[str, object] = {}

    for name, raw in value.items():
        try:
            parsed[name] = converters[name](raw)
        except (TypeError, ValueError) as failure:
            failure.add_note(f"While parsing {model.__name__}.{name}.")
            raise

    return model(**parsed)


def _parse_enum[T: StrEnum](enum_type: type[T], value: object) -> T:
    """Parse an enum value without making capitalization significant."""
    if not isinstance(value, str):
        raise TypeError("Expected a string.")

    normalized = value.strip().casefold()

    for member in enum_type:
        if member.value.casefold() == normalized:
            return member

    choices = ", ".join(member.value for member in enum_type)
    raise ValueError(f"Expected one of: {choices}.")


def _parse_bool(value: object) -> bool:
    """Require an actual TOML boolean."""
    if type(value) is not bool:
        raise TypeError("Expected a boolean.")
    return value


def _parse_quantity[T: units.Quantity](
    quantity_type: type[T],
    value: object,
) -> T:
    """Require a quantity string containing units."""
    if not isinstance(value, str):
        raise TypeError("Expected a quantity string containing units.")
    return quantity_type(value)


def _parse_voltage_range(value: object) -> units.VoltageRange:
    """Accept a symmetric range string or an explicit min/max table."""
    if isinstance(value, str):
        return units.VoltageRange(value)

    if not isinstance(value, Mapping) or value.keys() != {"min", "max"}:
        raise TypeError("Expected a symmetric voltage range string or a min/max table.")

    minimum = value["min"]
    maximum = value["max"]

    if not isinstance(minimum, str) or not isinstance(maximum, str):
        raise TypeError("Voltage range endpoints must contain units.")

    return units.VoltageRange(minimum, maximum)


def _parse_impedance(
    value: object,
) -> units.Resistance | ImpedanceMode:
    """Accept a resistance or the named high-impedance mode."""
    if isinstance(value, str) and value.strip().casefold() == "high":
        return ImpedanceMode.HIGH

    return _parse_quantity(units.Resistance, value)


def _parse_trigger_range(
    value: object,
) -> units.VoltageRange | ExternalTriggerRange:
    """Accept a voltage interval or the named TTL mode."""
    if isinstance(value, str) and value.strip().casefold() == "ttl":
        return ExternalTriggerRange.TTL

    return _parse_voltage_range(value)


def _parse_channel(value: object) -> ChannelProfile:
    """Parse one physical channel's settings."""
    return _parse_fields(
        value,
        ChannelProfile,
        {
            "enabled": _parse_bool,
            "inverted": _parse_bool,
            "coupling": lambda raw: _parse_enum(ChannelCoupling, raw),
            "impedance": _parse_impedance,
            "input_range": _parse_voltage_range,
            "offset": lambda raw: _parse_quantity(units.Voltage, raw),
        },
    )


def _parse_channels(value: object) -> tuple[ChannelProfile, ...]:
    """Convert a TOML array into immutable channel profiles."""
    if not isinstance(value, list):
        raise TypeError("channels must be an array.")

    return tuple(_parse_channel(item) for item in value)


def _parse_sample_clock(value: object) -> SampleClockProfile:
    """Parse the requested sampling clock."""
    return _parse_fields(
        value,
        SampleClockProfile,
        {
            "source": lambda raw: _parse_enum(SampleClockSource, raw),
            "rate": lambda raw: _parse_quantity(units.SampleRate, raw),
            "edge": lambda raw: _parse_enum(SampleClockEdge, raw),
        },
    )


def _parse_trigger(value: object) -> TriggerProfile:
    """Parse record-trigger defaults."""
    return _parse_fields(
        value,
        TriggerProfile,
        {
            "source": lambda raw: _parse_enum(TriggerSource, raw),
            "slope": lambda raw: _parse_enum(TriggerSlope, raw),
            "level": lambda raw: _parse_quantity(units.Voltage, raw),
            "external_coupling": lambda raw: _parse_enum(
                ExternalTriggerCoupling,
                raw,
            ),
            "external_impedance": _parse_impedance,
            "external_range": _parse_trigger_range,
        },
    )


def validate_digitizer_profile(profile: DigitizerProfile) -> None:
    """Validate programmatically constructed profiles as well as parsed ones."""
    if not isinstance(profile, DigitizerProfile):
        raise TypeError("profile must be a DigitizerProfile.")

    clock = profile.sample_clock
    if not isinstance(clock, SampleClockProfile):
        raise TypeError("sample_clock must be a SampleClockProfile.")
    if not isinstance(clock.source, SampleClockSource):
        raise TypeError("sample_clock.source must be a SampleClockSource.")
    if not isinstance(clock.edge, SampleClockEdge):
        raise TypeError("sample_clock.edge must be a SampleClockEdge.")
    if not isinstance(clock.rate, units.SampleRate):
        raise TypeError("sample_clock.rate must be a SampleRate.")
    if not math.isfinite(float(clock.rate)) or float(clock.rate) <= 0:
        raise ValueError("sample_clock.rate must be finite and positive.")

    if not isinstance(profile.channels, tuple) or not profile.channels:
        raise TypeError("channels must be a nonempty tuple.")

    for channel in profile.channels:
        if not isinstance(channel, ChannelProfile):
            raise TypeError("channels must contain ChannelProfile instances.")

        if type(channel.enabled) is not bool:
            raise TypeError("Channel enabled must be a bool.")
        if type(channel.inverted) is not bool:
            raise TypeError("Channel inverted must be a bool.")

        if channel.coupling is not None and not isinstance(
            channel.coupling,
            ChannelCoupling,
        ):
            raise TypeError("Channel coupling must be a ChannelCoupling.")

        _validate_impedance(channel.impedance)

        if channel.input_range is not None:
            _validate_voltage_range(channel.input_range)

        if channel.offset is not None:
            _validate_voltage(channel.offset)

    if not any(channel.enabled for channel in profile.channels):
        raise ValueError("At least one profile channel must be enabled.")

    trigger = profile.trigger
    if not isinstance(trigger, TriggerProfile):
        raise TypeError("trigger must be a TriggerProfile.")
    if not isinstance(trigger.source, TriggerSource):
        raise TypeError("trigger.source must be a TriggerSource.")
    if not isinstance(trigger.slope, TriggerSlope):
        raise TypeError("trigger.slope must be a TriggerSlope.")

    if trigger.level is not None:
        _validate_voltage(trigger.level)

    if trigger.external_coupling is not None and not isinstance(
        trigger.external_coupling,
        ExternalTriggerCoupling,
    ):
        raise TypeError("trigger.external_coupling must be an ExternalTriggerCoupling.")

    _validate_impedance(trigger.external_impedance)

    if isinstance(trigger.external_range, units.VoltageRange):
        _validate_voltage_range(trigger.external_range)
    elif trigger.external_range is not None and not isinstance(
        trigger.external_range,
        ExternalTriggerRange,
    ):
        raise TypeError(
            "trigger.external_range must be a VoltageRange or ExternalTriggerRange."
        )


def _validate_impedance(
    value: units.Resistance | ImpedanceMode | None,
) -> None:
    """Validate an optional resistance or named impedance mode."""
    if value is None or isinstance(value, ImpedanceMode):
        return

    if not isinstance(value, units.Resistance):
        raise TypeError("Impedance must be a Resistance or ImpedanceMode.")

    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Resistance must be finite and positive.")


def _validate_voltage(value: units.Voltage) -> None:
    """Require a finite voltage."""
    if not isinstance(value, units.Voltage):
        raise TypeError("Expected a Voltage.")
    if not math.isfinite(float(value)):
        raise ValueError("Voltage must be finite.")


def _validate_voltage_range(value: units.VoltageRange) -> None:
    """Require a voltage range with finite endpoints."""
    if not isinstance(value, units.VoltageRange):
        raise TypeError("Expected a VoltageRange.")

    if not all(math.isfinite(float(endpoint)) for endpoint in (value.min, value.max)):
        raise ValueError("Voltage range endpoints must be finite.")


def load_digitizer_profile(
    path: str | PathLike[str],
) -> DigitizerProfile:
    """Read and validate an immutable digitizer profile from TOML."""
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)

    profile = _parse_fields(
        values,
        DigitizerProfile,
        {
            "sample_clock": _parse_sample_clock,
            "channels": _parse_channels,
            "trigger": _parse_trigger,
        },
    )
    validate_digitizer_profile(profile)
    return profile


def capture_settings_from_profile(
    profile: DigitizerProfile,
    *,
    record_length: int,
    records_per_batch: int,
    batch_count: int | None,
    timestamps_enabled: bool,
    trigger_offset_samples: int,
    external_clock: ExternalSampleClock | None = None,
) -> CaptureSettings:
    """Combine profile timing with per-capture geometry.

    An external sampling clock requires an explicit receiver-side route
    supplied by the driver or its installation configuration.

    TriggerSource.INTERNAL denotes consecutive records without a separate
    per-record trigger. Other sources request independently triggered records.
    """
    validate_digitizer_profile(profile)

    clock = profile.sample_clock

    if clock.source is SampleClockSource.INTERNAL:
        if external_clock is not None:
            raise ValueError(
                "An internally clocked profile cannot use an external route."
            )
        if clock.edge is not SampleClockEdge.RISING:
            raise NotImplementedError(
                "CaptureSettings does not currently describe falling-edge "
                "internal sampling."
            )
    else:
        if external_clock is None:
            raise NotImplementedError(
                "This profile requires an external sample-clock route."
            )

        expected_edge = SignalEdge(clock.edge.value)
        if external_clock.active_edge is not expected_edge:
            raise ValueError(
                "External clock route and profile specify different edges."
            )

    trigger = profile.trigger
    record_trigger = None

    if trigger.source is not TriggerSource.INTERNAL:
        record_trigger = DigitalRecordTrigger(
            source=trigger.source.value,
            active_edge=SignalEdge(trigger.slope.value),
        )

    return CaptureSettings(
        sample_rate=clock.rate,
        channel_indices=tuple(
            index for index, channel in enumerate(profile.channels) if channel.enabled
        ),
        record_length=record_length,
        records_per_batch=records_per_batch,
        batch_count=batch_count,
        timestamps_enabled=timestamps_enabled,
        external_clock=external_clock,
        record_trigger=record_trigger,
        trigger_offset_samples=trigger_offset_samples,
    )
