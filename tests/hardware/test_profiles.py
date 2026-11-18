"""Test digitizer-profile parsing independently of hardware drivers."""

from pathlib import Path

import pytest

from dirigo import units
from dirigo.hardware.digitizer import (
    ChannelCoupling,
    ExternalTriggerCoupling,
    ImpedanceMode,
    SampleClockEdge,
    SampleClockSource,
    TriggerSlope,
    TriggerSource,
)
from dirigo.hardware.profiles import load_digitizer_profile

_PROFILE_TOML = """
channels = [
    {enabled = false},
    {enabled = false},
    {enabled = true, inverted = false, coupling = "DC", impedance = "50 ohm", input_range = "±2 V"},
    {enabled = true, inverted = false, coupling = "DC", impedance = "50 ohm", input_range = "±2 V"},
]

[sample_clock]
source = "External"
rate = "100.801 MS/s"
edge = "Rising"

[trigger]
source = "External"
slope = "Rising"
external_coupling = "DC"
external_impedance = "high"
external_range = {min = "0 V", max = "3.3 V"}
"""  # ruff: ignore[line-too-long]


def test_load_profile_parses_channels_clock_and_trigger(
    tmp_path: Path,
) -> None:
    """Parse representative stored settings without initializing hardware."""
    path = tmp_path / "profile.toml"
    path.write_text(_PROFILE_TOML, encoding="utf-8")

    profile = load_digitizer_profile(path)

    assert isinstance(profile.channels, tuple)
    assert tuple(channel.enabled for channel in profile.channels) == (
        False,
        False,
        True,
        True,
    )

    channel = profile.channels[2]
    assert channel.coupling is ChannelCoupling.DC
    assert channel.impedance == units.Resistance("50 ohm")
    assert channel.input_range == units.VoltageRange("±2 V")
    assert channel.inverted is False

    clock = profile.sample_clock
    assert clock.source is SampleClockSource.EXTERNAL
    assert clock.rate == units.SampleRate("100.801 MS/s")
    assert clock.edge is SampleClockEdge.RISING

    trigger = profile.trigger
    assert trigger.source is TriggerSource.EXTERNAL
    assert trigger.slope is TriggerSlope.RISING
    assert trigger.external_coupling is ExternalTriggerCoupling.DC
    assert trigger.external_impedance is ImpedanceMode.HIGH
    assert trigger.external_range == units.VoltageRange("0 V", "3.3 V")


@pytest.mark.parametrize(
    ("original", "replacement", "error", "message"),
    [
        (
            'source = "External"',
            'source = "unknown"',
            ValueError,
            "Expected one of",
        ),
        (
            'rate = "100.801 MS/s"',
            "rate = 100801000",
            TypeError,
            "quantity string",
        ),
        (
            'rate = "100.801 MS/s"',
            'rate = "0 S/s"',
            ValueError,
            "finite and positive",
        ),
        (
            "enabled = true",
            'enabled = "true"',
            TypeError,
            "boolean",
        ),
        (
            "enabled = true",
            "enabled = false",
            ValueError,
            "At least one",
        ),
        (
            "channels =",
            "unexpected = true\nchannels =",
            ValueError,
            "Unknown DigitizerProfile fields",
        ),
        (
            'slope = "Rising"',
            'slope = "Rising"\nunexpected = true',
            ValueError,
            "Unknown TriggerProfile fields",
        ),
    ],
)
def test_load_profile_rejects_invalid_settings(
    tmp_path: Path,
    original: str,
    replacement: str,
    error: type[Exception],
    message: str,
) -> None:
    """Reject malformed values and unknown fields before driver use."""
    path = tmp_path / "profile.toml"
    path.write_text(
        _PROFILE_TOML.replace(original, replacement),
        encoding="utf-8",
    )

    with pytest.raises(error, match=message):
        load_digitizer_profile(path)
