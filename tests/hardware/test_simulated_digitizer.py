"""Exercise finite synthetic capture through the public device/session API."""

from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.config import load_hardware
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.simulated_digitizer import (
    SimulatedDigitizer,
    SimulatedDigitizerConfig,
)
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    DigitalStartTrigger,
    ExternalSampleClock,
)


def _settings() -> CaptureSettings:
    """Request two batches with deliberately reversed physical-channel order."""
    return CaptureSettings(
        sample_rate=units.SampleRate("1 kS/s"),
        record_length=4,
        channel_indices=(1, 0),
        records_per_batch=2,
        batch_count=2,
        timestamps_enabled=True,
    )


def test_finite_capture_preserves_order_continuity_and_timestamps() -> None:
    """Deliver continuous samples and ideal record times across two batches.

    Reversed channels distinguish physical indices from output-column
    positions. Reusing the destination verifies that the session advances
    its own position. A zero timeout demonstrates that simulation is unpaced.
    """
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())
    session = digitizer.create_capture(_settings())

    try:
        session.prepare()
        layout = session.layout
        assert layout is not None
        assert layout.shape == (2, 4, 2)
        assert layout.channel_indices == (1, 0)
        assert layout.dtype == np.dtype(np.int16)

        samples = np.empty(layout.shape, dtype=layout.dtype)
        session.start()

        first = session.read_into(samples, timeout=0)
        assert first is not None
        np.testing.assert_array_equal(
            samples[:, :, 0],
            [
                [-32511, -32510, -32509, -32508],
                [-32507, -32506, -32505, -32504],
            ],
        )
        np.testing.assert_array_equal(
            samples[:, :, 1],
            [
                [-32768, -32767, -32766, -32765],
                [-32764, -32763, -32762, -32761],
            ],
        )
        first_times = first.timestamps
        assert first_times is not None
        np.testing.assert_allclose(first_times, [0.0, 0.004])

        second = session.read_into(samples, timeout=0)
        assert second is not None
        assert samples[0, 0, 0] == -32503
        assert samples[0, 0, 1] == -32760

        second_times = second.timestamps
        assert second_times is not None
        np.testing.assert_allclose(second_times, [0.008, 0.012])
        np.testing.assert_allclose(first_times, [0.0, 0.004])

        assert session.batches_acquired == 2
        assert session.state is CaptureState.COMPLETED
        assert digitizer.is_active
        assert session.read_into(samples) is None
    finally:
        session.close()
        digitizer.close()


def test_sessions_reuse_device_and_restart_sample_position() -> None:
    """Plan future runs during a reservation and restart their synthetic data.

    Constructing a session is passive. Preparing a contender is rejected
    without disturbing the owner. A separate planned session can prepare
    after the owner closes and starts at sample zero.
    """
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())
    settings = replace(_settings(), batch_count=1)
    first = digitizer.create_capture(settings)
    first.prepare()

    planned = digitizer.create_capture(settings)
    contender = digitizer.create_capture(settings)

    try:
        with pytest.raises(RuntimeError, match="reserved"):
            contender.prepare()
        assert first.state is CaptureState.PREPARED

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.close()

        first.start()
        samples = np.empty((2, 4, 2), dtype=np.int16)
        assert first.read_into(samples) is not None
        expected = samples.copy()
        first.close()

        planned.prepare()
        planned.start()
        info = planned.read_into(samples)
        assert info is not None

        np.testing.assert_array_equal(samples, expected)
        times = info.timestamps
        assert times is not None
        np.testing.assert_allclose(times, [0.0, 0.004])
    finally:
        contender.close()
        planned.close()
        first.close()
        digitizer.close()


def test_signed_codes_wrap_and_timestamps_can_be_disabled() -> None:
    """Wrap at the int16 code boundary without implicit timestamps."""
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())
    session = digitizer.create_capture(
        CaptureSettings(
            sample_rate=units.SampleRate("1 kS/s"),
            record_length=65538,
        )
    )

    try:
        session.prepare()
        session.start()
        samples = np.empty((1, 65538, 1), dtype=np.int16)

        info = session.read_into(samples)
        assert info is not None
        assert info.timestamps is None
        np.testing.assert_array_equal(
            samples[0, 65534:, 0],
            [32766, 32767, -32768, -32767],
        )
    finally:
        session.close()
        digitizer.close()


@pytest.mark.parametrize("channel_count", [True, 1.5, 0, -1])
def test_invalid_simulator_config(channel_count: object) -> None:
    """Reject noninteger or nonpositive physical-channel counts."""
    error_type = TypeError if isinstance(channel_count, (bool, float)) else ValueError

    with pytest.raises(error_type, match="channel_count"):
        SimulatedDigitizerConfig(channel_count=cast(int, channel_count))


def test_factory_validation_and_closed_device() -> None:
    """Reject invalid requests without consuming a device reservation."""
    with pytest.raises(TypeError, match="config"):
        SimulatedDigitizer(cast(SimulatedDigitizerConfig, object()))

    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())

    with pytest.raises(TypeError, match="settings"):
        digitizer.create_capture(cast(CaptureSettings, object()))

    with pytest.raises(ValueError, match="channel_indices"):
        digitizer.create_capture(replace(_settings(), channel_indices=(2,)))

    assert not digitizer.is_active
    digitizer.close()

    with pytest.raises(RuntimeError, match="closed"):
        digitizer.create_capture(_settings())


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_count": None},
        {
            "external_clock": ExternalSampleClock(
                source="PFI0",
                max_rate=units.SampleRate("1 kS/s"),
            )
        },
        {"start_trigger": DigitalStartTrigger(source="PFI1")},
    ],
)
def test_unsupported_capture_modes(changes: dict[str, object]) -> None:
    """Reject unsupported modes during passive session construction."""
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())

    try:
        with pytest.raises(NotImplementedError):
            digitizer.create_capture(replace(_settings(), **changes))
        assert not digitizer.is_active
    finally:
        digitizer.close()


def test_simulator_snapshot_describes_storage_and_execution() -> None:
    """Advertise synthetic capabilities through normal device metadata."""
    config = SimulatedDigitizerConfig(channel_count=3)
    digitizer = SimulatedDigitizer(config)

    try:
        assert digitizer.config is config
        assert digitizer.data_range == units.IntRange(-32768, 32767)
        assert digitizer.snapshot()["settings"] == {
            "channel_count": 3,
            "bit_depth": 16,
            "sample_dtype": "int16",
            "simulation_mode": "unpaced",
        }
    finally:
        digitizer.close()


def test_discovery_and_loading_from_toml(tmp_path: Path) -> None:
    """Load the simulator through categorized discovery and TOML parsing.

    Check the installed entry-point target, initialization-config parsing,
    and Hardware lookup through the generic Digitizer interface. No explicit
    registry bypasses plugin discovery.
    """
    entries = importlib_metadata.entry_points(
        group="dirigo.devices.digitizers",
        name="simulated",
    )
    assert len(entries) == 1
    assert next(iter(entries)).load() is SimulatedDigitizer

    path = tmp_path / "system_config.toml"
    path.write_text(
        '[devices.adc]\ncategory = "digitizers"\ndriver = "simulated"\n'
        "[devices.adc.config]\nchannel_count = 3\n",
        encoding="utf-8",
    )

    hardware = load_hardware(path)
    try:
        device = hardware.require("adc", Digitizer)
        assert isinstance(device, SimulatedDigitizer)
        assert device.channel_count == 3
    finally:
        hardware.close()


def test_record_trigger_is_rejected_explicitly() -> None:
    """Reject event-driven capture that the current simulator cannot honor."""
    digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())
    try:
        with pytest.raises(NotImplementedError, match="record triggers"):
            digitizer.create_capture(
                replace(
                    _settings(),
                    record_trigger=DigitalRecordTrigger(source="external"),
                )
            )

        assert not digitizer.is_active
    finally:
        digitizer.close()
