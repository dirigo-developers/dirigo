"""Test stored profiles and deferred geometry using synthetic capture."""

from collections.abc import Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.capture_session import (
    CaptureState,
    DigitizerCaptureSession,
)
from dirigo.hardware.config import load_system
from dirigo.hardware.digitizer import (
    ChannelProfile,
    Digitizer,
    DigitizerProfile,
    SampleClockProfile,
    SampleClockSource,
    TriggerProfile,
    TriggerSlope,
    TriggerSource,
)
from dirigo.hardware.simulated_digitizer import (
    SimulatedDigitizer,
    SimulatedDigitizerConfig,
)


def _profile(
    rate: str = "1 kS/s",
    *,
    enabled_channel: int = 0,
) -> DigitizerProfile:
    """Select one of two synthetic inputs without analog settings."""
    return DigitizerProfile(
        sample_clock=SampleClockProfile(
            source=SampleClockSource.INTERNAL,
            rate=units.SampleRate(rate),
        ),
        channels=tuple(
            ChannelProfile(enabled=index == enabled_channel) for index in range(2)
        ),
        trigger=TriggerProfile(
            source=TriggerSource.INTERNAL,
            slope=TriggerSlope.RISING,
        ),
    )


@pytest.fixture
def digitizer() -> Iterator[SimulatedDigitizer]:
    """Own a synthetic digitizer with an initial stored profile."""
    device = SimulatedDigitizer(SimulatedDigitizerConfig())
    device.profile = _profile()

    try:
        yield device
    finally:
        device.close()


def _configure(
    capture: DigitizerCaptureSession,
    *,
    record_length: int = 4,
    records_per_batch: int = 2,
    batch_count: int | None = 2,
    timestamps_enabled: bool = False,
) -> None:
    """Supply complete geometry independently of stored profile settings."""
    capture.configure(
        record_length=record_length,
        records_per_batch=records_per_batch,
        batch_count=batch_count,
        timestamps_enabled=timestamps_enabled,
    )


def test_profile_capture_requires_a_stored_profile() -> None:
    """A missing profile does not prevent subsequently configuring the device."""
    with closing(SimulatedDigitizer(SimulatedDigitizerConfig())) as digitizer:
        with pytest.raises(RuntimeError, match="No digitizer profile"):
            digitizer.create_capture()

        digitizer.profile = _profile()

        with closing(digitizer.create_capture()) as capture:
            assert capture.requested_sample_rate == units.SampleRate("1 kS/s")


def test_capture_retains_its_profile_snapshot(
    digitizer: SimulatedDigitizer,
) -> None:
    """Replacing device defaults affects future captures only."""
    original = digitizer.profile

    with closing(digitizer.create_capture()) as first:
        replacement = _profile("2 kS/s", enabled_channel=1)
        digitizer.profile = replacement

        with closing(digitizer.create_capture()) as second:
            assert first.profile is original
            assert second.profile is replacement

            assert first.requested_sample_rate == units.SampleRate("1 kS/s")
            assert second.requested_sample_rate == units.SampleRate("2 kS/s")

            _configure(first)
            _configure(second)

            assert first.settings.channel_indices == (0,)
            assert second.settings.channel_indices == (1,)

            # Creating and configuring second did not reserve the device.
            first.prepare()
            assert first.layout.channel_indices == (0,)

        # Closing an unprepared second session did not disturb first.
        assert first.state is CaptureState.PREPARED


def test_prepare_requires_geometry_and_allows_recovery(
    digitizer: SimulatedDigitizer,
) -> None:
    """Premature preparation leaves the session available for configuration."""
    with closing(digitizer.create_capture()) as capture:
        with pytest.raises(RuntimeError, match="must be configured"):
            capture.prepare()

        assert capture.state is CaptureState.NEW

        _configure(capture)
        capture.prepare()

        assert capture.state is CaptureState.PREPARED
        assert capture.layout.shape == (2, 4, 1)


def test_failed_configuration_preserves_previous_request(
    digitizer: SimulatedDigitizer,
) -> None:
    """Invalid replacement geometry does not destroy valid configuration."""
    with closing(digitizer.create_capture()) as capture:
        _configure(capture, record_length=4)
        previous = capture.settings

        with pytest.raises(ValueError, match="record_length"):
            _configure(capture, record_length=0)

        assert capture.settings is previous
        assert capture.state is CaptureState.NEW

        capture.prepare()
        assert capture.layout.shape == (2, 4, 1)


def test_geometry_can_be_replaced_before_preparation(
    digitizer: SimulatedDigitizer,
) -> None:
    """The latest successful configuration determines layout and delivery."""
    with closing(digitizer.create_capture()) as capture:
        _configure(capture, record_length=4)

        _configure(
            capture,
            record_length=8,
            records_per_batch=3,
            batch_count=1,
        )
        capture.prepare()

        assert capture.layout.shape == (3, 8, 1)
        assert capture.batch_count == 1


def test_geometry_is_locked_after_preparation(
    digitizer: SimulatedDigitizer,
) -> None:
    """Reject geometry changes without invalidating the prepared capture."""
    with closing(digitizer.create_capture()) as capture:
        _configure(capture)
        capture.prepare()

        with pytest.raises(RuntimeError, match="must be new"):
            _configure(capture, record_length=8)

        assert capture.state is CaptureState.PREPARED
        assert capture.layout.shape == (2, 4, 1)


def test_profile_changes_are_locked_while_capture_is_reserved(
    digitizer: SimulatedDigitizer,
) -> None:
    """Retain device defaults until the prepared session releases ownership."""
    original = digitizer.profile
    replacement = _profile("2 kS/s")

    with closing(digitizer.create_capture()) as capture:
        _configure(capture)
        capture.prepare()

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.profile = replacement

        assert digitizer.profile is original

    digitizer.profile = replacement
    assert digitizer.profile is replacement


def test_profile_capture_delivers_data_timestamps_and_eof(
    digitizer: SimulatedDigitizer,
) -> None:
    """Combine stored sampling settings with deferred geometry during readout."""
    with closing(digitizer.create_capture()) as capture:
        _configure(
            capture,
            record_length=4,
            records_per_batch=2,
            batch_count=2,
            timestamps_enabled=True,
        )
        capture.prepare()
        capture.start()

        samples = np.empty(
            capture.layout.shape,
            dtype=capture.layout.dtype,
        )

        first = capture.read_into(samples, timeout=0)
        assert first is not None

        np.testing.assert_array_equal(
            samples[:, :, 0],
            [
                [-32768, -32767, -32766, -32765],
                [-32764, -32763, -32762, -32761],
            ],
        )

        first_times = first.timestamps
        assert first_times is not None
        np.testing.assert_allclose(first_times, [0.0, 0.004])

        second = capture.read_into(samples, timeout=0)
        assert second is not None

        np.testing.assert_array_equal(
            samples[:, :, 0],
            [
                [-32760, -32759, -32758, -32757],
                [-32756, -32755, -32754, -32753],
            ],
        )

        second_times = second.timestamps
        assert second_times is not None
        np.testing.assert_allclose(second_times, [0.008, 0.012])

        assert capture.read_into(samples, timeout=0) is None
        assert capture.batches_acquired == 2
        assert capture.state is CaptureState.COMPLETED


def test_explicit_settings_path_remains_independent_of_stored_profile(
    digitizer: SimulatedDigitizer,
) -> None:
    """Preserve complete-request capture while the acquisition API migrates."""
    settings = CaptureSettings(
        sample_rate=units.SampleRate("2 kS/s"),
        channel_indices=(1,),
        record_length=3,
        records_per_batch=1,
        batch_count=1,
    )

    with closing(digitizer.create_capture(settings)) as capture:
        assert capture.profile is None
        assert capture.settings is settings
        assert capture.requested_sample_rate == units.SampleRate("2 kS/s")

        with pytest.raises(RuntimeError, match="stored profile"):
            _configure(capture)

        capture.prepare()

        assert capture.layout.shape == (1, 3, 1)
        assert capture.layout.channel_indices == (1,)


def test_unsupported_profile_settings_fail_before_reservation(
    digitizer: SimulatedDigitizer,
) -> None:
    """Reject unsupported analog settings without reserving the input."""
    profile = digitizer.profile
    digitizer.profile = replace(
        profile,
        channels=(
            ChannelProfile(
                enabled=True,
                input_range=units.VoltageRange("±2 V"),
            ),
            ChannelProfile(enabled=False),
        ),
    )

    with closing(digitizer.create_capture()) as capture:
        with pytest.raises(NotImplementedError, match="analog channel"):
            _configure(capture)

        assert capture.state is CaptureState.NEW

        # Configuration failure did not acquire a device reservation.
        digitizer.profile = _profile()


def test_system_loads_relative_profile_without_rereading_it(
    tmp_path: Path,
) -> None:
    """Resolve profile paths relative to system configuration and retain data."""
    profile_directory = tmp_path / "profiles"
    profile_directory.mkdir()

    profile_path = profile_directory / "default.toml"
    profile_path.write_text(
        """
channels = [
    {enabled = true},
    {enabled = false},
]

[sample_clock]
source = "internal"
rate = "1 kS/s"

[trigger]
source = "internal"
slope = "rising"
""",
        encoding="utf-8",
    )

    system_path = tmp_path / "system.toml"
    system_path.write_text(
        """
[devices.digitizer]
category = "digitizers"
driver = "simulated"
profile = "profiles/default.toml"

[devices.digitizer.config]
channel_count = 2
""",
        encoding="utf-8",
    )

    with closing(
        load_system(
            system_path,
            drivers={
                "digitizers": {
                    "simulated": SimulatedDigitizer,
                },
            },
            sync_backends={},
        )
    ) as system:
        digitizer = system.require_device("digitizer", Digitizer)

        # Future capture creation must use the parsed profile, not the file.
        profile_path.write_text("invalid TOML [", encoding="utf-8")

        with closing(digitizer.create_capture()) as capture:
            assert capture.requested_sample_rate == units.SampleRate("1 kS/s")

            _configure(capture)
            capture.prepare()

            assert capture.layout.shape == (2, 4, 1)


def test_record_constraints_can_be_resolved_before_geometry(
    digitizer: SimulatedDigitizer,
) -> None:
    """Reserve during planning and reuse that reservation during preparation."""
    with closing(digitizer.create_capture()) as capture:
        assert capture.resolve_record_length_constraints() == (1, 1)
        assert capture.state is CaptureState.NEW

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.profile = _profile("2 kS/s")

        _configure(capture)
        capture.prepare()

        assert capture.state is CaptureState.PREPARED

    digitizer.profile = _profile("2 kS/s")
