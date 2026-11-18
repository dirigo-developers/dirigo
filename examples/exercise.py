from dirigo import units
from dirigo.hardware.digitizer import (
    ChannelProfile,
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

digitizer = SimulatedDigitizer(SimulatedDigitizerConfig())

digitizer.profile = DigitizerProfile(
    sample_clock=SampleClockProfile(
        source=SampleClockSource.INTERNAL,
        rate=units.SampleRate("1 MS/s"),
    ),
    channels=(
        ChannelProfile(enabled=True),
        ChannelProfile(enabled=False),
    ),
    trigger=TriggerProfile(
        source=TriggerSource.INTERNAL,
        slope=TriggerSlope.RISING,
    ),
)

try:
    capture = digitizer.create_capture()

    try:
        assert capture.requested_sample_rate == units.SampleRate("1 MS/s")

        capture.configure(
            record_length=1024,
            records_per_batch=32,
            batch_count=10,
        )
        capture.prepare()

        # capture.layout is now available.
        # Execution will call capture.start() and capture.read_into(...).
    finally:
        capture.close()
finally:
    digitizer.close()
