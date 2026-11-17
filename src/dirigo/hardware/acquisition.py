"""Contracts and discovery for frame acquisition sequences."""

from collections.abc import Mapping
from dataclasses import is_dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Protocol, cast

from dirigo.hardware.capture import CaptureInfo, CaptureLayout, SampleArray
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.hardware import Hardware
from dirigo.hardware.waveform import AngularWaveform

_GROUP = "dirigo.acquisition_sequences"


class FrameAcquisitionSequence(Protocol):
    """Own one finite acquisition run while borrowing its devices.

    Construction is passive. prepare() reserves resources and establishes
    layout without starting acquisition. start() arms or starts the run.

    Calls are serial but may occur on different threads: prepare() runs on
    the pipeline caller's thread; execution normally runs on a worker thread.

    read_into() fills complete batches in the requested plan's order. None
    means the complete coordinated run has finished, including its outputs.
    A retryable TimeoutError delivers no batch and preserves usability;
    destination contents are unspecified. A fatal read failure leaves state
    CLOSED. Caller validation errors must preserve an otherwise usable run.

    close() releases all owned resources, including after failed preparation.
    It is permanent and idempotent and never closes borrowed devices.
    """

    @property
    def state(self) -> CaptureState:
        """Current state; CLOSED distinguishes fatal from retryable timeouts."""
        ...

    @property
    def layout(self) -> CaptureLayout:
        """Prepared batch layout, retained after closure."""
        ...

    def prepare(self) -> None:
        """Prepare without starting; permit close() after partial failure."""
        ...

    def start(self) -> None:
        """Start or arm coordinated participants in dependency order."""
        ...

    def read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None = None,
    ) -> CaptureInfo | None:
        """Deliver a complete batch, raise a timeout, or return coordinated EOF."""
        ...

    def close(self) -> None:
        """Release the run once, leaving borrowed devices open."""
        ...


class FrameSequenceFactory[ConfigT](Protocol):
    """Construct a sequence from declared configuration and a run description."""

    @property
    def config_type(self) -> type[ConfigT]:
        """Configuration dataclass inspectable without creating a sequence."""
        ...

    def __call__(
        self,
        config: ConfigT,
        /,
        *,
        hardware: Hardware,
        plan: RecordFramePlan,
        waveforms: Mapping[str, AngularWaveform],
    ) -> FrameAcquisitionSequence:
        """Validate compatibility and return a new, caller-owned sequence.

        Waveform keys name scanner devices in hardware. Preserve the supplied
        capture-settings subclass. Construction performs no hardware access
        or reservation. Release partial ownership if construction raises.
        """
        ...


def discover_frame_sequences() -> dict[str, importlib_metadata.EntryPoint]:
    """Return unloaded sequence registrations, rejecting duplicate names."""
    result: dict[str, importlib_metadata.EntryPoint] = {}

    for entry in importlib_metadata.entry_points().select(group=_GROUP):
        previous = result.get(entry.name)
        if previous is not None:
            raise ValueError(
                f"Duplicate acquisition sequence {entry.name!r}: "
                f"{previous.value!r} and {entry.value!r}."
            )
        result[entry.name] = entry

    return result


def load_frame_sequence_factory(
    name: str,
    *,
    factories: Mapping[str, FrameSequenceFactory[Any]] | None = None,
) -> FrameSequenceFactory[Any]:
    """Resolve only the selected factory without constructing a sequence.

    An explicit registry, including an empty one, bypasses discovery.
    Entry points must export callable factory objects whose config_type
    attribute names a dataclass type.

    This checks declaration shape, not full protocol behavior.
    """
    choices: Mapping[str, object] = (
        discover_frame_sequences() if factories is None else factories
    )

    if name not in choices:
        available = ", ".join(sorted(choices)) or "(none)"
        raise ValueError(
            f"Unknown acquisition sequence {name!r}. Available sequences: {available}"
        )

    candidate = choices[name]

    if factories is None:
        entry = cast(importlib_metadata.EntryPoint, candidate)
        try:
            candidate = entry.load()
        except BaseException as failure:
            failure.add_note(f"While loading acquisition sequence {name!r}.")
            raise

    model = getattr(candidate, "config_type", None)
    if (
        not callable(candidate)
        or not isinstance(model, type)
        or not is_dataclass(model)
    ):
        raise TypeError(
            f"Sequence factory {name!r} must be callable and "
            "declare a dataclass config_type."
        )

    return cast(FrameSequenceFactory[Any], candidate)
