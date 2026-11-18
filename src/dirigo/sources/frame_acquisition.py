"""Acquire raw batches using an explicitly selected sequence factory."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np

from dirigo import units
from dirigo.hardware.acquisition import (
    FrameAcquisitionSequence,
    FrameSequenceFactory,
    load_frame_sequence_factory,
)
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.config import parse_config
from dirigo.hardware.digitizer import Digitizer
from dirigo.hardware.frame_plan import RecordFramePlan
from dirigo.hardware.scanner import BeamScanner
from dirigo.hardware.waveform import AngularWaveform
from dirigo.pipeline.product import Metadata, Product
from dirigo.pipeline.worker import Worker
from dirigo.records import RecordBatch
from dirigo.system import System


def create_raster_sync_session(
    fast_scanner: BeamScanner, slow_scanner: BeamScanner, digitizer: Digitizer
) -> None:
    """todo"""


@dataclass(frozen=True, slots=True, kw_only=True)
class RasterFrameAcquisitionSpec:
    pixel_width: units.Length
    pixel_height: units.Length
    frame_width: units.Length
    frame_height: units.Length
    bidirectional: bool
    frames_per_acquisition: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.pixel_width, units.Length):
            raise TypeError("pixel_width must be a Length")
        if float(self.pixel_width) <= 0:
            raise ValueError("pixel_width must be positive")

        if not isinstance(self.pixel_height, units.Length):
            raise TypeError("pixel_height must be a Length")
        if float(self.pixel_height) <= 0:
            raise ValueError("pixel_height must be positive")

        if not isinstance(self.frame_width, units.Length):
            raise TypeError("frame_width must be a Length")
        if float(self.frame_width) <= 0:
            raise ValueError("frame_width must be positive")

        if not isinstance(self.frame_height, units.Length):
            raise TypeError("frame_height must be a Length")
        if float(self.frame_height) <= 0:
            raise ValueError("frame_height must be positive")

        if not isinstance(self.bidirectional, bool):
            raise TypeError("bidirectional must be a bool")

        if self.frames_per_acquisition is not None:
            if type(self.frames_per_acquisition) is not int:
                raise TypeError("frames_per_acquisition must be an int or None")
            if self.frames_per_acquisition < 1:
                raise ValueError("frames_per_acquisition must be positive")


# class RasterFrameAcquisition(Worker[Never, RecordBatch]):
#     def __init__(
#         self, spec: RasterFrameAcquisitionSpec, *, name: str | None = None
#     ) -> None:
#         """Raster-scanned acquisition worker."""
#         if not isinstance(spec, RasterFrameAcquisitionSpec):
#             raise TypeError("spec must be RasterFrameAcquisitionSpec")

#         super().__init__(name=name)
#         self._spec = spec

#     def _prepare(self) -> None:
#         hw = self._hardware
#         fast_scanner = hw.require("fast_scanner", BeamScanner)
#         slow_scanner = hw.require("slow_scanner", BeamScanner)
#         digitizer = hw.require("digitizer", Digitizer)

#         self._sync = create_raster_sync_session(
#             fast_scanner=fast_scanner,
#             slow_scanner=slow_scanner,
#         )

#         self._capture = digitizer.create_capture()

#         # allocate pool

#     def _work(self) -> None:
#         # start the sequence
#         # read into & publish
#         pass

#     def _close(self) -> None:
#         """Close the sequence"""
#         pass


class FrameAcquisitionWorker(Worker[object, RecordBatch]):
    def __init__(
        self,
        plan: RecordFramePlan,
        *,
        system: System,
        sequence: str,
        waveforms: Mapping[str, AngularWaveform],
        sequence_config: Mapping[str, object] | None = None,
        factories: Mapping[str, FrameSequenceFactory[Any]] | None = None,
        read_timeout: float = 0.05,
        name: str | None = None,
    ) -> None:
        if not isinstance(system, System):
            raise TypeError("system must be a System.")
        if not isinstance(plan, RecordFramePlan):
            raise TypeError("plan must be a RecordFramePlan.")
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("sequence must be a nonempty string.")
        if isinstance(read_timeout, bool) or not isinstance(read_timeout, Real):
            raise TypeError("read_timeout must be a real number of seconds.")
        if not math.isfinite(read_timeout) or read_timeout <= 0:
            raise ValueError("read_timeout must be finite and positive.")

        super().__init__(name=name)
        self._system = system
        self._plan = plan
        self._sequence_name = sequence
        self._waveforms = dict(waveforms)
        self._sequence_config = (
            dict(sequence_config) if sequence_config is not None else {}
        )
        self._factories = dict(factories) if factories is not None else None
        self._read_timeout = float(read_timeout)
        self._sequence: FrameAcquisitionSequence | None = None

    def _prepare(self) -> None:
        """Select, configure, and prepare the sequence before workers start."""
        factory = load_frame_sequence_factory(
            self._sequence_name,
            factories=self._factories,
        )
        config = parse_config(
            factory.config_type,
            self._sequence_config,
        )
        sequence = factory(
            config,
            system=self._system,
            plan=self._plan,
            waveforms=self._waveforms,
        )
        self._sequence = sequence
        sequence.prepare()

        if (
            sequence.layout.records_per_batch
            != self._plan.capture_settings.records_per_batch
        ):
            raise ValueError("Sequence batch layout does not match the frame plan.")

    def _work(self) -> None:
        """Start the prepared run, publish batches, and observe coordinated EOF."""
        sequence = self._sequence
        assert sequence is not None

        if self._abort_requested.is_set():
            return
        sequence.start()

        layout = sequence.layout
        first_record = 0
        batch_index = 0

        while not self._abort_requested.is_set():
            samples = np.empty(layout.shape, dtype=layout.dtype)

            while not self._abort_requested.is_set():
                try:
                    info = sequence.read_into(
                        samples,
                        timeout=self._read_timeout,
                    )
                except TimeoutError:
                    if sequence.state is CaptureState.CLOSED:
                        raise
                    continue

                if info is None:
                    if first_record != self._plan.record_count:
                        raise RuntimeError(
                            "Sequence ended before the planned record count."
                        )
                    return

                if self._abort_requested.is_set():
                    return

                if first_record + layout.records_per_batch > self._plan.record_count:
                    raise RuntimeError("Sequence exceeded the planned record count.")

                samples.setflags(write=False)
                self._publish(
                    Product(
                        RecordBatch(
                            samples=samples,
                            layout=layout,
                            info=info,
                            first_record=first_record,
                            records_per_frame=self._plan.records_per_frame,
                        ),
                        metadata=Metadata(sequence_index=batch_index),
                    )
                )

                first_record += layout.records_per_batch
                batch_index += 1
                break

    def _close(self) -> None:
        """Close the owned sequence, including after failed preparation."""
        sequence = self._sequence
        self._sequence = None
        if sequence is not None:
            sequence.close()
