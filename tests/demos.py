from typing import Never

from dirigo.exceptions import EndOfStream
from dirigo.hardware import Device, Hardware
from dirigo.pipeline import Pipeline
from dirigo.source import HardwareSource
from dirigo.worker import Worker


class DemoDigitizer(Device):
    def read_samples(self) -> list[float]:
        return [0.1, 0.4, 0.2]

    def close(self) -> None:
        pass


class SampleSource(HardwareSource[list[float]]):
    def __init__(self, *, digitizer: str) -> None:
        super().__init__()
        self._digitizer_name = digitizer

    def _prepare(self) -> None:
        self._digitizer = self._hardware.require(
            self._digitizer_name,
            DemoDigitizer,
        )

    def _work(self) -> None:
        for _ in range(3):
            self._publish(self._digitizer.read_samples())

    def _close(self) -> None:
        # Release any acquisition-specific resources here.
        pass


class PrintSamples(Worker[list[float], Never]):
    def _prepare(self) -> None:
        pass

    def _work(self) -> None:
        while True:
            try:
                with self._receive() as product:
                    print(product.payload)
            except EndOfStream:
                return

    def _close(self) -> None:
        pass


hardware = Hardware({"primary_digitizer": DemoDigitizer()})

source = SampleSource(digitizer="primary_digitizer")
sink = PrintSamples(upstream=source)

pipeline = Pipeline(sink, hardware=hardware)

try:
    pipeline.start()
    pipeline.wait()
finally:
    hardware.close()
