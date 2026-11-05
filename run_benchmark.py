from pathlib import Path

from benchmarks.tiff import (
    TiffBenchmarkConfig,
    format_result,
    run_benchmark,
)


def main() -> None:
    config = TiffBenchmarkConfig(
        width=1024,
        height=1024,
        channels=2,
        dtype="uint16",
        interpretation="multichannel",
        fps=30,
        duration=4,
        buffers=16,
        contiguous=True,
    )
    directory = Path("C:/recording-test")

    capacity = run_benchmark(config, directory=directory, mode="capacity")
    print(format_result(capacity, config))

    paced = run_benchmark(config, directory=directory, mode="paced")
    print(format_result(paced, config))


if __name__ == "__main__":
    main()
