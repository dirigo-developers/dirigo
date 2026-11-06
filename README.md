# Dirigo

Dirigo is an extensible Python backend for scientific image acquisition,
developed with laser scanning microscopy in mind.

Version 0.4 is an ongoing rewrite. It establishes contracts for hardware
control, threaded processing pipelines, and recording. Interfaces may change
as acquisition workflows are developed.

## Current scope

The current implementation includes:

- Threaded worker pipelines with bounded delivery queues, cancellation,
  failure reporting, and resource cleanup.
- Reusable output buffers and metadata carried alongside product payloads.
- Hardware ownership, TOML configuration loading, and digitizer interfaces.
- Physical quantities and ranges.
- TIFF recording for monochrome, multichannel, and RGB frames.
- OME-TIFF recording for finite, ordered Z-stacks.

Hardware acquisition requires compatible driver implementations and device
configuration. The synthetic recording example runs without hardware.

## Development installation

The project currently targets Windows and requires Python 3.14 or later.
Use a dedicated environment.

For example, with conda:

```powershell
conda create -n dirigo-0-4 python=3.14
conda activate dirigo-0-4
```

From the root of a checkout of the v0.4 development branch, install Dirigo
and its development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

The editable installation makes changes under `src/` available without
reinstalling the package.

## First example

Generate synthetic frames and record them to a TIFF file:

```powershell
python examples/record_synthetic_frames.py
```

The script writes `synthetic_frames.tif` in the current working directory.
Existing output files are preserved: choose another destination or remove
the previous recording before running the example again.

Read the [example walkthrough](docs/recording-synthetic-frames.md) for an
explanation of sources, products, metadata, and pipeline execution.

## Development checks

Run these commands from the repository root with the development
environment activated:

```powershell
ruff check .
ruff format --check .
ty check
python -m pytest
```

Pytest reports coverage using the configuration in `pyproject.toml`.

During development, run the tests relevant to the change. For example:

```powershell
python -m pytest tests/sinks/test_tiff_writer.py --no-cov
```

Before pushing a completed increment, run the full checks above.

## Commit hooks

Enable the configured pre-commit hooks once per local checkout:

```powershell
pre-commit install
```

Run them manually against all tracked files:

```powershell
pre-commit run --all-files
```

The hooks check Python lint, formatting, and types. They report problems
without modifying files.

The local Ty hook uses the development environment. When committing through
VS Code's Source Control panel, launch VS Code from an activated conda
environment so the Git process can find Ty.

## Continuous integration

GitHub Actions runs installation, lint, formatting, type, and test checks
on pushes and pull requests.

## Updating documentation

When changing behavior or an interface, review the affected docstrings,
examples, and guides in the same change. Document implemented behavior and
identify proposed behavior explicitly.

## License

Dirigo is distributed under the [MIT License](LICENSE).

## Funding

Development of Dirigo has been supported in part by the National Cancer
Institute of the National Institutes of Health under award numbers
K99CA307867 and R01CA249151.

The content of this repository is solely the responsibility of the authors
and does not necessarily represent the official views of the National
Institutes of Health.