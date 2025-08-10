# Dirigo
**Dirigo** is an extensible, high-performance backend for scientific image acquisition, designed with high-speed laser scanning microscopy in mind but adaptable to a wide range of medium- to high-complexity imaging systems.

Dirigo separates hardware control, acquisition logic, and user interface, making it easy to:

- Add new hardware via plugin drivers that implement generic device interfaces (digitizers, scanners, stages, cameras, etc.)

- Reconfigure data acquisition and processing pipelines using a Worker/publisher–subscriber model

- Integrate adaptive acquisition strategies through feedback loops between processing and control

- Build custom GUIs or integrate with existing tools via a clean API (a reference GUI is available as a [separate package](https://github.com/dirigo-developers/dirigo-gui))

Performance-critical operations are accelerated with [Numba](https://numba.pydata.org/) JIT compilation, releasing the GIL during execution and enabling parallel/vectorized processing.

Dirigo follows a modular, package-oriented architecture: almost all components—hardware drivers, processing modules, GUIs—are separate Python packages that can be developed, installed, and updated independently.

Dirigo is in *very* early development. While the API and architecture are functional, documentation and ready-to-use releases are in progress.
