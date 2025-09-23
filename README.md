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


## Digitizer ↔ LSM mode compatibility

**Legend:** ✓ supported now · △ possible/experimental · ✗ not recommended/unsupported · — unknown  

| Digitizer (vendor/model family) | Galvo–Galvo **analog** | Galvo–Galvo **photon counting** | Resonant–Galvo **analog** | Polygon–Galvo **analog** |
|---|:---:|:---:|:---:|:---:|
| **NI X-Series** (e.g., PCIe-63xx) | ✓* | ✓ (up to 4 chan.) | ✗† | ✗† |
| **NI S-Series** (e.g., PCI-6110/6115) | ✓ | ✓ (up to 2 chan.) | △§ | △§ |
| **AlazarTech** (e.g., ATS9440) | ✓ | ✗ | ✓ | ✓ |
| **Other / custom** (contact [TDW](https://github.com/tweber225)) | — | — | — | — |

*Notes*  
\* Multichannel acquisition subject to aggregate AI sample rate (e.g 2 channels: 500 kS/s, 4 channels 250 kS/s).  
† AI sample rate typically insufficient for resonant/polygon rates.  
§ Borderline: Max sample rate may limit pixels per line, dependent on scanner frequency. Not yet validated.  
