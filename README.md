# Dirigo
[![PyPI](https://img.shields.io/pypi/v/dirigo)](https://pypi.org/project/dirigo/)
[![Documentation Status](https://readthedocs.org/projects/dirigo/badge/?version=latest)](https://dirigo.readthedocs.io/en/latest/?badge=latest)

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
| **Teledyne SP Devices** (e.g., ADQ32) | ✓ | ✗ | ✓ | ✓ |
| **Other / custom** (contact [TDW](https://github.com/tweber225)) | — | — | — | — |

*Notes*  
\* Multichannel acquisition subject to aggregate AI sample rate (e.g 2 channels: 500 kS/s, 4 channels 250 kS/s).  
† AI sample rate typically insufficient for resonant/polygon rates.  
§ Borderline: Max sample rate may limit pixels per line, dependent on scanner frequency. Not yet validated.  


## Installation
Dirigo should be installed in a virtual environment. We recommend [miniconda](https://docs.conda.io/en/latest/). Install Dirigo from PyPI with `pip install dirigo`. To enable additional hardware and acquisition modes, also install Dirigo plugins. A list can be found here: https://github.com/orgs/dirigo-developers/repositories
 
## Initialization
Verify installations (with miniconda, `conda list`) and run the initialization script by running `dirigo-init`. This script will set up Dirigo settings folders and files. It will not overwrite existing files.

### System configuration
The `system_config.toml` file contains a list of the available hardware, the associated Python entry point, and information required to configure each device. An example can be found here: [system_config.toml](examples/system_config.toml)

### Frame specification
Acquisition sequence variables (e.g. frame size) are described with a specification file. The initialization script will generate a `FrameAcquisition` specification file at `Dirigo/acquisitions/frame/default.toml` which should be updated before starting Dirigo.

### Digitizer profile
Groups of settable parameters for devices are group into profiles. The initialization script will generate a blank digitizer profile at `Dirigo/acquisitions/frame/default.toml` which will need to be updated. An example from the `dirigo-alazar` plugin can be found here: [default_ats9440.toml](https://github.com/dirigo-developers/dirigo-alazar/blob/main/examples/default_ats9440.toml)


## Funding

Development of Dirigo has been supported in part by the National Cancer Institute of the National Institutes of Health under award number R01CA249151.

The content of this repository is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.