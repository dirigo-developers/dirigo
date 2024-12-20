# Dirigo
Dirigo ('I direct') is a collection of [interfaces](https://en.wikipedia.org/wiki/Interface_(computing)#Software_interfaces) for customizable image data acquisition in Python.

Virtually all components are plugins which implement interfaces. Acquisition workers execute data collection logic by coordinating Resources (hardware devices, data loggers, image processing workflows, etc.).

By design, Dirigo is a [backend](https://en.wikipedia.org/wiki/Frontend_and_backend) layer only, however it provides an API for frontend applications.

## Acquisition
Built-in plugins:

- **LineAcquisition** \
Continuous or finite 1D (e.g. line scan) acquisition.

- **FrameAcquisition** \
Continuous or finite 2D (e.g. frame) acquisition


## Hardware
Defined interfaces:

- **Digitizer** \
Sequential analog to digital device, typically dedicated (high-speed) hardware

- **Stage** \
Sample translation device

- **Multi-function input/output (MFIO)** \
Device with multiple digitial and analog input and output capabilities. Well-known example: NIDAQ card

- **Raster beam scanning** \
Seperate interfaces for fast and slow axes


## Conventions
- In user-facing locations (GUIs, configuration files, etc), variables should explicitly include units when applicable. Use a string with a space between the value and unit. Examples: `rate = "100 MS/s`, `objective_focal_length = "12.5 mm"`
- In non-user-facing locations (internally within objects), variables should be stored in  SI base units (`float` or `int`). This simplifies internal calculations assuming there are no dimension errors in the calculations. Examples: 100 MS/s should be stored internally as `100e6`, 12.5 mm should be stored as `12.5e-3`