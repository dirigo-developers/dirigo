"""Own the configured devices and synchronization backends of an instrument."""

from collections.abc import Mapping

from dirigo.hardware.device import Device
from dirigo.hardware.synchronization import SyncBackend
from dirigo.optics import LaserScanningOptics
from dirigo.resonant_raster import ResonantRasterSettings


class System:
    """Own fixed collections of named devices and synchronization backends.

    Construction copies the mappings without copying their components.
    Multiple names may refer to the same component.

    Successful construction transfers responsibility for closing components
    to this System. Failed validation leaves ownership with the caller.

    Lookups return borrowed references. The System must outlive every worker
    and session using its components. Lookup does not reserve a component or
    coordinate concurrent use.

    Close synchronization backends before devices. Within each collection,
    close in reverse insertion order, skipping repeated instances.
    """

    def __init__(
        self,
        devices: Mapping[str, Device],
        *,
        syncs: Mapping[str, SyncBackend] | None = None,
        optics: LaserScanningOptics | None = None,
        resonant_raster: ResonantRasterSettings | None = None,
    ) -> None:
        """Validate and take ownership of the supplied components."""
        configured_devices = dict(devices)
        configured_syncs = {} if syncs is None else dict(syncs)

        self._validate_components(
            configured_devices,
            Device,
            category="Device",
        )
        self._validate_components(
            configured_syncs,
            SyncBackend,
            category="Synchronization backend",
        )
        if optics is not None and not isinstance(optics, LaserScanningOptics):
            raise TypeError("optics must be LaserScanningOptics or None.")
        if resonant_raster is not None and not isinstance(
            resonant_raster,
            ResonantRasterSettings,
        ):
            raise TypeError("resonant_raster must be ResonantRasterSettings or None.")

        self._devices = configured_devices
        self._syncs = configured_syncs
        self._optics = optics
        self._resonant_raster = resonant_raster
        self._closed = False

    @staticmethod
    def _validate_components(
        components: Mapping[str, object],
        interface: type[Device] | type[SyncBackend],
        *,
        category: str,
    ) -> None:
        """Validate names and component types without changing ownership."""
        for name, component in components.items():
            if not isinstance(name, str):
                raise TypeError(f"{category} names must be strings.")

            if not name:
                raise ValueError(f"{category} names must not be empty.")

            if not isinstance(component, interface):
                raise TypeError(
                    f"{category} {name!r} must be a {interface.__name__} instance."
                )

    @property
    def device_names(self) -> tuple[str, ...]:
        """Configured device names in insertion order, including aliases."""
        return tuple(self._devices)

    @property
    def sync_names(self) -> tuple[str, ...]:
        """Configured synchronization names in insertion order."""
        return tuple(self._syncs)

    def require_device[T: Device](
        self,
        name: str,
        interface: type[T],
    ) -> T:
        """Borrow a named device implementing the requested interface.

        Raise KeyError for a missing name, TypeError for an interface
        mismatch, or RuntimeError if this System is closed.
        """
        self._require_open()
        device = self._devices[name]

        if not isinstance(device, interface):
            raise TypeError(
                f"Device {name!r} must implement {interface.__name__}; "
                f"got {type(device).__name__}."
            )

        return device

    def require_sync[T: SyncBackend](
        self,
        name: str,
        interface: type[T],
    ) -> T:
        """Borrow a named synchronization backend of the requested type.

        Raise KeyError for a missing name, TypeError for an interface
        mismatch, or RuntimeError if this System is closed.
        """
        self._require_open()
        backend = self._syncs[name]

        if not isinstance(backend, interface):
            raise TypeError(
                f"Synchronization backend {name!r} must implement "
                f"{interface.__name__}; got {type(backend).__name__}."
            )

        return backend

    def require_optics(self) -> LaserScanningOptics:
        """Return configured laser-scanning optics.

        Raise KeyError when optical calibration is absent, or RuntimeError
        after the System has closed.
        """
        self._require_open()

        if self._optics is None:
            raise KeyError("optics")

        return self._optics

    def require_resonant_raster(self) -> ResonantRasterSettings:
        """Return installed resonant-raster settings."""
        self._require_open()

        if self._resonant_raster is None:
            raise KeyError("resonant_raster")

        return self._resonant_raster

    def _require_open(self) -> None:
        """Reject component lookup after closure."""
        if self._closed:
            raise RuntimeError("System is closed.")

    def close(self) -> None:
        """Attempt cleanup of every distinct component.

        Close synchronization backends first, then devices. Both collections
        close in reverse insertion order. Attempt all cleanup even if an
        earlier component raises.

        Closure is permanent and idempotent. A cleanup failure does not
        establish that every underlying resource was successfully released.

        All borrowing workers and sessions must have finished first.
        """
        if self._closed:
            return

        self._closed = True
        closed_ids: set[int] = set()
        failures: list[BaseException] = []

        components = (
            *self._devices.values(),
            *self._syncs.values(),
        )

        for component in reversed(components):
            component_id = id(component)

            if component_id in closed_ids:
                continue

            closed_ids.add(component_id)

            try:
                component.close()
            except BaseException as exc:
                failures.append(exc)

        if failures:
            raise BaseExceptionGroup(
                "System cleanup failed.",
                failures,
            )
