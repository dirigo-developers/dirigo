from abc import ABC, abstractmethod
from collections.abc import Mapping


class Device(ABC):
    """A hardware component whose lifetime is owned by Hardware."""

    @abstractmethod
    def close(self) -> None:
        """Stop activity and release this device's owned resources.

        Repeated calls must be safe.

        Support cleanup after partial acquisition preparation or a failed
        start. Attempt remaining cleanup even if an earlier step fails.

        Do not release resources borrowed from another Device.
        """


class Hardware:
    """A fixed collection that owns its named devices.

    Workers borrow references. The collection must remain open until
    all users of its devices have finished.
    """

    def __init__(self, devices: Mapping[str, Device]) -> None:
        configured = dict(devices)

        for name, device in configured.items():
            if not isinstance(name, str):
                raise TypeError("Device names must be strings.")

            if not name:
                raise ValueError("Device names must not be empty.")

            if not isinstance(device, Device):
                raise TypeError(f"Device {name!r} must be a Device instance.")

        self._devices = configured
        self._closed = False

    @property
    def names(self) -> tuple[str, ...]:
        """Configured names in insertion order, including after closure."""
        return tuple(self._devices)

    def require[T: Device](
        self,
        name: str,
        interface: type[T],
    ) -> T:
        """Return the named device after checking its interface.

        Raises:
            RuntimeError: If the collection is closed.
            KeyError: If the name is not configured.
            TypeError: If the device does not implement the interface.
        """
        if self._closed:
            raise RuntimeError("Hardware collection is closed.")

        device = self._devices[name]

        if not isinstance(device, interface):
            raise TypeError(
                f"Device {name!r} must implement {interface.__name__}; "
                f"got {type(device).__name__}."
            )

        return device

    def close(self) -> None:
        """Close distinct devices in reverse insertion order.

        Attempt every device before reporting collected failures.
        The collection remains closed even if cleanup fails.
        Repeated calls do nothing.

        All users must have finished before calling this method.
        Must not run concurrently with lookup or device use.
        """
        if self._closed:
            return

        self._closed = True
        closed_ids: set[int] = set()
        failures: list[BaseException] = []

        for device in reversed(self._devices.values()):
            device_id = id(device)

            if device_id in closed_ids:
                continue

            closed_ids.add(device_id)

            try:
                device.close()
            except BaseException as exc:
                failures.append(exc)

        if failures:
            raise BaseExceptionGroup(
                "Hardware cleanup failed.",
                failures,
            )
