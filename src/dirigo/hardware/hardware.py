from collections.abc import Mapping

from .device import Device


class Hardware:
    """Own a fixed collection of named device instances.

    Construction copies the supplied mapping without copying its devices.
    Names can identify instrument roles or individual components. Multiple
    names may refer to the same device instance.

    Successful construction transfers responsibility for closing the devices
    to this collection. Failed construction leaves that responsibility with
    the caller.

    Lookups return borrowed references. The collection must remain open until
    every user, including pipeline workers, has finished using its devices.
    Lookup does not reserve a device or coordinate access between users.

    Devices are closed in reverse mapping order, skipping instances already
    encountered. When one device borrows another, place the resource owner
    before its borrower so the borrower closes first.

    Closing must not overlap lookup or device use. Device operations otherwise
    follow the threading contract of their particular interface.
    """

    def __init__(self, devices: Mapping[str, Device]) -> None:
        """Take ownership of the supplied devices after validating the mapping.

        Args:
            devices: Device instances keyed by nonempty names. An empty mapping
                is valid. Mapping order determines cleanup order.

        Raises:
            TypeError: A name is not a string or a value is not a Device.
            ValueError: A name is empty.

        No supplied device is closed if validation fails.
        """
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
        """Configured names in mapping order, including aliases.

        Names remain available after closure.
        """
        return tuple(self._devices)

    def require[T: Device](
        self,
        name: str,
        interface: type[T],
    ) -> T:
        """Borrow a named device implementing the requested interface.

        Repeated lookups return the same instance. The caller must not close
        the returned device; its lifetime belongs to this collection.

        Args:
            name: Configured device name.
            interface: Required Device subclass or device interface ABC.

        Returns:
            The configured device, checked against interface with isinstance().

        Raises:
            RuntimeError: The collection is closed.
            KeyError: The name is not configured.
            TypeError: The device does not implement the requested interface.
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
        """End collection access and attempt cleanup of every distinct device.

        Close devices in reverse mapping order. Aliases do not cause repeated
        cleanup of the same instance.

        Mark the collection closed before beginning cleanup, and attempt every
        device even if an earlier device raises. Report collected failures
        after all cleanup attempts.

        The collection remains closed when cleanup fails. Subsequent calls do
        nothing: they neither retry cleanup nor repeat a previous exception.
        A closed collection therefore does not establish that every underlying
        resource was successfully released.

        All users must have finished before this method is called. Closure
        cannot revoke device references that have already been borrowed.

        Raises:
            BaseExceptionGroup: One or more devices raised during cleanup.
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
