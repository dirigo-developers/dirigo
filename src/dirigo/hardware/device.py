from abc import ABC, abstractmethod
from dataclasses import asdict

from dirigo.hardware.metadata import (
    DeviceIdentity,
    DeviceSnapshot,
    DriverIdentity,
    encode_metadata,
)


class Device(ABC):
    """Base interface for a hardware component with an explicit lifetime.

    A concrete implementation owns its connections, tasks, buffers, and other
    resources. It releases those resources through close().

    Hardware owns the lifetimes of devices placed in its collection. Workers
    and other users borrow references to those devices.

    Operational methods and threading requirements belong to the specific
    device interface. This base does not prescribe an acquisition lifecycle.

    Constructors that fail before returning a device must clean up any
    resources they have already acquired.
    """

    _configured_identity: DeviceIdentity = DeviceIdentity()

    @abstractmethod
    def close(self) -> None:
        """Stop activity and release this device's owned resources.

        Repeated calls must be safe. Support cleanup after partial acquisition
        preparation or a failed start, and attempt remaining cleanup even if
        an earlier cleanup step fails.

        Release only owned resources. Resources borrowed from another device
        remain the responsibility of their owner.

        Closing ends the device object's usable hardware lifetime. It does
        not imply powering off the physical equipment.

        The caller must ensure that device use has finished before closing.
        """

    def configure_identity(self, identity: DeviceIdentity) -> None:
        """Replace the user-supplied identity without accessing hardware.

        Reported values take precedence over configured values.
        Pass DeviceIdentity() to clear all configured fields.

        Raises:
            TypeError: identity is not a DeviceIdentity.
        """
        if not isinstance(identity, DeviceIdentity):
            raise TypeError("identity must be a DeviceIdentity.")
        self._configured_identity = identity

    @property
    def configured_identity(self) -> DeviceIdentity:
        """User-supplied identity; unknown fields are None."""
        return self._configured_identity

    @property
    def reported_identity(self) -> DeviceIdentity:
        """Driver-reported identity; unknown fields are None.

        Drivers may override this property to expose cached physical-device
        identity.
        """
        return DeviceIdentity()

    @property
    def identity(self) -> DeviceIdentity:
        """Effective identity, using configured values for unreported fields."""
        configured = self.configured_identity
        reported = self.reported_identity

        return DeviceIdentity(
            manufacturer=(
                reported.manufacturer
                if reported.manufacturer is not None
                else configured.manufacturer
            ),
            model=(reported.model if reported.model is not None else configured.model),
            serial_number=(
                reported.serial_number
                if reported.serial_number is not None
                else configured.serial_number
            ),
        )

    @property
    def driver_identity(self) -> DriverIdentity:
        """Concrete implementation identity, without guessing its distribution.

        Drivers may override this to supply a known distribution and version.
        """
        cls = type(self)
        return DriverIdentity(implementation=f"{cls.__module__}.{cls.__qualname__}")

    def snapshot(self) -> DeviceSnapshot:
        """Return a detached JSON-compatible identity and settings record.

        Capture only the fields explicitly supplied by identity properties and
        _snapshot_settings(). Quantities retain explicit base-unit labels.

        Returned dictionaries and lists may be modified by the caller without
        changing device state. Later device changes do not alter earlier
        snapshots.

        Calls must follow the device's threading and lifecycle requirements.
        This operation does not coordinate concurrent device changes.

        Raises:
            TypeError: A selected metadata value has an unsupported type.
            ValueError: A selected numeric value is nonfinite.
        """
        return {
            "identity": encode_metadata(asdict(self.identity)),
            "driver": encode_metadata(asdict(self.driver_identity)),
            "settings": encode_metadata(self._snapshot_settings()),
        }

    def _snapshot_settings(self) -> dict[str, object]:
        """Select scientific settings for snapshot(); defaults to no settings.

        Return supported scalar values, quantities, mappings, lists, or tuples.
        The returned containers may be borrowed: snapshot() copies them.
        """
        return {}
