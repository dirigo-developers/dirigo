from abc import ABC, abstractmethod


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
