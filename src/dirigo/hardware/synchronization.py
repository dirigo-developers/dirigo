"""Configured synchronization backends shared across acquisition runs."""


class SyncBackend:
    """Base for configured synchronization backends.

    Backends belong to the System. Workers borrow them and own the
    acquisition sessions they create.
    """

    def close(self) -> None:
        """Release persistent backend resources.

        The default implementation has nothing to release. Overrides must
        permit repeated calls and must not close borrowed devices.

        All sessions created by this backend must already be closed.
        """
