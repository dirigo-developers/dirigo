

class EndOfStream(Exception): # ruff: ignore[error-suffix-on-exception-name]
    """The subscription has ended and no queued products remain."""


class OperationCancelled(Exception): # ruff: ignore[error-suffix-on-exception-name]
    """An operation was interrupted by cooperative cancellation."""
