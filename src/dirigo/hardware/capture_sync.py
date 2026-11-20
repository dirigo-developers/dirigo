"""Capture lifecycle coordination for acquisitions without timed outputs."""

from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession


class CaptureOnlySyncSession:
    """Borrow one capture session without allocating other hardware resources.

    Preparation and startup delegate to capture. Closure stops a capture whose
    preparation was attempted; the acquisition worker owns capture.close().
    No capture data is read through this object.
    """

    def __init__(self, capture: DigitizerCaptureSession) -> None:
        """Bind a new capture session without reserving hardware."""
        if not isinstance(capture, DigitizerCaptureSession):
            raise TypeError("capture must be a DigitizerCaptureSession.")
        if capture.state is not CaptureState.NEW:
            raise RuntimeError("Capture must be NEW.")
        self._capture = capture
        self._prepare_attempted = False
        self._closed = False

    def prepare(self) -> None:
        """Prepare the borrowed capture without starting acquisition."""
        self._require_open()
        self._prepare_attempted = True
        self._capture.prepare()

    def start(self) -> None:
        """Start or arm the prepared capture."""
        self._require_open()
        self._capture.start()

    def close(self) -> None:
        """Stop once after attempted preparation; leave capture closure to its owner."""
        if self._closed:
            return
        self._closed = True
        if self._prepare_attempted:
            self._capture.stop()

    def _require_open(self) -> None:
        """Reject operations after permanent closure."""
        if self._closed:
            raise RuntimeError("Capture-only sync session is closed.")
