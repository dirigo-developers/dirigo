from abc import ABC, abstractmethod
import threading
import queue

from dirigo.components.hardware import Hardware



class Acquisition(threading.Thread, ABC):
    """
    Dirigo interface for data acquisition worker thread.
    """
    def __init__(self, hw:Hardware, data_queue:queue.Queue):
        super().__init__(self)
        self.hw = hw
        self.data_queue = data_queue
        self.running = False

    @abstractmethod
    def run(self):
        pass

    def stop(self):
        """Sets a flag to stop acquisition."""
        self.running = False


# TODO, move to its own module & entry point
from dataclasses import dataclass


@dataclass
class LineAcquisitionSpec():
    bidirectional_scanning: bool
    line_width: float # meters
    pixel_size: float # meters
    fill_fraction: float
    lines_per_buffer: int

    @property
    def scan_width(self):
        """
        Returns the scan width (in meters) required to reach the line width 
        with the given fill fraction (scan width > line width)
        """
        return self.line_width / self.fill_fraction
    
    @property
    def records_per_buffer(self):
        """
        Returns the number of records (ie triggered recordings) per buffer.

        A value > 1 indicates that data for multiple lines is contained in a
        single record (e.g. bi-directional scanning).
        """
        if self.bidirectional_scanning:
            return self.lines_per_buffer // 2
        else:
            return self.lines_per_buffer


class LineAcquisition(Acquisition):
    required_resources = [
        "Digitizer",
        "FastRasterScanner"
    ]
    def __init__(self, hw, data_queue, spec:LineAcquisitionSpec):
        # Tip: since this method runs on main thread, limit to HW init tasks that will return fast, prepend slower tasks to run method
        super().__init__(hw, data_queue) # stores hw and queue
        # check resources

        # Start scanner
        self.hw.fast_raster_scanner.enabled = True
        self.hw.fast_raster_scanner.amplitude = \
            self.hw.optics.object_position_to_scan_angle(spec.scan_width)

        # Prepare digitizer
        # Get digitizer default config, set it

    def run(self):
        pass
