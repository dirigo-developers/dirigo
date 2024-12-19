from abc import ABC, abstractmethod

"""
Dirigo beam scanner interfaces.

1D (line or stage-assisted strip) scanning requires only a fast scanner.

2D (frame) scanning requires both a fast and slow scanner.

Common fast raster scanners include resonant polygon scanner. Slow scanner is 
usually a galvanometer.
"""


class FastRasterScanner(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def axis(self):
        pass
    
    @property
    @abstractmethod
    def nominal_line_rate(self):
        pass

    @property
    @abstractmethod
    def min_scan_angle(self) -> float:
        pass

    @property
    @abstractmethod
    def max_scan_angle(self) -> float:
        pass

    @property
    @abstractmethod
    def enabled(self):
        pass

    @enabled.setter
    @abstractmethod
    def enabled(self, new_value):
        pass

    @property
    @abstractmethod
    def amplitude(self):
        pass

    @amplitude.setter
    @abstractmethod
    def amplitude(self, new_value):
        pass


class SlowRasterScanner(ABC):
    pass