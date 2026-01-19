from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from typing import Optional, List, TypeVar, Generic, Any

from dirigo.components import units
from dirigo.hw_interfaces.hw_interface import Device



class Detector(Device):
    attr_name= "detector"
    """Abstract interface for a single detection channel."""
    
    @property
    def index(self) -> int:
        """0-based position inside its DetectorSet (read-only for users)."""
        return self._index

    def _set_index(self, idx: int) -> None:
        self._index = idx

    def __init__(self):
        self._index: int = -1 # overwritten by DetectorSet

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @enabled.setter
    @abstractmethod
    def enabled(self, state: bool) -> bool: ...

    @property
    @abstractmethod
    def gain(self) -> Any:
        """Switchable gain; raise NotImplementedError if fixed."""
        ...

    @gain.setter
    @abstractmethod
    def gain(self, value: Any) -> None: ...

    @property
    @abstractmethod
    def gain_range(self) -> units.IntRange: ...

    @property
    @abstractmethod
    def bandwidth(self) -> units.Frequency: 
        """Switchable bandwidth; raise NotImplementedError if fixed."""
        ...

    @bandwidth.setter
    @abstractmethod
    def bandwidth(self, value: units.Frequency): ...



D = TypeVar("D", bound=Detector)


class DetectorSet(Device, MutableSequence[D], Generic[D]):
    attr_name = "detectors" # more intuitive than typing hw.detector_set[0]. etc 
    """Indexable container for Detectors."""

    def __init__(self, detectors: Optional[List[D]] = None) -> None:
        self._detectors: List[D] = []
        if detectors:
            for detector in detectors:
                self.append(detector)

    def __len__(self) -> int:
        return len(self._detectors)

    def __getitem__(self, idx: int) -> D:
        self._ensure_int(idx)
        return self._detectors[idx]

    def __setitem__(self, idx: int, value: D) -> None:
        self._ensure_int(idx)
        self._ensure_detector(value)
        self._detectors[idx] = value
        self._refresh_indices()

    def __delitem__(self, idx: int) -> None:
        self._ensure_int(idx)
        del self._detectors[idx]
        self._refresh_indices()

    def insert(self, idx: int, value: D) -> None:
        self._ensure_int(idx)
        self._ensure_detector(value)
        self._detectors.insert(idx, value)
        self._refresh_indices()

    def _refresh_indices(self) -> None:
        for i, detector in enumerate(self._detectors):
            detector._set_index(i)

    def _ensure_int(self, idx: object) -> None:
        if not isinstance(idx, int):
            raise TypeError("DetectorSet only supports integer indices")

    def _ensure_detector(self, detector: object) -> None:
        if not isinstance(detector, Detector):
            raise TypeError("DetectorSet only supports setting/inserting Detector subclasses")

    def __iter__(self):
        return iter(self._detectors)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._detectors!r})"
