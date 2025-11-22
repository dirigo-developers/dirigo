from abc import abstractmethod
from typing import Optional
import re

import numpy as np

from dirigo import io
from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces.acquisition import Acquisition, AcquisitionProduct, Loader
from dirigo.sw_interfaces.processor import Processor, ProcessorProduct
from dirigo.sw_interfaces.display import Display


class Writer(Worker):
    """Dirigo interface for data writing."""
    def __init__(self, 
                 upstream: Acquisition | Loader | Processor,
                 basename: str = "experiment",
                 ) -> None:
        """Instantiate with either an upstream Acquisition or Processor"""
        super().__init__("Writer") # sets up the thread and the publisher-subcriber interface
        
        if isinstance(upstream, (Processor, Display)): 
            self._processor = upstream
            self._acquisition = upstream._acquisition
        elif isinstance(upstream, (Acquisition, Loader)):
            self._processor = None
            self._acquisition = upstream
        else:
            raise ValueError("Upstream Worker must be either an Acquisition or a Processor")

        self.basename = basename
        self.number_files = True    # False: disable automatic number postpending
        self.save_path = io.data_path()
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.file_ext: Optional[str] = None # must be set by concrete class
        self.last_saved_file_path = None
        
        self.frames_per_file: int = 1

    @abstractmethod
    def save_data(self, data: np.ndarray):
        """Save an increment of data"""
        pass

    # def _receive_product(self, block = True, timeout = None) -> AcquisitionProduct | ProcessorProduct:
    #     return super()._receive_product(block, timeout)

    def _file_path(self, image_index: Optional[int] = None):
        """
        Build an output path that never overwrites an existing file.

        If ``image_index`` is None, scan ``self.save_path`` for files that already
        match ``f"{self.basename}_<n>.{self.file_ext}"`` and return the next free index.
        
        If ``image_index`` is given, raise ``FileExistsError`` if that specific
        file already exists.
        """
        if self.number_files:
            # Regex that captures the numeric suffix of files like "<basename>_123.<file_ext>"
            pattern = re.compile(rf"{re.escape(self.basename)}_(\d+)\.{self.file_ext}$")

            if image_index is None:
                # Collect any numeric suffixes on existing files
                existing = [
                    int(m.group(1))
                    for p in self.save_path.glob(f"{self.basename}_*.{self.file_ext}")
                    if (m := pattern.match(p.name))
                ]

                next_index = (max(existing) + 1) if existing else 0
                proposed_file_path = self.save_path / f"{self.basename}_{next_index}.{self.file_ext}"

            else:
                # check that we won't overwrite something
                proposed_file_path = self.save_path / f"{self.basename}_{image_index}.{self.file_ext}"
                if proposed_file_path.exists():
                    raise FileExistsError(f"File already exists: {proposed_file_path}")
        else:
            proposed_file_path = self.save_path / f"{self.basename}.{self.file_ext}"
            if proposed_file_path.exists():
                    raise FileExistsError(f"File already exists: {proposed_file_path}")

        return proposed_file_path