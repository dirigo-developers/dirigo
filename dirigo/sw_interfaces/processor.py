from typing import Optional
from abc import abstractmethod
from typing import Self

import numpy as np

from dirigo import units 
from dirigo.sw_interfaces.worker import Worker, Product
from dirigo.sw_interfaces.acquisition import Acquisition



class ProcessorProduct(Product):
    """
    Container for processors's product(s): (processed frame) data, timestamps 
    (optional), and positions (optional).
    
    Automatically returns itself to processor product pool when released by 
    all subscribing consumers (functionality of the Product base class).
    """
    __slots__ = ("data", "timestamps", "positions", "phase")
    def __init__(self, 
                 pool, 
                 data: np.ndarray, 
                 timestamps = None, 
                 positions = None,
                 phase = None):
        super().__init__(pool)
        self.data = data
        self.timestamps = timestamps
        self.positions = positions
        self.phase: Optional[float] = phase # should be in radians


class Processor(Worker):
    """
    Dirigo interface for data processing worker thread.
    """
    def __init__(self, upstream: Acquisition | Self):
        """Stores the acquisition and spec in private attributes"""
        super().__init__()
        if isinstance(upstream, Acquisition):
            self._acq = upstream
            self._spec = upstream.spec
        elif isinstance(upstream, Processor):
            self._acq = upstream._acq
            self._spec = upstream._acq.spec
        else:
            raise ValueError("Upstream worker passed to Processor must be either an Acquisition or another Processor")
    
    def init_product_pool(self, n, shape, dtype):
        for _ in range(n):
            aq_buf = ProcessorProduct(
                pool=self._product_pool,
                data=np.empty(shape, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(aq_buf)

    def get_free_product(self) -> ProcessorProduct:
        return self._product_pool.get()

    @property
    @abstractmethod # Not sure this is absolutely needed for every subclass of this.
    def data_range(self) -> units.IntRange:
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        pass