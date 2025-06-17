from typing import Optional, cast
from abc import abstractmethod
from typing import TypeVar, Generic, Self

import numpy as np

from dirigo.components import units 
from dirigo.sw_interfaces.worker import Worker, Product
from dirigo.sw_interfaces.acquisition import Acquisition, Loader



U_co = TypeVar("U_co", bound="Acquisition | Processor", contravariant=True)


class ProcessorProduct(Product):
    """
    Container for processors's product(s): (processed frame) data, timestamps 
    (optional), and positions (optional).
    
    Automatically returns itself to processor product pool when released by 
    all subscribing consumers (functionality of the Product base class).
    """
    __slots__ = ("data", "timestamps", "positions", "phase", "frequency")
    def __init__(self, 
                 pool, 
                 data: np.ndarray, 
                 timestamps = None, 
                 positions = None,
                 phase = None,
                 frequency = None):
        super().__init__(pool, data)
        self.timestamps = timestamps
        self.positions = positions
        self.phase: Optional[float] = phase # should be in radians
        self.frequency: Optional[float] = frequency # should be in hertz
        self.data: np.ndarray


class Processor(Generic[U_co], Worker):
    """
    Dirigo interface for data processing worker thread.
    """
    Product = ProcessorProduct

    def __init__(self, upstream: U_co):
        """Stores the acquisition and spec in private attributes"""
        super().__init__(name="Processor")
        if isinstance(upstream, (Acquisition, Loader)):
            self._acquisition = upstream
            self._spec = upstream.spec
        elif isinstance(upstream, Processor):
            self._acquisition = upstream._acquisition
            self._spec = upstream._acquisition.spec
        else:
            raise ValueError("Upstream worker passed to Processor must be either an Acquisition or another Processor")

    def _get_free_product(self) -> ProcessorProduct:
        """Gets an available ProcessorProduct from the product pool."""
        return super()._get_free_product() # type: ignore
    
    # def _receive_product(self, block = True, timeout = None) -> AcquisitionProduct | ProcessorProduct:
    #     """Receive incoming product from the _inbox. """
    #     return cast(ProcessorProduct, super()._receive_product(block, timeout))

    @property
    @abstractmethod # Not sure this is absolutely needed for every subclass of this.
    def data_range(self) -> units.IntRange:
        """
        The data range after processing (resampling) has been performed.

        May be higher than the native bit depth of the data capture device.
        """
        pass

