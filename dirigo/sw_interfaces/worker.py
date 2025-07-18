from abc import ABC, abstractmethod
import threading
import queue
from typing import Optional

import numpy as np


class EndOfStream(Exception):
    """Raised by _receive_product when upstream is closed or the worker is stopping."""
    pass


class Product:
    """
    Buffer that automatically returns to pool once released by all consumers.
    """
    __slots__ = ("_remaining", "data", "_pool", "_lock")

    def __init__(self, pool: queue.Queue, data: np.ndarray):
        self._pool = pool
        self.data = data
        self._remaining: int = 0          # set automatically by Worker.publish
        self._lock = threading.Lock()

    def _add_consumers(self, n: int):
        """Add n consumer references. Called once by Worker.publish just before 
        the publish (fan-out)."""
        with self._lock:
            self._remaining += n
            if self._remaining < 1:
                self._pool.put(self)
                self._remaining = 0

    def _release(self):
        with self._lock:
            self._remaining -= 1
            if self._remaining < 1:
                self._pool.put(self)
                self._remaining = 0

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._release()

    def hold_once(self):
        """Adds 1 to reference count, so the product will need to be context 
        manager-entered again to release."""
        with self._lock:
            self._remaining += 1


class Worker(threading.Thread, ABC):
    Product = Product

    def __init__(self, name: str = "Worker"):
        """
        Sets up a worker Thread object with internal publisher-subscriber model.

        Each Worker has a queue.Queue `inbox` which it can call `get()` on to
        obtain incoming data. Each worker can send out data using the `publish`
        method to any subscribers 
        """
        super().__init__(name=name) # Sets up Thread
        self._stop_event = threading.Event()  # Event to signal thread termination

        # Publisher-subscriber objects
        self._inbox = queue.Queue()
        self._subscribers: list['Worker'] = []

        # Pool for re-usable product objects
        self._product_pool = queue.Queue()
        self._product_shape: Optional[tuple[int, ...]] = None

    @abstractmethod
    def run(self):
        pass
    
    def _init_product_pool(self, 
                           n: int, 
                           shape: tuple[int, ...], 
                           dtype, 
                           fill_value: int = 0) -> None:
        """Initialize the product pool. For Workers not producing products, pass"""
        self._product_shape = shape
        self._product_dtype = dtype
        for _ in range(n):
            aq_buf = self.Product(
                pool=self._product_pool,
                data=np.full(shape, fill_value, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(aq_buf)

    @property
    def product_shape(self) -> tuple[int, ...]:
        if self._product_shape is None:
            raise RuntimeError("Product pool is not initialized")
        return self._product_shape
    
    @property
    def product_dtype(self):
        if self._product_dtype is None:
            raise RuntimeError("Product pool is not initialized")
        return self._product_dtype

    def stop(self, blocking: bool = False):
        """Sets a flag to stop thread."""
        self._stop_event.set()

        if blocking:
            self.join() # does not return until thread completes

    # Publisher-Subscriber model
    def add_subscriber(self, subscriber: 'Worker'):
        """Add a reference to a new subscriber."""
        self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: 'Worker'):
        """Remove reference to a subscriber."""
        self._subscribers.remove(subscriber)

    def _publish(self, obj: Product | None):
        """Fan out obj to all subscribers. """
        if obj is None: # sentinel for work finished
            for s in self._subscribers:
                s._inbox.put(None)
            return
        else:
            if not isinstance(obj, Product):
                raise ValueError("Can only publish subclasses of Product.")
            
        obj._add_consumers(len(self._subscribers))

        for subscriber in self._subscribers:
            subscriber._inbox.put(obj) # "Thrilled to share ..."

    def _get_free_product(self) -> Product:
        return self._product_pool.get()

    def _receive_product(self,
              block: bool = True,
              timeout: float | None = None
              ) -> Product:
        """
        Wrapper around inbox.get().
        
        Raises EndOfStream error if sentinel None is received.
        """

        if self._stop_event.is_set():
            raise EndOfStream

        product = self._inbox.get(block=block, timeout=timeout)
        
        if product is None:
            print(f"Shutting down thread: {self.__class__.__name__}")
            raise EndOfStream

        if not isinstance(product, Product):
            raise TypeError(f"{self.name}: expected Product, got {type(product)!r}")

        return product


