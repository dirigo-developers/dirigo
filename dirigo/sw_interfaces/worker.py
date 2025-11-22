from abc import ABC, abstractmethod
import threading, queue

import numpy as np

from dirigo.components.profiling import run_id_var, group_var, plugin_var, worker_var



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
        if n < 0 or not isinstance(n, int):
            raise ValueError("n must be integer >= 0")
        with self._lock:
            self._remaining += n
            if self._remaining < 0:
                raise RuntimeError("Product refcount went negative in _add_consumers")
            if self._remaining < 1:
                # No consumers → immediately return to pool
                self._pool.put(self)
                self._remaining = 0

    def _release(self):
        with self._lock:
            self._remaining -= 1
            if self._remaining < 0:
                self._remaining = 0
                raise RuntimeError("Product refcount went negative in _release()")
            if self._remaining < 1:
                self._pool.put(self)
                self._remaining = 0

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._release()

    def hold_once(self):
        """Increments the reference count by 1 (manual retain)."""
        with self._lock:
            self._remaining += 1


class Worker(threading.Thread, ABC):
    Product = Product

    def __init__(self, name: str = "worker"):
        """
        Sets up a worker Thread object with internal publisher-subscriber model.
        """
        super().__init__(name=name) # Sets up Thread
        self._stop_event = threading.Event()  # Event to signal thread termination

        # Publisher-subscriber objects
        self._inbox: "queue.Queue[Product | None]" = queue.Queue()
        self._subscribers: list['Worker'] = []
        self._subs_lock = threading.RLock()  # protect subscriber list & publishing snapshot

        # Pool for re-usable product objects
        self._product_pool: "queue.Queue[Product]" = queue.Queue()
        self._product_shape: tuple[int, ...] | None = None
        self._product_dtype = None # numpy dtype
        
        # Set context
        self._dirigo_group: str | None = None    # "acquisition"/"processor"/"writer"/"display"/"loader"
        self._dirigo_plugin: str | None = None   # entry point name, e.g. "raster_line"
        self._dirigo_run_id: str | None = None   # set by acquisitions; others inherit

    def run(self):
        # Final: subclasses should NOT shadow this method
        run_id_var.set(self._dirigo_run_id) # type: ignore
        group_var.set(self._dirigo_group) # type: ignore 
        plugin_var.set(self._dirigo_plugin) # type: ignore
        worker_var.set(self.name)  # type: ignore
        try:
            self._work()  # subclasses implement this
        finally:
            pass

    @abstractmethod
    def _work(self):
        ...
    
    def _init_product_pool(self, 
                           n: int, 
                           shape: tuple[int, ...], 
                           dtype, 
                           fill_value: int = 0) -> None:
        """Initialize the product pool. For Workers not producing products, pass"""
        self._product_shape = shape
        self._product_dtype = dtype
        for _ in range(n):
            product = self.Product(
                pool=self._product_pool,
                data=np.full(shape, fill_value, dtype) # pre-allocates for large buffers
            )
            self._product_pool.put(product)

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

    def stop(self, blocking: bool = False, propagate: bool = True):
        """
        Signal the thread to stop. Wakes the inbox and (optionally) fans out 
        sentinel None to subscribers.
        """
        self._stop_event.set()

        # Wake ourselves if blocked in _inbox.get()
        try:
            self._inbox.put_nowait(None)
        except queue.Full:
            pass

        # Optionally propagate sentinel None to downstream
        if propagate:
            self._publish(None)

        if blocking:
            self.join() # does not return until thread completes

    # Publisher-Subscriber model
    def add_subscriber(self, subscriber: 'Worker'):
        """Add a reference to a new subscriber and inherit run ID if applicable."""
        with self._subs_lock:
            self._subscribers.append(subscriber)
            if getattr(self, "_dirigo_run_id", None):
                subscriber._dirigo_run_id = self._dirigo_run_id

    def remove_subscriber(self, subscriber: 'Worker'):
        """Remove reference to a subscriber."""
        with self._subs_lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass  # already removed

    def _publish(self, obj: Product | None):
        """Fan out obj to all subscribers. """
        with self._subs_lock:
            subs_snapshot = tuple(self._subscribers)

        if obj is None: # sentinel for work finished
            for s in subs_snapshot:
                s._inbox.put(None)
            return
        
        if not isinstance(obj, Product):
            raise ValueError("Can only publish subclasses of Product.")
        
        # Freeze while product is out (read-only to consumers)
        try:
            obj.data.flags.writeable = False
        except Exception:
            pass  # non-ndarray payloads could be supported later
            
        obj._add_consumers(len(subs_snapshot))
        for subscriber in subs_snapshot:
            subscriber._inbox.put(obj) # "Thrilled to share ..."

    def _get_free_product(self) -> Product:
        p = self._product_pool.get()
        # Unfreeze for producer write
        try:
            p.data.flags.writeable = True
        except Exception:
            pass
        return p

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

        try:
            product = self._inbox.get(block=block, timeout=timeout)
        except queue.Empty:
            raise EndOfStream
        
        if product is None:
            # TODO add logging e.g. f"Shutting down thread: {self.__class__.__name__}"
            raise EndOfStream

        if not isinstance(product, Product):
            raise TypeError(f"{self.name}: expected Product, got {type(product)!r}")

        return product

