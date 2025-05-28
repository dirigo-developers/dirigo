from abc import ABC, abstractmethod
import threading
import queue



class EndOfStream(Exception):
    """Raised by _receive_product when upstream is closed or the worker is stopping."""
    pass


class Product:
    """
    Buffer that automatically returns to pool once released by all consumers.
    """
    __slots__ = ("_remaining", "_pool", "_lock")

    def __init__(self, pool: queue.Queue):
        self._pool = pool
        self._remaining: int = 0          # set automatically by Worker.publish
        self._lock = threading.Lock()

    def _add_consumers(self, n: int):
        """Add n consumer references. Called once by Worker.publish just before 
        the publish (fan-out)."""
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


class Worker(threading.Thread, ABC):
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

    @abstractmethod
    def run(self):
        pass
    
    # often define init_product_pool (for workers creating something: 
    # Acquisition, Processor, Display, but not Logger)

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
            raise EndOfStream

        if not isinstance(product, Product):
            raise TypeError(f"{self.name}: expected Product, got {type(product)!r}")

        return product


