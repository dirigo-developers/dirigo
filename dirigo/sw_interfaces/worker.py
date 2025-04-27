from abc import ABC, abstractmethod
import threading
import queue




class Product:
    """
    Buffer that automatically returns to pool once released by all consumers.
    """
    __slots__ = ("_remaining", "_pool", "_lock")

    def __init__(self, pool: queue.Queue):
        self._pool = pool
        self._remaining: int = 0          # set automatically by Worker.publish
        self._lock = threading.Lock()

    def _set_consumers(self, n: int):
        """Called once by Worker.publish just before the publish (fan-out)."""
        if n < 1:
            self._release() # trigger return to pool
        self._remaining = n

    def _release(self):
        with self._lock:
            self._remaining -= 1
            if self._remaining < 1:
                self._pool.put(self)


class Worker(threading.Thread, ABC):
    def __init__(self):
        """
        Sets up a worker Thread object with internal publisher-subscriber model.

        Each Worker has a queue.Queue `inbox` which it can call `get()` on to
        obtain incoming data. Each worker can send out data using the `publish`
        method to any subscribers 
        """
        super().__init__() # Sets up Thread
        self._stop_event = threading.Event()  # Event to signal thread termination

        # Publisher-subscriber objects
        self.inbox = queue.Queue()
        self._subscribers: list['Worker'] = []

    @abstractmethod
    def run():
        pass

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

    def publish(self, obj: Product | None):
        """Fan out obj to all subscribers. """
        if obj is None: # sentinel for work finished
            for s in self._subscribers:
                s.inbox.put(None)
            return
        else:
            if not isinstance(obj, Product):
                raise ValueError("Can only publish subclasses of RefCounted.")
            
        obj._set_consumers(len(self._subscribers))

        for subscriber in self._subscribers:
            subscriber.inbox.put(obj) # "Thrilled to share ..."

