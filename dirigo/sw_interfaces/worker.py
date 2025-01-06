from abc import ABC, abstractmethod
import threading
import queue



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

    def publish(self, data):
        """Send data to all subscribers.
        
        Similar to Queue.put(data), but automatically sends to multiple queues.
        """
        for subscriber in self._subscribers:
            subscriber.inbox.put(data) # "Thrilled to share ..."

