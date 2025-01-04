from abc import ABC, abstractmethod
import threading
import queue



class Publisher:
    """A multi-Queue distribution system."""
    def __init__(self):
        self.subscribers: list[queue.Queue] = []  # List of queues for subscribers

    def subscribe(self, subscriber_queue: queue.Queue):
        """Add a new subscriber queue."""
        self.subscribers.append(subscriber_queue)

    def unsubscribe(self, subscriber_queue: queue.Queue):
        """Remove a subscriber queue."""
        self.subscribers.remove(subscriber_queue)

    def publish(self, data):
        """Send data to all subscribers.
        
        Similar to Queue.put(data), but automatically sends to multiple queues.
        """
        for q in self.subscribers:
            q.put(data) # "I'm thrilled to share ..."


class Worker(threading.Thread, ABC):
    def __init__(self):
        """Sets up a worker Thread object with a Publisher to send out data and
        an inbox (Queue) to recieve data.
        """
        super().__init__() # Sets up Thread
        self.publisher = Publisher()
        self.inbox = queue.Queue()
        self._stop_event = threading.Event()  # Event to signal thread termination

    @abstractmethod
    def run():
        pass

    def stop(self, blocking: bool = False):
        """Sets a flag to stop thread."""
        self._stop_event.set()

        if blocking:
            self.join() # does not return until thread completes


