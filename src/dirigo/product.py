from dataclasses import dataclass


@dataclass(frozen=True, slots=True, eq=False)
class Product[T]:
    """A typed payload shared with downstream consumers.

    Do not modify the payload after publication. Freezing this wrapper
    does not make the payload itself immutable.
    """

    payload: T


# One related contract remains unfinished: when may the producer reuse a mutable payload’s storage? That matters if an acquisition worker cycles through preallocated image buffers. The current delivery code does not track when every consumer finishes using a product: it removes a product from the queue before processing and has no release action when the receiving context exits. We should resolve that when reviewing delivery. There is no need to add ownership machinery to Product before deciding where that responsibility belongs.