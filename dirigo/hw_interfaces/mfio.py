from abc import ABC, abstractmethod

# TODO, this module is potentially unncessary

"""
Dirigo multifunction I/O (MFIO) device interface.

MFIO devices generally feature 1 or more of:
- analog in/out
- digital in/out
- counters

Example: NI PCIe-63XX

In Dirigo, (primary) analog data acquisition is managed by the Digitizer 
interface because this will often be a separate card.

"""

class MultiFunctionIO(ABC):
    def __init__(self):
        pass


    @abstractmethod
    def foo(self):
        pass

    # Analog methods
    @abstractmethod
    def generate_analog_out_channel(self):
        pass

    # Digital methods

    # Counter IO