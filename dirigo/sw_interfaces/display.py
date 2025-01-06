from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces import Acquisition, Processor



class Display(Worker):
    """
    Dirigo interface for display processing.
    """
    def __init__(self, acquisition: Acquisition = None, processor: Processor = None):
        """Instantiate with a display queue and either an Acquisition or Processor"""
        super().__init__()

        if (acquisition is not None) and (processor is not None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor, not both")
        elif (acquisition is None) and (processor is None):
            raise ValueError("Error creating Display worker: "
                             "Provide either acquisition or processor.")
        
        self._acquisition = acquisition
        self._processor = processor

    @property
    def nchannels(self):
        """Returns the number of channels expected from Acquisition or Processor."""
        if self._acquisition is not None:
            return self._acquisition.spec.nchannels
        else:
            return self._processor._spec.nchannels
