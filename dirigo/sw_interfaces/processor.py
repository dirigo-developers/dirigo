from dirigo.sw_interfaces.worker import Worker
from dirigo.sw_interfaces.acquisition import Acquisition


# TODO, 
# Must a Processor always be associated with an Acquisition? 
# Can Processors be cascaded?
# Limitation: currently needs to be linked to Acquisition specifically

class Processor(Worker):
    """
    Dirigo interface for data processing worker thread.
    """
    def __init__(self, acquisition: Acquisition):
        super().__init__()
        self._acq = acquisition
        self._spec = acquisition.spec
    
