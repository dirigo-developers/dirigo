import nidaqmx
from nidaqmx.constants import LineGrouping

from dirigo.hw_interfaces.illuminator import Illuminator
from dirigo.plugins.scanners import validate_ni_channel


class LEDViaNI(Illuminator):
    def __init__(self, enable_channel: str, initial_state: bool = False, **kwargs):
        # is super init useful for anything?

        validate_ni_channel(enable_channel)
        self._enable_channel = enable_channel

        self._task = nidaqmx.Task("Line illumination")
        self._task.do_channels.add_do_chan(
            lines=enable_channel,
            line_grouping=LineGrouping.CHAN_PER_LINE
        )
        
        # implement initial state
        if initial_state:
            self.turn_on()
        else:
            self.turn_off()

    def turn_on(self):
        self._task.write(True)

    def turn_off(self):
        self._task.write(False)

    @property
    def intensity(self):
        pass