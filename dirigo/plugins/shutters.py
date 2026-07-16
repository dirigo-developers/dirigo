import nidaqmx
from nidaqmx.constants import LineGrouping


from dirigo.hw_interfaces.shutter import Shutter
from dirigo.plugins.scanners import validate_ni_channel


class ShutterViaNI(Shutter):
    def __init__(self, control_channel: str, **kwargs):
        super().__init__()
        self._is_open: bool = False

        validate_ni_channel(control_channel)
        self._control_channel = control_channel

        self.close() # make sure shutter is closed
    
    def open(self):
        with nidaqmx.Task("Shutter") as task:
            task.do_channels.add_do_chan(
                lines=self._control_channel,
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            task.write(True)
            self._is_open = True

    def close(self):
        with nidaqmx.Task("Shutter") as task:
            task.do_channels.add_do_chan(
                lines=self._control_channel,
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            task.write(False)
            self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open
