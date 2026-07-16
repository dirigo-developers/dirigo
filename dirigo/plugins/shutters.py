import nidaqmx
from nidaqmx.constants import LineGrouping


from dirigo.hw_interfaces.shutter import Shutter
from dirigo.plugins.scanners import validate_ni_channel


class ShutterViaNI(Shutter):
    def __init__(self, control_channel: str):
        super().__init__()

        validate_ni_channel(control_channel)
        self._control_channel = control_channel
    
    def open(self):
        with nidaqmx.Task("Shutter") as task:
            task.do_channels.add_do_chan(
                lines=self._control_channel,
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            task.write(True)

    def close(self):
        with nidaqmx.Task("Shutter") as task:
            task.do_channels.add_do_chan(
                lines=self._control_channel,
                line_grouping=LineGrouping.CHAN_PER_LINE
            )
            task.write(False)