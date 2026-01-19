from pydantic import Field, field_validator
import nidaqmx
from nidaqmx.constants import LineGrouping

from dirigo.hw_interfaces.illuminator import IlluminatorConfig, Illuminator
from dirigo.plugins.scanners import validate_ni_channel



class LEDViaNIConfig(IlluminatorConfig):
    enable_channel: str = Field(
        ...,
        description="Digital line controlling on/off (e.g. Dev1/port0/line15)"
    )

    @field_validator("enable_channel")
    @classmethod
    def _validate_enable_channel(cls, v: str) -> str:
        validate_ni_channel(v)
        return v


class LEDViaNI(Illuminator):
    config_model = LEDViaNIConfig
    title = "NI Digital Line LED"

    def __init__(self, cfg: LEDViaNIConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self._enable_channel = cfg.enable_channel

        # Lifecycle-managed resources
        self._task: nidaqmx.Task | None = None
        self._enabled: bool = False # default OFF

    def _connect_impl(self) -> None:
        # Reserve DAQmx resources
        task = nidaqmx.Task() # no Task name avoids collisions for multi-illuminator setups
        try:
            task.do_channels.add_do_chan(
                lines=self._enable_channel,
                line_grouping=LineGrouping.CHAN_PER_LINE,
            )
        except Exception:
            # If channel add fails, ensure task is closed before re-raising
            task.close()
            raise

        self._task = task

        # Apply current logical state on connect (safe default: off)
        self._task.write(bool(self._enabled))

    def _close_impl(self) -> None:
        # Release DAQmx resources
        if self._task is not None:
            try:
                # Turn off on close
                self._task.write(False)
            finally:
                self._task.close()
                self._task = None

    def _require_connected(self) -> None:
        if not self.is_connected or self._task is None:
            raise RuntimeError(f"{type(self).__name__} is not connected. Call connect() first.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)
        # If connected, apply immediately; otherwise store state to apply on connect
        if self._task is not None:
            self._require_connected()
            self._task.write(self._enabled)
