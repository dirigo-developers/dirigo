import importlib.metadata as im
from functools import lru_cache
from pathlib import Path

from pydantic import ValidationError

from dirigo import io
from dirigo.core.device import BaseDevice
from dirigo.config.schema import SystemConfig


@lru_cache
def _eps(group: str) -> dict[str, im.EntryPoint]:
    return {ep.name.lower(): ep for ep in im.entry_points().select(group=group)}


class Dirigo2:
    """High-level facade for building Dirigo acquisition pipelines."""
    def __init__(self,
                 system_config_path: Path | str | None = None):
        
        # locate system config
        cfg_path = Path(system_config_path or io.config_path() / "system_config2.toml")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Could not find system configuration file at {cfg_path}."
            )
        # If it doesn't exist, we could still run in no-acquisition mode. TODO
        
        # load and validate system config
        cfg_raw = io.load_toml(cfg_path)
        try:
            self.system_config = SystemConfig.model_validate(cfg_raw)
        except ValidationError as e:
            raise RuntimeError(
                f"System configuration file at {cfg_path} is invalid:\n{e}"
            ) from e
        
        # validate device configs and instantiate devices
        self.devices: dict[str, BaseDevice] = {}
        for device_def in self.system_config.devices:
            group_eps = _eps(f"dirigo.devices.{device_def.kind}")
            if device_def.plugin_id not in group_eps:
                raise RuntimeError(
                    f"Device plugin '{device_def.plugin_id}' for device "
                    f"'{device_def.name}' of kind '{device_def.kind}' not found. "
                    f"Available: {list(group_eps)}"
                )
            plugin_cls: BaseDevice = group_eps[device_def.plugin_id].load()
            self.devices[device_def.name] = plugin_cls(
                config=plugin_cls.validate_config(device_def.config)
            )

        a=1



if __name__ == "__main__":
    dirigo = Dirigo2()