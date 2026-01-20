from abc import ABC
from functools import cached_property
from typing import ClassVar
from pydantic import BaseModel, ConfigDict, field_validator



class DeviceConfig(BaseModel):
    # reject any data containing fields not explicitly defined in the model
    model_config = ConfigDict(extra="forbid") 

    vendor: str | None = None
    model: str | None = None

    @field_validator("vendor", "model")
    @classmethod
    def _strip_nonempty_if_provided(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("Must be a non-empty string if provided.")
        return v


class Device(ABC):
    """
    Base-class for Devices in Dirigo.
    
    Devices are configured via a Pydantic `DeviceConfig` model (or subclass).
    Instances are typically constructed as `Device(cfg, ...)`.
    """

    config_model: ClassVar[type[DeviceConfig]] = DeviceConfig

    # human readable title for identification in GUIs
    title: ClassVar[str | None] = None 

    def __init__(self, cfg: DeviceConfig, **kwargs):
        if not isinstance(cfg, type(self).config_model):
            raise TypeError(
                f"{type(self).__name__} expected cfg of type {type(self).config_model.__name__}, "
                f"got {type(cfg).__name__}"
            )
        self.cfg = cfg

        # Lifecycle state
        self._is_connected: bool = False
        self._is_closed: bool = False

        _ = self.vendor  # triggers vendor resolution now; model resolved after connect

    @classmethod
    def display_title(cls) -> str:
        # back-up in case title metadata not specified
        return cls.title or cls.__name__
    
    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_closed(self) -> bool:
        return self._is_closed
    
    def connect(self) -> None:
        """
        Acquire external resources / talk to hardware.

        Subclasses that need I/O should override `_connect_impl()` rather than
        overriding `connect()`, so this idempotence and state handling stays consistent.
        """
        if self._is_closed:
            raise RuntimeError(f"{type(self).__name__}: cannot connect a closed device.")
        if self._is_connected:
            return

        self._connect_impl()
        self._is_connected = True

        # triggers model resolution
        _ = self.model

    def close(self) -> None:
        """
        Release external resources.

        Subclasses that hold resources should override `_close_impl()` rather than
        overriding `close()`, so this idempotence and state handling stays consistent.
        """
        if self._is_closed:
            return

        # Best effort cleanup: allow close even if connect failed partway.
        try:
            self._close_impl()
        finally:
            self._is_connected = False
            self._is_closed = True

    # Subclass hooks for lifecycle
    def _connect_impl(self) -> None:
        """Override to perform actual hardware connection/reservation. Default is no-op."""
        return None

    def _close_impl(self) -> None:
        """Override to release resources. Default is no-op."""
        return None
    
    @cached_property
    def vendor(self) -> str:
        return self._resolve_vendor()
    
    @cached_property
    def model(self) -> str:
        return self._resolve_model()
    
    def _resolve_vendor(self) -> str:
        v = self._introspect_vendor() or self.cfg.vendor
        if v is None:
            raise RuntimeError(f"{type(self).__name__}: vendor unavailable.")
        return v
    
    def _resolve_model(self) -> str:
        m = self._introspect_model() or self.cfg.model
        if m is None:
            raise RuntimeError(f"{type(self).__name__}: no model available (introspection/config).")
        return m
    
    # Hooks for devices that can introspect identity (override in subclasses)
    def _introspect_vendor(self) -> str | None:
        # Vendor is almost always locked to the Device class, so this is rarely used
        return None

    def _introspect_model(self) -> str | None:
        # Override if model can be obtained by via the vendor API
        return None
    
    
