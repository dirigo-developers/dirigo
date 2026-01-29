from abc import ABC
from typing import ClassVar, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator



class DeviceSettingError(RuntimeError):
    """Base class for errors applying device settings."""


class SettingNotSettableError(DeviceSettingError):
    """Raised when attempting to set a read-only / fixed device setting."""


class DeviceConfig(BaseModel):
    """
    A DeviceConfig declares information required to properly instantiate the Device.
    
    Intended to be subclassed to collect device-specific information.
    """
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
    

class DeviceIdentity(BaseModel):
    """
    Optional, cached identity snapshot for a Device.

    `vendor` and `model` are always required once the Device exists.
    Other fields are optional and may be unavailable for some devices.

    Not intended to be subclassed.
    """
    model_config = ConfigDict(extra="forbid")

    vendor: str
    model: str

    serial: str | None = None
    firmware: str | None = None
    hardware_rev: str | None = None

    driver: str | None = None
    driver_version: str | None = None

    # Escape hatch for device-specific identity fields.
    extras: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "vendor",
        "model",
        "serial",
        "firmware",
        "hardware_rev",
        "driver",
        "driver_version",
    )
    @classmethod
    def _strip_nonempty_if_provided(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            raise ValueError("Must be a non-empty string if provided.")
        return v

    @field_validator("extras")
    @classmethod
    def _validate_extras(cls, d: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(f"extras keys must be str, got {type(k)}")
            k2 = k.strip()
            if not k2:
                raise ValueError("extras keys must be non-empty strings")
            if not isinstance(v, str):
                raise TypeError(f"extras['{k2}'] must be str, got {type(v)}")
            v2 = v.strip()
            if not v2:
                raise ValueError(f"extras['{k2}'] must be a non-empty string")
            out[k2] = v2
        return out


class DeviceSettings(BaseModel):
    """
    Persistable snapshot of a Device's adjustable state. Should capture subset
    of device properties that affect data generation or interpretation.
    
    Implementations should subclass this model per device-kind (CameraSettings,
    DigitizerSettings, StageSettings...) and/or per concrete device, as needed.
    """
    model_config = ConfigDict(extra="forbid")


class Device(ABC):
    """
    Base-class for Devices in Dirigo.
    
    Devices are configured via a Pydantic `DeviceConfig` model (or subclass).
    Instances are typically constructed as `Device(cfg, ...)`.
    """

    config_model: ClassVar[type[DeviceConfig]] = DeviceConfig
    settings_model: ClassVar[type[DeviceSettings]] = DeviceSettings

    # human readable title for identification in GUIs
    title: ClassVar[str | None] = None 

    def __init__(self, cfg: DeviceConfig, **kwargs):
        self._check_settings()

        if not isinstance(cfg, type(self).config_model):
            raise TypeError(
                f"{type(self).__name__} expected cfg of type "
                f"{type(self).config_model.__name__}, got {type(cfg).__name__}"
            )
        self.cfg = cfg

        self._identity: DeviceIdentity | None = None

        # Lifecycle state
        self._is_connected: bool = False
        self._is_closed: bool = False

    @classmethod
    def display_title(cls) -> str:
        # back-up in case title not specified
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

        self._identity = self._resolve_identity()

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
    
    # ---- Identity ----
    @property
    def identity(self) -> DeviceIdentity:
        if self._identity is None:
            raise RuntimeError("Device must connect before accessing identity.")
        return self._identity

    def _resolve_identity(self) -> DeviceIdentity:
        introspected = self._introspect_identity() or {}

        if not isinstance(introspected, dict):
            raise TypeError(
                f"{type(self).__name__}._introspect_identity() must return dict[str, Any] or None; "
                f"got {type(introspected)}"
            )
        
        vendor = self.cfg.vendor or introspected.get("vendor")
        model = self.cfg.model or introspected.get("model")

        if vendor is None:
            raise RuntimeError(f"{type(self).__name__}: vendor unavailable (introspection/config).")

        if model is None:
            raise RuntimeError(f"{type(self).__name__}: model unavailable (introspection/config).")

        return DeviceIdentity(
            vendor         = vendor,
            model          = model,
            serial         = introspected.get("serial"),
            firmware       = introspected.get("firmware"),
            hardware_rev   = introspected.get("hardware_rev"),
            driver         = introspected.get("driver"),
            driver_version = introspected.get("driver_version"),
            extras         = introspected.get("extras") or {},
        )
    
    def _introspect_identity(self) -> dict[str, Any] | None:
        """
        Optional identity fields beyond vendor/model.

        Allowed keys:
          - vendor, model (optional overrides)
          - serial, firmware, hardware_rev, driver, driver_version
          - extras: dict[str, str]
        """
        return None
    
    # ---- Settings (snapshot + restore) ----
    @classmethod
    def _check_settings(cls):
        """
        Validate that all DeviceSettings fields correspond to readable and
        settable live attributes on this Device.

        This enforces the convention: DeviceSettings.foo  <->  Device.foo (getter + setter)
        """
        settings_model = cls.settings_model

        errors: list[str] = []

        for name in settings_model.model_fields:
            # Attribute must exist on the instance or class
            if not hasattr(cls, name):
                errors.append(
                    f"- missing property '{name}' required by {settings_model.__name__}"
                )
                continue

            cls_attr = getattr(cls, name, None)
            if not isinstance(cls_attr, property):
                errors.append(
                    f"- missing property '{name}' required by {settings_model.__name__}"
                )
                continue

            if cls_attr.fget is None:
                errors.append(
                    f"- property '{name}' has no getter (required for snapshot)"
                )
            if cls_attr.fset is None:
                errors.append(
                    f"- property '{name}' has no setter (required for apply)"
                )
                
        if errors:
            msg = (
                f"{cls.__name__}: settings_model {settings_model.__name__} "
                f"is incompatible with device properties:\n"
                + "\n".join(errors)
            )
            raise TypeError(msg)

    def snapshot_settings(self) -> DeviceSettings:
        """
        Capture a validated settings snapshot.
        """
        if not self._is_connected:
            raise RuntimeError("Device must connect before snapshotting settings.")
        
        data: dict[str, Any] = {}

        for name in type(self).settings_model.model_fields:
            data[name] = getattr(self, name)

        return type(self).settings_model(**data)

    def apply_settings(self, settings: DeviceSettings) -> None:
        """
        Apply settings to hardware.
        """
        if not self._is_connected:
            raise RuntimeError("Device must connect before applying settings.")

        if not isinstance(settings, type(self).settings_model):
            raise TypeError(
                f"{type(self).__name__} expected settings of type "
                f"{type(self).settings_model.__name__}, got {type(settings).__name__}"
            )

        for name in type(settings).model_fields:
            field_value = getattr(settings, name)

            # skip any fields that are None
            if field_value is not None:
                setattr(self, name, field_value)

