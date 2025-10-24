from typing import Any, Literal, Optional
from datetime import datetime
from packaging.version import Version, InvalidVersion
import importlib.metadata as im

from pydantic import BaseModel, Field, field_validator


SYSTEM_CONFIG_SCHEMA_VERSION = 1    # Schema version for system config files

VALID_DEVICE_KINDS = Literal[
    "digitizer", "frame_grabber",   # data sources
    "detector", "camera",           # sensors (n.b. USB camera can also be data source)
    "optics",                       # e.g. laser scanning optics
    "scanner", "stage", "encoder",  # postioning/motion devices
    "illuminator", "laser"          # light sources
]
    

class DeviceDef(BaseModel):
    """Hardware device definition"""
    kind: VALID_DEVICE_KINDS
    plugin_id: str = Field(..., description="Plugin identifier, e.g. 'dirigo.ni_digitizer'")
    name: str = Field(..., description="Unique device name in the system")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Device-specific configuration (validated by plugin)"
    )


class SystemMetadata(BaseModel):
    """System-level metadata"""
    name: str = Field(
        default     = "Default System",
        description = "Descriptive name for this imaging system"
    )
    version: str | None = Field(
        default     = None,
        description = "System version or revision identifier"
    )
    notes: str | None = Field(
        default     = None,
        description = "Additional notes or description"
    )
    created_at: datetime = Field(
        default_factory = datetime.now,
        description     = "Timestamp when system configuration was created"
    )


def _dirigo_version() -> str:
    return im.version("dirigo")


class SystemConfig(BaseModel):
    """Complete system configuration"""
    dirigo_system_config_schema_version: int = Field(
        default     = SYSTEM_CONFIG_SCHEMA_VERSION, 
        ge          = 1,
        description = "Schema version for system configuration files"
    )
    dirigo_version: str = Field(
        default_factory = _dirigo_version,
        description     = "Dirigo version that created this system configuration"
    )
    system: SystemMetadata = Field(default_factory=SystemMetadata)
    devices: list[DeviceDef] = Field(default_factory=list)

    @field_validator("dirigo_version", mode="before")
    @classmethod
    def _validate_version(cls, v):
        # ensure valid PEP 440 version string
        try:
            return str(Version(str(v)))
        except InvalidVersion as e:
            raise ValueError(f"Invalid PEP 440 version: {v}") from e
