from importlib.metadata import version, PackageNotFoundError
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


SYSTEM_CONFIG_SCHEMA_VERSION = 1


def _get_dirigo_version() -> str:
    try:
        return version("dirigo")
    except PackageNotFoundError:
        return "unknown"


class SystemMetadata(BaseModel):
    """System-level metadata"""
    name: str = Field(
        title       = "System Name", 
        default     = "Unnamed System",
        description = "Descriptive name for this imaging system"
    )
    version: str | None = Field(
        title       = "Version",
        default     = None,
        description = "System version or revision identifier"
    )
    notes: str | None = Field(
        title       = "Notes",
        default     = None,
        description = "Additional notes or description"
    )

    @field_validator("name")
    @classmethod
    def _nonempty_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("System Name cannot be empty.")
        return v



class DeviceDef(BaseModel):
    name: str = Field(
        ...,
        description = "Descriptive name for this device"
    )
    kind: str = Field(
        ..., 
        description = "Device category, e.g. 'digitizer', 'stage'"
    )
    entry_point: str = Field(
        ..., 
        description = "Entry point name for this device kind, e.g. 'alazar', 'teledyne'"
    )
    config: dict[str, Any] = Field(
        default_factory = dict, 
        description     = "Device-specific config"
    )

    @field_validator("name", "kind", "entry_point")
    @classmethod
    def _strip_nonempty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Must not be empty.")
        return v


class SystemConfig(BaseModel):
    config_type: Literal["dirigo.system"] = "dirigo.system"

    schema_version: int = Field(
        default = SYSTEM_CONFIG_SCHEMA_VERSION, 
    )
    generated_by: str | None = Field(
        default     = None,
        description = "Tool that generated this configuration (e.g., dirigo-config 0.1.0, handwritten, ...)",
    )
    created_at: datetime = Field(
        default_factory = datetime.now,
        description     = "Time this system config was created",
    )
    dirigo_version: str = Field(
        default_factory = _get_dirigo_version,
        description     = "Dirigo package version used to generate/validate this configuration",
    )

    system: SystemMetadata
    devices: list[DeviceDef] = Field(default_factory=list)

    @field_validator("generated_by")
    @classmethod
    def _strip_generated_by(cls, v: str | None) -> str | None:
        return v.strip() if isinstance(v, str) else v

    def to_toml(self) -> str:
        """
        Custom TOML text generator to list top-level config metadata, followed 
        by system metadata, followed by a list of devices definitions.
        """
        lines: list[str] = []

        # ---- Header ----
        lines.append(f'config_type = "{self.config_type}"')
        lines.append(f'schema_version = {self.schema_version}')
        if self.generated_by:
            lines.append(f'generated_by = "{self.generated_by}"')
        lines.append(f'created_at = {self.created_at.strftime("%Y-%m-%d %H:%M:%S")}')
        lines.append(f'dirigo_version = "{self.dirigo_version}"')
        lines.append("")

        # ---- Metadata ----
        lines.append("[system]")
        lines.append(f'name = "{self.system.name}"')
        if self.system.version:
            lines.append(f'version = "{self.system.version}"')
        if self.system.notes:
            lines.append(f'notes = "{self.system.notes}"')
        lines.append("")

        # ---- Devices ----
        for dev in self.devices:
            lines.append("[[devices]]")
            lines.append(f'name = "{dev.name}"')
            lines.append(f'kind = "{dev.kind}"')
            lines.append(f'entry_point = "{dev.entry_point}"')

            if dev.config:
                lines.append("config = {")
                for k, v in dev.config.items():
                    lines.append(f'  {k} = {repr(v)}')
                lines.append("}")
            else:
                lines.append("config = {}")

            lines.append("")

        return "\n".join(lines)