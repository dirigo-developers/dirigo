from importlib.metadata import version
from datetime import datetime

from pydantic import BaseModel, Field



def _get_dirigo_version() -> str:
    return version("dirigo")


class SystemMetadata(BaseModel):
    """System-level metadata"""

    # Excluded fields
    schema_version: int = Field(
        default = 1, 
        exclude = True
    )
    dirigo_version: str = Field(
        default_factory = _get_dirigo_version,
        exclude         = True,
        description     = "Dirigo package version used to generate/validate this configuration",
    )
    generated_by: str | None = Field(
        default     = None,
        exclude     = True,
        description = "Tool that generated this configuration (e.g., dirigo-config 0.1.0, handwritten, ...)",
    )

    # User-facing fields
    name: str = Field(
        title       = "System Name", 
        default     = "Default System",
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
    created_at: datetime = Field(
        title           = "Created",
        default_factory = datetime.now,
        description     = "Timestamp when system configuration was created"
    )

