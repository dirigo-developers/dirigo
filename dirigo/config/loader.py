from tomlkit import parse as toml_parse, dumps as toml_dumps
from pydantic import ValidationError
from .schema import CoreConfig
from .registry import get_plugin_adapter
from .validators import run_semantic_checks
from .errors import ConfigError



def load_raw_toml(text: str):
    return toml_parse(text)  # preserves comments/ordering


def core_validate(doc) -> CoreConfig:
    return CoreConfig.model_validate(doc)


def validate_with_plugins(core: CoreConfig):
    typed_components = []
    for ref in core.components:
        adapter = get_plugin_adapter(ref.plugin_id)
        if adapter is None:
            raise ConfigError(f"Unknown plugin_id: {ref.plugin_id} (component '{ref.name}')")
        # per-plugin migrate + validate
        migrated_params = adapter.migrate(ref.params)  # no-op allowed
        typed = adapter.model.model_validate(migrated_params)
        typed_components.append((ref, typed, adapter))
    # cross-component checks (channels, clocks, etc.)
    run_semantic_checks(core, typed_components)
    return typed_components


def write_toml(doc) -> str:
    return toml_dumps(doc)
