

class PluginRegistry:
    _plugins = {}

    @classmethod
    def register_plugin(cls, hardware_type, plugin_class):
        """Registers a plugin class for a given hardware type."""
        cls._plugins.setdefault(hardware_type, []).append(plugin_class)

    @classmethod
    def get_plugins(cls, hardware_type):
        """Returns all plugins for a given hardware type."""
        return cls._plugins.get(hardware_type, [])