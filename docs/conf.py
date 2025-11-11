# -- Dirigo documentation build configuration -----------------------------
import os
import sys
from datetime import date

# Make package importable from repo root
sys.path.insert(0, os.path.abspath(".."))

project = "Dirigo"
author = "Dirigo Developers"
copyright = f"{date.today().year}, {author}"

# Try to get version from installed package; fall back if unavailable
try:
    from importlib.metadata import version as _pkg_version
    release = _pkg_version("dirigo")
    version = release
except Exception:
    release = version = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "substitution",
    "tasklist",
]

# Theme
html_theme = "furo"
html_title = "Dirigo Documentation"
html_static_path = ["_static"]
templates_path = ["_templates"]

# Autodoc / autosummary
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Mock heavy deps so RTD builds don't fail
autodoc_mock_imports = [
    "nidaqmx", "PySide6", "spinnaker", "numba", "alazar", "alazartech",
    "atsbindings", "ctypes", "torch", "pandas", "numpy",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # add plugin inventories here later, e.g. "dirigo_ats": ("https://...", {}),
}

# Fail build on broken refs once stable; keep False initially
nitpicky = False

# -- Plugin gallery pre-build hook ----------------------------------------
def _maybe_generate_plugin_gallery(app):
    try:
        from _ext.generate_plugin_gallery import write_markdown as _write
        _write(path=os.path.join(app.srcdir, "reference", "_generated", "plugin-gallery.md"))
    except Exception as e:
        app.logger.warning(f"Plugin gallery generation skipped: {e}")

def setup(app):
    app.connect("builder-inited", _maybe_generate_plugin_gallery)
