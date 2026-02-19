# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from pathlib import Path

# Make the src layout visible to autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -------------------------------------------------------
project = "RactoGateway"
author = "Ved Prakash Pathak"
copyright = "2026, Ved Prakash Pathak"  # noqa: A001
version = "0.1.1"
release = "0.1.1"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",          # NumPy / Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",     # sphinx-autodoc-typehints>=2.0
    "myst_parser",                  # myst-parser>=3.0  (.md source files)
]

# MyST options
myst_enable_extensions = ["colon_fence", "deflist"]

# autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
always_use_bars_union = True        # PEP 604 union style in docs

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False          # type shown inline, not in :rtype:

# intersphinx — link to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------
html_theme = "sphinx_rtd_theme"     # sphinx-rtd-theme>=2.0
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

# -- Source suffix -------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
