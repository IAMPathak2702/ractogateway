# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol


class _SphinxApp(Protocol):
    def connect(self, event: str, callback: object, priority: int = 500) -> int: ...


# Make the src layout visible to autodoc
DOCS_DIR = Path(__file__).parent
SRC_DIR = DOCS_DIR.parent / "src"
PACKAGE_DIR = SRC_DIR / "ractogateway"
GENERATED_MODULES_DIR = DOCS_DIR / "api" / "modules"
sys.path.insert(0, str(SRC_DIR))

# -- Project information -------------------------------------------------------
project = "RactoGateway"
author = "Ved Prakash Pathak"
copyright = "2026, Ved Prakash Pathak"
version = "0.1.3"
release = "0.1.3"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # NumPy / Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",  # sphinx-autodoc-typehints>=2.0
    "myst_parser",  # myst-parser>=3.0  (.md source files)
]

# MyST options
myst_enable_extensions = ["colon_fence", "deflist"]

# Mock all optional dependencies so autodoc doesn't fail when they aren't installed.
# Keep this list in sync with [project.optional-dependencies] in pyproject.toml.
autodoc_mock_imports = [
    # LLM providers
    "openai",
    "anthropic",
    "google",
    "google.genai",
    # RAG — file readers
    "pypdf",
    "docx",
    "openpyxl",
    "PIL",
    "nltk",
    # RAG — vector stores
    "chromadb",
    "faiss",
    "pinecone",
    "qdrant_client",
    "weaviate",
    "pymilvus",
    "psycopg2",
    "pgvector",
    # RAG — embeddings
    "voyageai",
    "numpy",
    # Performance / cache
    "tiktoken",
    # Redis infrastructure
    "redis",
    # Celery task queue
    "celery",
    "kombu",
    # Kafka event streaming
    "kafka",
    # MCP (Model Context Protocol)
    "mcp",
    "starlette",
    "uvicorn",
]

# autodoc options
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
always_use_bars_union = True  # PEP 604 union style in docs
autoclass_content = "both"

# Napoleon settings — source files use NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False  # type shown inline, not in :rtype:

# Suppress known non-actionable warnings
suppress_warnings = [
    "py.duplicate",  # classes re-exported via kit __init__.py
    "sphinx_autodoc_typehints.forward_reference",  # JsonValue forward ref in engine.py
    "sphinx_autodoc_typehints.guarded_import",  # classmethod subscript in typehints
]

# intersphinx — link to upstream docs
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output ---------------------------------------------------
html_theme = "sphinx_rtd_theme"  # sphinx-rtd-theme>=2.0
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

# -- Source suffix -------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}


def _iter_package_modules() -> list[str]:
    modules: list[str] = []
    for py_file in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = py_file.relative_to(PACKAGE_DIR)
        if py_file.name == "__init__.py":
            module_parts = ("ractogateway", *rel.parts[:-1])
        else:
            module_parts = ("ractogateway", *rel.with_suffix("").parts)
        modules.append(".".join(module_parts))
    return sorted(set(modules))


def _module_doc_stem(module: str) -> str:
    return module.replace(".", "_")


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    path.write_text(content, encoding="utf-8")


def _generate_module_reference(_: _SphinxApp) -> None:
    GENERATED_MODULES_DIR.mkdir(parents=True, exist_ok=True)
    modules = _iter_package_modules()

    expected_files: set[str] = {"index.rst"}
    toctree_entries: list[str] = []

    for module in modules:
        stem = _module_doc_stem(module)
        expected_files.add(f"{stem}.rst")
        toctree_entries.append(f"   {stem}")

        title = module
        module_page = (
            f"{title}\n"
            f"{'=' * len(title)}\n\n"
            f".. automodule:: {module}\n"
            "   :members:\n"
            "   :undoc-members:\n"
            "   :show-inheritance:\n"
            "   :member-order: bysource\n"
            "   :no-index:\n"
        )
        _write_if_changed(GENERATED_MODULES_DIR / f"{stem}.rst", module_page)

    index_page = (
        "Complete Module Reference\n"
        "=========================\n\n"
        "This section is auto-generated from ``src/ractogateway`` and includes\n"
        "every package and module.\n\n"
        ".. toctree::\n"
        "   :maxdepth: 1\n\n" + "\n".join(toctree_entries) + "\n"
    )
    _write_if_changed(GENERATED_MODULES_DIR / "index.rst", index_page)

    for stale_file in GENERATED_MODULES_DIR.glob("*.rst"):
        if stale_file.name not in expected_files:
            stale_file.unlink()


def setup(app: _SphinxApp) -> None:
    app.connect("builder-inited", _generate_module_reference)
