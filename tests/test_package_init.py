from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

import pytest


def _source_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "src" / "ractogateway" / "_version.py"
    text = version_file.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        raise AssertionError("Unable to find __version__ in src/ractogateway/_version.py")
    return match.group(1)


def test_package_import_is_lazy() -> None:
    # Save all ractogateway modules so we can restore them after this test.
    # Without restoring, Pydantic model identity breaks for subsequent tests
    # (two copies of the same class exist in different import cycles).
    saved = {k: v for k, v in sys.modules.items()
             if k == "ractogateway" or k.startswith("ractogateway.")}

    for module_name in list(saved):
        sys.modules.pop(module_name, None)

    try:
        module = importlib.import_module("ractogateway")

        assert module.__version__ == _source_version()
        assert "ractogateway.rag" not in sys.modules
        assert "ractogateway.openai_developer_kit" not in sys.modules
    finally:
        # Restore original modules so later tests use consistent class identities
        sys.modules.update(saved)


def test_dir_exposes_public_api() -> None:
    module = importlib.import_module("ractogateway")
    public_names = dir(module)

    assert "RactoPrompt" in public_names
    assert "rag" in public_names
    assert "kafka" in public_names


def test_unknown_attribute_raises_attribute_error() -> None:
    module = importlib.import_module("ractogateway")
    missing_name = "__definitely_missing__"

    with pytest.raises(AttributeError):
        getattr(module, missing_name)
