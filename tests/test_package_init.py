from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _pyproject_version() -> str:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version ="):
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")

    raise AssertionError("Unable to find project version in pyproject.toml")


def test_package_import_is_lazy() -> None:
    sys.modules.pop("ractogateway", None)

    module = importlib.import_module("ractogateway")

    assert module.__version__ == _pyproject_version()
    assert "ractogateway.rag" not in sys.modules
    assert "ractogateway.openai_developer_kit" not in sys.modules


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
