from __future__ import annotations

import importlib
import sys

import pytest


def test_package_import_is_lazy() -> None:
    sys.modules.pop("ractogateway", None)

    module = importlib.import_module("ractogateway")

    assert module.__version__ == "0.1.1"
    assert "ractogateway.rag" not in sys.modules
    assert "ractogateway.openai_developer_kit" not in sys.modules


def test_dir_exposes_public_api() -> None:
    module = importlib.import_module("ractogateway")
    public_names = dir(module)

    assert "RactoPrompt" in public_names
    assert "rag" in public_names


def test_unknown_attribute_raises_attribute_error() -> None:
    module = importlib.import_module("ractogateway")
    missing_name = "__definitely_missing__"

    with pytest.raises(AttributeError):
        getattr(module, missing_name)
