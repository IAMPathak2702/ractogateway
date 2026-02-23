"""OpenAI Structured Outputs schema sanitisation utilities.

OpenAI's Structured Outputs API (``response_format`` with ``type='json_schema'``)
rejects schemas that contain keywords unsupported by its implementation.
This module sanitises Pydantic v2 JSON schemas to conform to OpenAI's supported
subset and provides early sanity-checks that surface problems *before* an API call
is made — giving actionable errors instead of opaque API rejections.

Unsupported keywords stripped automatically
-------------------------------------------
* ``default`` — any type
* ``title``   — any level
* ``$schema`` — top-level only

Number constraints (OpenAI ignores / rejects):
  ``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``, ``multipleOf``

String constraints:
  ``minLength``, ``maxLength``, ``pattern``, ``format``

Array constraints:
  ``minItems``, ``maxItems``, ``uniqueItems``

Content encoding:
  ``contentEncoding``, ``contentMediaType``

Strict-mode invariants enforced
--------------------------------
* Every ``object`` schema's ``properties`` keys all appear in ``required``.
* Every ``object`` schema has ``"additionalProperties": false``.

These two rules are required by OpenAI when ``"strict": true`` is set.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Keywords that OpenAI Structured Outputs does not support and that we strip.
_STRIP_KEYWORDS: frozenset[str] = frozenset(
    {
        "default",
        "title",
        "$schema",
        # Number constraints
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        # String constraints
        "minLength",
        "maxLength",
        "pattern",
        "format",
        # Array constraints
        "minItems",
        "maxItems",
        "uniqueItems",
        # Content encoding
        "contentEncoding",
        "contentMediaType",
    }
)

#: Compound schema keywords that OpenAI Structured Outputs does not support and
#: that *cannot* be safely removed — we raise early with a helpful message.
_UNSUPPORTED_COMPOUND: tuple[str, ...] = ("not", "if", "then", "else", "allOf")

#: Primitive types supported by OpenAI Structured Outputs.
_SUPPORTED_TYPES: frozenset[str] = frozenset(
    {"string", "number", "integer", "boolean", "object", "array", "null"}
)

# ---------------------------------------------------------------------------
# Internal recursive sanitiser
# ---------------------------------------------------------------------------


def _sanitize_node(node: Any) -> Any:
    """Recursively sanitise a JSON-Schema node for OpenAI Structured Outputs."""
    if isinstance(node, list):
        return [_sanitize_node(item) for item in node]

    if not isinstance(node, dict):
        return node

    # Strip unsupported keyword keys first.
    cleaned: dict[str, Any] = {k: v for k, v in node.items() if k not in _STRIP_KEYWORDS}

    # Recurse into all remaining child values.
    cleaned = {k: _sanitize_node(v) for k, v in cleaned.items()}

    # Enforce strict-mode requirements for object schemas.
    if cleaned.get("type") == "object" or "properties" in cleaned:
        props = cleaned.get("properties", {})
        if props:
            # Every declared property must be listed in ``required``.
            cleaned["required"] = list(props.keys())
        cleaned.setdefault("additionalProperties", False)

    return cleaned


# ---------------------------------------------------------------------------
# Internal validator (pre-sanitisation checks)
# ---------------------------------------------------------------------------


def _check_node(node: Any, *, path: str, model_name: str) -> None:
    """Walk *node* and raise :exc:`ValueError` for constructs we cannot fix."""
    if not isinstance(node, dict):
        return

    # Compound keywords we cannot automatically remove without changing semantics.
    for kw in _UNSUPPORTED_COMPOUND:
        if kw in node:
            raise ValueError(
                f"[RactoGateway] {model_name}: schema node at '{path}' uses '{kw}', "
                "which is not supported by OpenAI Structured Outputs. "
                "Simplify the field to use a plain type or an Optional union (anyOf)."
            )

    # Validate any declared ``type`` value(s).
    type_val = node.get("type")
    if isinstance(type_val, str) and type_val not in _SUPPORTED_TYPES:
        raise ValueError(
            f"[RactoGateway] {model_name}: schema node at '{path}' declares "
            f"unsupported type '{type_val}'. "
            f"Supported types: {sorted(_SUPPORTED_TYPES)}."
        )
    if isinstance(type_val, list):
        for t in type_val:
            if t not in _SUPPORTED_TYPES:
                raise ValueError(
                    f"[RactoGateway] {model_name}: schema node at '{path}' "
                    f"declares unsupported type '{t}' in type array. "
                    f"Supported types: {sorted(_SUPPORTED_TYPES)}."
                )

    # Recurse into every child node.
    for key, value in node.items():
        child_path = f"{path}.{key}"
        if isinstance(value, dict):
            _check_node(value, path=child_path, model_name=model_name)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                _check_node(item, path=f"{child_path}[{i}]", model_name=model_name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sanitize_for_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitised copy of *schema* for OpenAI Structured Outputs.

    Strips every keyword OpenAI does not support and enforces strict-mode
    invariants (all properties required, ``additionalProperties: false``).

    Parameters
    ----------
    schema:
        A JSON Schema dict as produced by
        ``pydantic.BaseModel.model_json_schema()``.

    Returns
    -------
    dict
        A new dict with all incompatible keywords removed and strict-mode
        invariants applied.  The original *schema* dict is not mutated.
    """
    result = _sanitize_node(schema)
    assert isinstance(result, dict)  # _sanitize_node returns dict when given dict
    return result


def validate_schema_for_openai(schema: dict[str, Any], *, model_name: str = "model") -> None:
    """Raise :exc:`ValueError` if *schema* contains constructs that cannot be
    safely sanitised for OpenAI Structured Outputs.

    Call this *before* sanitisation to get a clear, early error instead of an
    opaque API rejection from OpenAI.

    Parameters
    ----------
    schema:
        Raw schema from ``pydantic.BaseModel.model_json_schema()``.
    model_name:
        Human-readable name included in error messages (use the class name).

    Raises
    ------
    ValueError
        Describing the incompatibility and a suggested fix.
    """
    _check_node(schema, path="$", model_name=model_name)


def build_response_format(model: type[BaseModel]) -> dict[str, Any]:
    """Build an OpenAI ``response_format`` parameter dict for *model*.

    Generates the Pydantic model's JSON Schema, validates it for OpenAI
    compatibility, sanitises it, and returns a ``response_format`` value
    ready to pass directly to an OpenAI Chat Completions call.

    Parameters
    ----------
    model:
        A Pydantic :class:`~pydantic.BaseModel` subclass whose schema will
        be used for OpenAI Structured Outputs.

    Returns
    -------
    dict
        ``{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}``

    Raises
    ------
    ValueError
        If the model's schema contains constructs that OpenAI Structured
        Outputs does not support (raised *before* any API call is made).
    """
    raw = model.model_json_schema()
    validate_schema_for_openai(raw, model_name=model.__name__)
    sanitized = sanitize_for_openai(raw)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": _to_snake_case(model.__name__),
            "schema": sanitized,
            "strict": True,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_snake_case(name: str) -> str:
    """Convert *name* from ``CamelCase`` to ``snake_case``."""
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1).lower()
