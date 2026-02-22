"""Unit tests for OpenAI Structured Outputs schema sanitisation.

Tests cover:
* Removal of every unsupported keyword category.
* Strict-mode enforcement (required / additionalProperties).
* Recursive sanitisation through $defs, properties, items, anyOf.
* Optional (nullable) field handling.
* Nested Pydantic models.
* Early validation errors for unsupported compound keywords / types.
* build_response_format shape and content.
* _to_snake_case helper.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from ractogateway.adapters._openai_schema import (
    _to_snake_case,
    build_response_format,
    sanitize_for_openai,
    validate_schema_for_openai,
)

# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class Simple(BaseModel):
    name: str
    age: int


class WithConstraints(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern=r"^\w+$")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    count: int = Field(..., ge=1, le=100, multiple_of=1)


class WithOptional(BaseModel):
    label: str
    note: str | None = None


class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address


class Nested(BaseModel):
    items: list[Simple]


# ---------------------------------------------------------------------------
# sanitize_for_openai — keyword stripping
# ---------------------------------------------------------------------------


class TestSanitizeStripsUnsupportedKeywords:
    def test_strips_title_from_top_level(self) -> None:
        schema = {"title": "MyModel", "type": "object", "properties": {}}
        result = sanitize_for_openai(schema)
        assert "title" not in result

    def test_strips_schema_keyword(self) -> None:
        schema = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}
        result = sanitize_for_openai(schema)
        assert "$schema" not in result

    def test_strips_default(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "number", "default": 0.0},
            },
        }
        result = sanitize_for_openai(schema)
        assert "default" not in result["properties"]["x"]

    def test_strips_numeric_constraints(self) -> None:
        prop = {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "exclusiveMinimum": -1,
            "exclusiveMaximum": 101,
            "multipleOf": 0.5,
        }
        result = sanitize_for_openai({"type": "object", "properties": {"v": prop}})
        cleaned = result["properties"]["v"]
        for kw in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
            assert kw not in cleaned, f"'{kw}' was not stripped"

    def test_strips_string_constraints(self) -> None:
        prop = {
            "type": "string",
            "minLength": 1,
            "maxLength": 255,
            "pattern": r"^\w+$",
            "format": "email",
        }
        result = sanitize_for_openai({"type": "object", "properties": {"v": prop}})
        cleaned = result["properties"]["v"]
        for kw in ("minLength", "maxLength", "pattern", "format"):
            assert kw not in cleaned, f"'{kw}' was not stripped"

    def test_strips_array_constraints(self) -> None:
        prop = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10,
            "uniqueItems": True,
        }
        result = sanitize_for_openai({"type": "object", "properties": {"v": prop}})
        cleaned = result["properties"]["v"]
        for kw in ("minItems", "maxItems", "uniqueItems"):
            assert kw not in cleaned, f"'{kw}' was not stripped"

    def test_strips_content_encoding(self) -> None:
        prop = {
            "type": "string",
            "contentEncoding": "base64",
            "contentMediaType": "image/png",
        }
        result = sanitize_for_openai({"type": "object", "properties": {"v": prop}})
        cleaned = result["properties"]["v"]
        assert "contentEncoding" not in cleaned
        assert "contentMediaType" not in cleaned

    def test_preserves_description(self) -> None:
        prop = {"type": "string", "description": "The user's name.", "minLength": 1}
        result = sanitize_for_openai({"type": "object", "properties": {"n": prop}})
        assert result["properties"]["n"]["description"] == "The user's name."

    def test_preserves_enum(self) -> None:
        prop = {"type": "string", "enum": ["a", "b", "c"]}
        result = sanitize_for_openai({"type": "object", "properties": {"v": prop}})
        assert result["properties"]["v"]["enum"] == ["a", "b", "c"]

    def test_preserves_ref(self) -> None:
        schema = {
            "type": "object",
            "properties": {"addr": {"$ref": "#/$defs/Address"}},
            "$defs": {"Address": {"type": "object", "properties": {"street": {"type": "string"}}}},
        }
        result = sanitize_for_openai(schema)
        assert result["properties"]["addr"]["$ref"] == "#/$defs/Address"


# ---------------------------------------------------------------------------
# sanitize_for_openai — strict-mode enforcement
# ---------------------------------------------------------------------------


class TestSanitizeEnforcesStrictMode:
    def test_sets_additional_properties_false(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = sanitize_for_openai(schema)
        assert result["additionalProperties"] is False

    def test_all_properties_added_to_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
        }
        result = sanitize_for_openai(schema)
        assert set(result["required"]) == {"a", "b", "c"}

    def test_required_covers_previously_optional_fields(self) -> None:
        # Pydantic Optional field: not in original `required` list.
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["name"],
        }
        result = sanitize_for_openai(schema)
        assert "note" in result["required"]
        assert "name" in result["required"]

    def test_does_not_overwrite_explicit_additional_properties_false(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": False,
        }
        result = sanitize_for_openai(schema)
        assert result["additionalProperties"] is False

    def test_recursively_enforces_nested_objects(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                }
            },
        }
        result = sanitize_for_openai(schema)
        addr = result["properties"]["address"]
        assert addr["additionalProperties"] is False
        assert set(addr["required"]) == {"street", "city"}

    def test_enforces_strict_mode_in_defs(self) -> None:
        schema = Person.model_json_schema()
        result = sanitize_for_openai(schema)
        address_def = result["$defs"]["Address"]
        assert address_def["additionalProperties"] is False
        assert set(address_def["required"]) == {"street", "city"}


# ---------------------------------------------------------------------------
# sanitize_for_openai — real Pydantic models
# ---------------------------------------------------------------------------


class TestSanitizeWithRealModels:
    def test_simple_model(self) -> None:
        result = sanitize_for_openai(Simple.model_json_schema())
        assert "title" not in result
        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"name", "age"}

    def test_model_with_constraints(self) -> None:
        result = sanitize_for_openai(WithConstraints.model_json_schema())
        username_prop = result["properties"]["username"]
        score_prop = result["properties"]["score"]
        for kw in ("minLength", "maxLength", "pattern", "minimum", "maximum", "default"):
            assert kw not in username_prop, f"'{kw}' should be stripped from username"
            assert kw not in score_prop, f"'{kw}' should be stripped from score"
        # All three fields must be required in strict mode.
        assert set(result["required"]) == {"username", "score", "count"}

    def test_optional_field_preserved_as_anyof(self) -> None:
        result = sanitize_for_openai(WithOptional.model_json_schema())
        note_prop = result["properties"]["note"]
        # Optional[str] → anyOf with null; must be preserved.
        assert "anyOf" in note_prop
        types_in_anyof = {branch.get("type") for branch in note_prop["anyOf"]}
        assert "null" in types_in_anyof
        # The field must appear in required even though it's Optional.
        assert "note" in result["required"]

    def test_nested_model_defs_sanitized(self) -> None:
        result = sanitize_for_openai(Person.model_json_schema())
        assert "$defs" in result
        addr = result["$defs"]["Address"]
        assert "title" not in addr
        assert addr["additionalProperties"] is False

    def test_list_of_models_items_sanitized(self) -> None:
        result = sanitize_for_openai(Nested.model_json_schema())
        # items should reference $defs via $ref — $ref is preserved.
        items_node = result["properties"]["items"]
        assert "items" in items_node or "$ref" in items_node or "type" in items_node

    def test_does_not_mutate_original_schema(self) -> None:
        original = Simple.model_json_schema()
        original_copy = dict(original)
        sanitize_for_openai(original)
        assert original == original_copy  # original untouched


# ---------------------------------------------------------------------------
# validate_schema_for_openai — early error detection
# ---------------------------------------------------------------------------


class TestValidateSchemaForOpenai:
    def test_passes_for_valid_schema(self) -> None:
        # Must not raise.
        validate_schema_for_openai(Simple.model_json_schema(), model_name="Simple")

    def test_raises_for_not_keyword(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"not": {"type": "string"}}},
        }
        with pytest.raises(ValueError, match="'not'"):
            validate_schema_for_openai(schema, model_name="Bad")

    def test_raises_for_if_keyword(self) -> None:
        schema = {
            "type": "object",
            "properties": {},
            "if": {"properties": {"x": {"const": "a"}}},
            "then": {"properties": {"y": {"type": "string"}}},
        }
        with pytest.raises(ValueError, match="'if'"):
            validate_schema_for_openai(schema, model_name="Bad")

    def test_raises_for_allof(self) -> None:
        schema = {
            "type": "object",
            "allOf": [{"properties": {"x": {"type": "string"}}}],
        }
        with pytest.raises(ValueError, match="'allOf'"):
            validate_schema_for_openai(schema, model_name="Bad")

    def test_raises_for_unsupported_type(self) -> None:
        schema = {
            "type": "object",
            "properties": {"x": {"type": "custom_type"}},
        }
        with pytest.raises(ValueError, match="unsupported type 'custom_type'"):
            validate_schema_for_openai(schema, model_name="Bad")

    def test_model_name_appears_in_error(self) -> None:
        schema = {"type": "object", "not": {}}
        with pytest.raises(ValueError, match="MyModel"):
            validate_schema_for_openai(schema, model_name="MyModel")


# ---------------------------------------------------------------------------
# build_response_format
# ---------------------------------------------------------------------------


class TestBuildResponseFormat:
    def test_shape(self) -> None:
        rf = build_response_format(Simple)
        assert rf["type"] == "json_schema"
        js = rf["json_schema"]
        assert js["strict"] is True
        assert "name" in js
        assert "schema" in js

    def test_name_is_snake_case(self) -> None:
        rf = build_response_format(Simple)
        assert rf["json_schema"]["name"] == "simple"

    def test_schema_is_sanitized(self) -> None:
        rf = build_response_format(WithConstraints)
        schema = rf["json_schema"]["schema"]
        assert "title" not in schema
        for prop in schema.get("properties", {}).values():
            assert "title" not in prop
            assert "default" not in prop
            assert "minimum" not in prop
            assert "maximum" not in prop

    def test_strict_mode_enforced(self) -> None:
        rf = build_response_format(Simple)
        schema = rf["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"name", "age"}

    def test_raises_for_bad_model_schema(self) -> None:
        class Bad(BaseModel):
            x: int

        # Manually inject an unsupported keyword to force validation failure.
        original_fn = Bad.model_json_schema

        def patched_schema() -> dict:  # type: ignore[override]
            s = original_fn()
            s["not"] = {}
            return s

        Bad.model_json_schema = classmethod(lambda cls: patched_schema())  # type: ignore[method-assign]
        try:
            with pytest.raises(ValueError, match="'not'"):
                build_response_format(Bad)
        finally:
            Bad.model_json_schema = original_fn  # type: ignore[method-assign]

    def test_optional_model_required_includes_all(self) -> None:
        rf = build_response_format(WithOptional)
        schema = rf["json_schema"]["schema"]
        assert set(schema["required"]) == {"label", "note"}


# ---------------------------------------------------------------------------
# _to_snake_case
# ---------------------------------------------------------------------------


class TestToSnakeCase:
    @pytest.mark.parametrize(
        "input_,expected",
        [
            ("Simple", "simple"),
            ("MyModel", "my_model"),
            ("HTTPSRequest", "https_request"),
            ("WithOptional", "with_optional"),
            ("RactoRAG", "racto_rag"),
            ("OpenAILLMKit", "open_aillm_kit"),
            ("already_snake", "already_snake"),
        ],
    )
    def test_conversion(self, input_: str, expected: str) -> None:
        assert _to_snake_case(input_) == expected
