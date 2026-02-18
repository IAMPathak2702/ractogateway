"""Tool Registry — convert Python functions and Pydantic models into tool schemas.

Users should never hand-write nested JSON dicts for function-calling. Instead,
they decorate plain Python functions with ``@tool`` or register Pydantic models
directly.  Each LLM adapter then translates the registry's canonical schema
into the provider-specific format it needs.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from enum import Enum
from typing import Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Canonical parameter / tool schema models
# ---------------------------------------------------------------------------


class ParamSchema(BaseModel):
    """Schema for a single function parameter."""

    name: str
    type: str  # JSON Schema type string
    description: str = ""
    required: bool = True
    enum: list[str] | None = None
    default: Any = None

    model_config = {"arbitrary_types_allowed": True}


class ToolSchema(BaseModel):
    """Provider-agnostic canonical representation of a callable tool."""

    name: str
    description: str
    parameters: list[ParamSchema] = Field(default_factory=list)

    # --- JSON Schema export ---------------------------------------------------

    def to_json_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible JSON Schema ``parameters`` object."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type}
            if p.description:
                prop["description"] = p.description
            if p.enum is not None:
                prop["enum"] = p.enum
            if not p.required and p.default is not None:
                prop["default"] = p.default
            properties[p.name] = prop
            if p.required:
                required.append(p.name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        schema["additionalProperties"] = False
        return schema


# ---------------------------------------------------------------------------
# Python type → JSON Schema type mapping
# ---------------------------------------------------------------------------

_PYTHON_TO_JSON_TYPE: dict[type | None, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    None: "string",
}


def _resolve_json_type(annotation: Any) -> tuple[str, list[str] | None]:
    """Map a Python type annotation to a JSON Schema type string.

    Returns
    -------
    tuple[str, list[str] | None]
        (json_type, enum_values_or_none)
    """
    # Unwrap Optional[X] / X | None
    origin = get_origin(annotation)
    if origin is type(int | str):  # types.UnionType (3.10+)
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _resolve_json_type(args[0])

    # Handle typing.Union
    if origin is __import__("typing").Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _resolve_json_type(args[0])

    # Enum → string with enum list
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return "string", [e.value for e in annotation]

    # list[X] → array
    if origin is list:
        return "array", None

    # dict[K, V] → object
    if origin is dict:
        return "object", None

    # Direct lookup
    json_type = _PYTHON_TO_JSON_TYPE.get(annotation, "string")
    return json_type, None


# ---------------------------------------------------------------------------
# Function introspection → ToolSchema
# ---------------------------------------------------------------------------


def _schema_from_function(fn: Callable[..., Any]) -> ToolSchema:
    """Introspect a Python function and build a ``ToolSchema``."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    doc = inspect.getdoc(fn) or ""

    # Parse numpy/google-style parameter descriptions from the docstring.
    param_docs = _parse_param_docs(doc)

    params: list[ParamSchema] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, str)
        json_type, enum_values = _resolve_json_type(annotation)
        has_default = param.default is not inspect.Parameter.empty
        desc = param_docs.get(name, "")

        params.append(
            ParamSchema(
                name=name,
                type=json_type,
                description=desc,
                required=not has_default,
                enum=enum_values,
                default=param.default if has_default else None,
            )
        )

    # Use first line of docstring as tool description.
    description = doc.split("\n")[0].strip() if doc else fn.__name__

    return ToolSchema(
        name=fn.__name__,
        description=description,
        parameters=params,
    )


def _schema_from_model(model: type[BaseModel]) -> ToolSchema:
    """Introspect a Pydantic model and build a ``ToolSchema``."""
    schema = model.model_json_schema()
    properties = schema.get("properties", {})
    required_set = set(schema.get("required", []))

    params: list[ParamSchema] = []
    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        desc = field_info.get("description", "")
        enum_values = field_info.get("enum")
        default = field_info.get("default")

        params.append(
            ParamSchema(
                name=field_name,
                type=json_type,
                description=desc,
                required=field_name in required_set,
                enum=enum_values,
                default=default,
            )
        )

    doc = inspect.getdoc(model) or model.__name__
    description = doc.split("\n")[0].strip()

    return ToolSchema(
        name=model.__name__,
        description=description,
        parameters=params,
    )


# ---------------------------------------------------------------------------
# Docstring param parsing (lightweight)
# ---------------------------------------------------------------------------


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from a docstring.

    Supports simple formats::

        :param name: description
        Args:
            name: description
            name (type): description
    """
    result: dict[str, str] = {}
    if not docstring:
        return result

    lines = docstring.splitlines()
    for line in lines:
        stripped = line.strip()

        # Sphinx-style  :param name: description
        if stripped.startswith(":param "):
            rest = stripped[7:]
            if ":" in rest:
                pname, desc = rest.split(":", 1)
                result[pname.strip()] = desc.strip()

        # Google-style   name: description  or  name (type): description
        elif ":" in stripped and not stripped.startswith(("Returns", "Raises", "Args")):
            candidate = stripped.split(":", 1)
            pname = candidate[0].strip()
            # Strip optional (type) suffix
            if "(" in pname:
                pname = pname.split("(")[0].strip()
            # Only accept single-word param names (avoid matching sentences)
            if pname.isidentifier() and len(pname) < 40:
                desc = candidate[1].strip()
                if desc:
                    result[pname] = desc

    return result


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[..., Any]:
    """Decorator that marks a function as an LLM-callable tool.

    Can be used bare (``@tool``) or with overrides
    (``@tool(name="my_tool", description="…")``).

    The decorated function gains a ``_tool_schema`` attribute containing
    the canonical ``ToolSchema``.
    """

    def _wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        schema = _schema_from_function(f)
        if name is not None:
            schema.name = name
        if description is not None:
            schema.description = description
        f._tool_schema = schema  # type: ignore[attr-defined]
        return f

    if fn is not None:
        # Bare @tool usage (no parentheses)
        return _wrap(fn)
    # @tool(...) usage — return the decorator
    return _wrap


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """A registry that collects tools and exposes them as canonical schemas.

    Usage::

        registry = ToolRegistry()

        @registry.register
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Get the current weather for a city.'''
            ...

        # Or register a Pydantic model:
        registry.register(WeatherRequest)

        # Iterate schemas:
        for schema in registry.schemas:
            print(schema.name)
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        target: Callable[..., Any] | type[BaseModel] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[..., Any] | type[BaseModel]:
        """Register a function or Pydantic model as a tool.

        Works as a decorator (``@registry.register``) or as a direct call
        (``registry.register(MyModel)``).
        """

        def _do_register(
            t: Callable[..., Any] | type[BaseModel],
        ) -> Callable[..., Any] | type[BaseModel]:
            if isinstance(t, type) and issubclass(t, BaseModel):
                schema = _schema_from_model(t)
            elif callable(t):
                # Check if already decorated with @tool
                existing: ToolSchema | None = getattr(t, "_tool_schema", None)
                schema = existing if existing is not None else _schema_from_function(t)
            else:
                raise TypeError(
                    f"Expected a callable or Pydantic BaseModel subclass, got {type(t)}"
                )

            if name is not None:
                schema.name = name
            if description is not None:
                schema.description = description

            self._tools[schema.name] = schema
            if callable(t) and not (isinstance(t, type) and issubclass(t, BaseModel)):
                self._callables[schema.name] = t
            return t

        if target is not None:
            return _do_register(target)
        # Decorator with arguments: @registry.register(name="x")
        return _do_register

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    @property
    def schemas(self) -> list[ToolSchema]:
        """Return all registered tool schemas."""
        return list(self._tools.values())

    def get_schema(self, name: str) -> ToolSchema | None:
        """Look up a single tool schema by name."""
        return self._tools.get(name)

    def get_callable(self, name: str) -> Callable[..., Any] | None:
        """Look up the original callable by tool name."""
        return self._callables.get(name)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
