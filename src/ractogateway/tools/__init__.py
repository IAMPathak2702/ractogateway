"""ractogateway.tools — Tool Registry & Schema Builder.

Provides a ``ToolRegistry`` that accepts standard Python functions (via a
``@tool`` decorator) or Pydantic models and translates them into
provider-specific tool/function-calling schemas.
"""

from ractogateway.tools.registry import ToolRegistry, ToolSchema, tool

__all__ = ["ToolRegistry", "ToolSchema", "tool"]
