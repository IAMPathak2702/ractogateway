"""Base adapter and shared response models.

Every LLM provider adapter inherits from ``BaseLLMAdapter`` and implements
the same interface: tool translation, request building, execution, and
response normalisation.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry

# ---------------------------------------------------------------------------
# Standardised response model
# ---------------------------------------------------------------------------


class FinishReason(str, Enum):
    """Why the model stopped generating."""

    STOP = "stop"
    TOOL_CALL = "tool_call"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"


class ToolCallResult(BaseModel):
    """A single tool/function call returned by the model."""

    id: str = ""
    name: str
    arguments: dict[str, Any]


class LLMResponse(BaseModel):
    """Unified, provider-agnostic response envelope.

    Every adapter's ``run()`` method returns one of these, regardless of
    whether the underlying provider is OpenAI, Gemini, or Anthropic.
    """

    content: str | None = Field(
        default=None,
        description="The text content of the response (cleaned of markdown fences).",
    )
    parsed: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description="Auto-parsed JSON when the response is valid JSON.",
    )
    tool_calls: list[ToolCallResult] = Field(
        default_factory=list,
        description="Tool/function calls requested by the model.",
    )
    finish_reason: FinishReason = FinishReason.STOP
    usage: dict[str, int] = Field(
        default_factory=dict,
        description="Token usage breakdown (prompt_tokens, completion_tokens, total_tokens).",
    )
    raw: Any = Field(
        default=None,
        description="The raw, unmodified provider response for debugging.",
    )

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Markdown / fence stripping
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that wrap JSON payloads."""
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def try_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    """Attempt to parse *text* as JSON after stripping fences.

    Returns ``None`` if the text is not valid JSON.
    """
    cleaned = strip_markdown_fences(text)
    try:
        return json.loads(cleaned)  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------


class BaseLLMAdapter(ABC):
    """Abstract base class that every provider adapter must implement.

    Parameters
    ----------
    model:
        The model identifier (e.g. ``"gpt-4o"``, ``"gemini-2.0-flash"``).
    api_key:
        Provider API key.  When *None*, each concrete adapter should
        fall back to an environment variable.
    """

    provider: str = "base"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self._extra = kwargs

    # ------------------------------------------------------------------
    # Tool translation
    # ------------------------------------------------------------------

    @abstractmethod
    def translate_tools(
        self,
        registry: ToolRegistry,
    ) -> list[dict[str, Any]]:
        """Convert canonical ``ToolSchema`` objects into the provider's
        native tool/function-calling format.
        """

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a prompt against the provider and return a normalised response."""

    @abstractmethod
    async def arun(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async variant of ``run()``."""

    # ------------------------------------------------------------------
    # Helpers available to subclasses
    # ------------------------------------------------------------------

    def _build_response(
        self,
        *,
        content: str | None = None,
        tool_calls: list[ToolCallResult] | None = None,
        finish_reason: FinishReason = FinishReason.STOP,
        usage: dict[str, int] | None = None,
        raw: Any = None,
    ) -> LLMResponse:
        """Construct an ``LLMResponse`` with automatic JSON parsing."""
        parsed = None
        if content:
            content = strip_markdown_fences(content)
            parsed = try_parse_json(content)

        return LLMResponse(
            content=content,
            parsed=parsed,
            tool_calls=tool_calls or [],
            finish_reason=finish_reason,
            usage=usage or {},
            raw=raw,
        )
