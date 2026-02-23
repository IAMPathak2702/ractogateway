"""Strongly-typed input models for chat completion calls."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry


class MessageRole(str, Enum):
    """Role of a single message in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """A single conversation turn."""

    role: MessageRole
    content: str


class ChatConfig(BaseModel):
    """Validated input for every ``chat`` / ``achat`` / ``stream`` / ``astream`` call.

    This is the **only** object you pass to a developer kit method — no raw
    dicts, no positional sprawl.

    Example::

        config = ChatConfig(
            user_message="Explain Python generators.",
            temperature=0.3,
        )
        response = kit.chat(config)
    """

    user_message: str = Field(
        ...,
        min_length=1,
        description="The end-user's query or instruction.",
    )
    prompt: RactoPrompt | None = Field(
        default=None,
        description=(
            "Optional RACTO prompt. When omitted the kit's default_prompt is used. "
            "At least one must be set."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 = deterministic.",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens in the completion.",
    )
    tools: ToolRegistry | None = Field(
        default=None,
        description="Tool registry for function/tool calling.",
    )
    response_model: type[BaseModel] | None = Field(
        default=None,
        description=(
            "Optional Pydantic model. When set and the LLM returns valid JSON, "
            "the output is validated against this model."
        ),
    )
    max_validation_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Number of automatic retries when response_model validation fails. "
            "On each retry the exact Pydantic errors and the bad JSON are fed "
            "back to the LLM with a targeted correction prompt. "
            "Set to 0 to disable retries and raise ResponseModelValidationError "
            "immediately on the first failure."
        ),
    )
    history: list[Message] = Field(
        default_factory=list,
        description="Prior conversation turns for multi-turn chat.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific pass-through kwargs (top_p, stop, seed, …).",
    )

    model_config = {"arbitrary_types_allowed": True}
