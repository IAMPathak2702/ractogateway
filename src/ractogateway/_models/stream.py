"""Streaming response models — every chunk the user sees is a ``StreamChunk``."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ractogateway.adapters.base import FinishReason, ToolCallResult


class StreamDelta(BaseModel):
    """Incremental content produced by a single streaming event."""

    text: str = ""
    tool_call_id: str | None = None
    tool_call_name: str | None = None
    tool_call_args_fragment: str | None = None


class StreamChunk(BaseModel):
    """A single piece of a streaming response.

    Consumers iterate over ``StreamChunk`` objects — they never touch raw
    provider events directly.

    Attributes
    ----------
    delta:
        The incremental content for this chunk.
    accumulated_text:
        Running concatenation of all ``delta.text`` values so far.
    finish_reason:
        ``None`` for intermediate chunks; set on the final chunk.
    tool_calls:
        Empty until the final chunk (``is_final=True``).
    usage:
        Token counts — populated on the final chunk only.
    is_final:
        ``True`` only for the very last chunk in the stream.
    raw:
        The underlying provider event (escape-hatch for advanced users).
    """

    delta: StreamDelta = Field(default_factory=StreamDelta)
    accumulated_text: str = ""
    finish_reason: FinishReason | None = None
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    is_final: bool = False
    parsed: dict[str, Any] | list[Any] | None = Field(
        default=None,
        description=(
            "Auto-parsed and validated JSON from accumulated_text on the final chunk. "
            "Populated only when ChatConfig.response_model is set and the stream completes."
        ),
    )
    raw: Any = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}
