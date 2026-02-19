"""Anthropic Claude Developer Kit — ``from ractogateway import anthropic_developer_kit as claude``.

Short usage::

    from ractogateway import anthropic_developer_kit as claude

    kit = claude.Chat(model="claude-sonnet-4-6")         # short alias
    kit = claude.AnthropicDeveloperKit(model="claude-sonnet-4-6")  # full name (same class)
"""

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.anthropic_developer_kit.kit import AnthropicDeveloperKit

#: Short alias — ``claude.Chat(model="claude-sonnet-4-6")`` is identical to ``claude.AnthropicDeveloperKit(...)``.
Chat = AnthropicDeveloperKit

__all__ = [
    "AnthropicDeveloperKit",
    "Chat",
    "ChatConfig",
    "FinishReason",
    "LLMResponse",
    "Message",
    "MessageRole",
    "StreamChunk",
    "StreamDelta",
    "ToolCallResult",
]
