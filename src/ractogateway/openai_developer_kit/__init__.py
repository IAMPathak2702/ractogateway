"""OpenAI Developer Kit — ``from ractogateway import openai_developer_kit as gpt``.

Short usage::

    from ractogateway import openai_developer_kit as gpt

    kit = gpt.Chat(model="gpt-4o")          # short alias
    kit = gpt.OpenAIDeveloperKit(model="gpt-4o")  # full name (same class)
"""

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.openai_developer_kit.kit import OpenAIDeveloperKit

#: Short alias — ``gpt.Chat(model="gpt-4o")`` is identical to ``gpt.OpenAIDeveloperKit(...)``.
Chat = OpenAIDeveloperKit

__all__ = [
    "Chat",
    "ChatConfig",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingVector",
    "FinishReason",
    "LLMResponse",
    "Message",
    "MessageRole",
    "OpenAIDeveloperKit",
    "StreamChunk",
    "StreamDelta",
    "ToolCallResult",
]
