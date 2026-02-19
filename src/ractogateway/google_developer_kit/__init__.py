"""Google Gemini Developer Kit — ``from ractogateway import google_developer_kit as gemini``.

Short usage::

    from ractogateway import google_developer_kit as gemini

    kit = gemini.Chat(model="gemini-2.0-flash")    # short alias
    kit = gemini.GoogleDeveloperKit(model="gemini-2.0-flash")  # full name (same class)
"""

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.google_developer_kit.kit import GoogleDeveloperKit

#: Short alias — ``gemini.Chat(model="gemini-2.0-flash")`` is identical to
#: ``gemini.GoogleDeveloperKit(...)``.
Chat = GoogleDeveloperKit

__all__ = [
    "Chat",
    "ChatConfig",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingVector",
    "FinishReason",
    "GoogleDeveloperKit",
    "LLMResponse",
    "Message",
    "MessageRole",
    "StreamChunk",
    "StreamDelta",
    "ToolCallResult",
]
