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
from ractogateway.cache.exact_cache import ExactMatchCache
from ractogateway.cache.semantic_cache import SemanticCache
from ractogateway.google_developer_kit.kit import GoogleDeveloperKit
from ractogateway.routing._models import RoutingTier
from ractogateway.routing.router import CostAwareRouter
from ractogateway.truncation._models import TruncationConfig
from ractogateway.truncation.truncator import TokenTruncator

#: Short alias — ``gemini.Chat(model="gemini-2.0-flash")`` is identical to
#: ``gemini.GoogleDeveloperKit(...)``.
Chat = GoogleDeveloperKit

__all__ = [
    "Chat",
    "ChatConfig",
    "CostAwareRouter",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingVector",
    "ExactMatchCache",
    "FinishReason",
    "GoogleDeveloperKit",
    "LLMResponse",
    "Message",
    "MessageRole",
    "RoutingTier",
    "SemanticCache",
    "StreamChunk",
    "StreamDelta",
    "TokenTruncator",
    "ToolCallResult",
    "TruncationConfig",
]
