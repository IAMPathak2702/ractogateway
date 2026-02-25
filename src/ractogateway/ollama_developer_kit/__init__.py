"""Ollama Developer Kit — ``from ractogateway import ollama_developer_kit as local``.

Short usage::

    from ractogateway import ollama_developer_kit as local

    kit = local.Chat(model="llama3.2")          # short alias
    kit = local.OllamaDeveloperKit(model="llama3.2")  # full name (same class)

No API key required — Ollama runs locally::

    ollama serve
    ollama pull llama3.2
"""

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.cache.exact_cache import ExactMatchCache
from ractogateway.cache.semantic_cache import SemanticCache
from ractogateway.ollama_developer_kit.kit import OllamaDeveloperKit
from ractogateway.ollama_developer_kit.server import OllamaServerManager
from ractogateway.routing._models import RoutingTier
from ractogateway.routing.router import CostAwareRouter
from ractogateway.truncation._models import TruncationConfig
from ractogateway.truncation.truncator import TokenTruncator

#: Short alias — ``local.Chat(model="llama3.2")`` is identical to
#: ``local.OllamaDeveloperKit(...)``.
Chat = OllamaDeveloperKit

__all__ = [
    "Chat",
    "ChatConfig",
    "CostAwareRouter",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingVector",
    "ExactMatchCache",
    "FinishReason",
    "LLMResponse",
    "Message",
    "MessageRole",
    "OllamaDeveloperKit",
    "OllamaServerManager",
    "RoutingTier",
    "SemanticCache",
    "StreamChunk",
    "StreamDelta",
    "TokenTruncator",
    "ToolCallResult",
    "TruncationConfig",
]
