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
from ractogateway.batch._models import BatchItem, BatchJobInfo, BatchResult, BatchStatus
from ractogateway.batch.openai_batch import OpenAIBatchProcessor
from ractogateway.cache.exact_cache import ExactMatchCache
from ractogateway.cache.semantic_cache import SemanticCache
from ractogateway.openai_developer_kit.kit import OpenAIDeveloperKit
from ractogateway.routing._models import RoutingTier
from ractogateway.routing.router import CostAwareRouter
from ractogateway.truncation._models import TruncationConfig
from ractogateway.truncation.truncator import TokenTruncator

#: Short alias — ``gpt.Chat(model="gpt-4o")`` is identical to ``gpt.OpenAIDeveloperKit(...)``.
Chat = OpenAIDeveloperKit

__all__ = [
    "BatchItem",
    "BatchJobInfo",
    "BatchResult",
    "BatchStatus",
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
    "OpenAIBatchProcessor",
    "OpenAIDeveloperKit",
    "RoutingTier",
    "SemanticCache",
    "StreamChunk",
    "StreamDelta",
    "ToolCallResult",
    "TokenTruncator",
    "TruncationConfig",
]
