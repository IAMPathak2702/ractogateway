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
from ractogateway.batch._models import BatchItem, BatchJobInfo, BatchResult, BatchStatus
from ractogateway.batch.anthropic_batch import AnthropicBatchProcessor
from ractogateway.cache.exact_cache import ExactMatchCache
from ractogateway.cache.semantic_cache import SemanticCache
from ractogateway.routing._models import RoutingTier
from ractogateway.routing.router import CostAwareRouter
from ractogateway.truncation._models import TruncationConfig
from ractogateway.truncation.truncator import TokenTruncator

#: Short alias — ``claude.Chat(model="claude-sonnet-4-6")`` is identical to
#: ``claude.AnthropicDeveloperKit(...)``.
Chat = AnthropicDeveloperKit

__all__ = [
    "AnthropicBatchProcessor",
    "AnthropicDeveloperKit",
    "BatchItem",
    "BatchJobInfo",
    "BatchResult",
    "BatchStatus",
    "Chat",
    "ChatConfig",
    "CostAwareRouter",
    "ExactMatchCache",
    "FinishReason",
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
