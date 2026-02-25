"""HuggingFace Developer Kit — ``from ractogateway import huggingface_developer_kit as hf``.

Short usage::

    from ractogateway import huggingface_developer_kit as hf

    # Cloud inference (set HF_TOKEN env var)
    kit = hf.Chat(model="meta-llama/Llama-3.2-3B-Instruct")

    # Local TGI / vLLM server (no token needed)
    kit = hf.Chat(model="tgi", base_url="http://localhost:8080")

    # Full class name (identical)
    kit = hf.HuggingFaceDeveloperKit(model="meta-llama/Llama-3.2-3B-Instruct")
"""

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.cache.exact_cache import ExactMatchCache
from ractogateway.cache.semantic_cache import SemanticCache
from ractogateway.huggingface_developer_kit.kit import HuggingFaceDeveloperKit
from ractogateway.routing._models import RoutingTier
from ractogateway.routing.router import CostAwareRouter
from ractogateway.truncation._models import TruncationConfig
from ractogateway.truncation.truncator import TokenTruncator

#: Short alias — ``hf.Chat(model="...")`` is identical to
#: ``hf.HuggingFaceDeveloperKit(...)``.
Chat = HuggingFaceDeveloperKit

__all__ = [
    "Chat",
    "ChatConfig",
    "CostAwareRouter",
    "EmbeddingConfig",
    "EmbeddingResponse",
    "EmbeddingVector",
    "ExactMatchCache",
    "FinishReason",
    "HuggingFaceDeveloperKit",
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
