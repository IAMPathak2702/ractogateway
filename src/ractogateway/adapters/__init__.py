"""ractogateway.adapters — LLM Provider Adapters.

Implements the Adapter Design Pattern to normalize API differences across
LLM providers. Each adapter translates tool schemas, handles request
formatting, and standardizes response parsing.
"""

from ractogateway.adapters.base import (
    BaseLLMAdapter,
    FinishReason,
    LLMResponse,
    ToolCallResult,
)

__all__ = [
    "BaseLLMAdapter",
    "FinishReason",
    "LLMResponse",
    "ToolCallResult",
    # Concrete adapters are imported on demand to avoid pulling in
    # optional dependencies at package import time:
    #   from ractogateway.adapters.openai_kit import OpenAILLMKit
    #   from ractogateway.adapters.google_kit import GoogleLLMKit
    #   from ractogateway.adapters.anthropic_kit import AnthropicLLMKit
    #   from ractogateway.adapters.ollama_kit import OllamaLLMKit
    #   from ractogateway.adapters.huggingface_kit import HuggingFaceLLMKit
]
