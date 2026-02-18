"""RactoGateway — Unified AI SDK with anti-hallucination prompting via RACTO.

Developer Kits (primary API)::

    from ractogateway import openai_developer_kit as opd
    from ractogateway import google_developer_kit as god
    from ractogateway import anthropic_developer_kit as anth

Core utilities::

    from ractogateway import RactoPrompt, ToolRegistry, tool, Gateway
"""

# Developer Kit subpackages — available as module-level imports.
# Each kit lazily imports its provider SDK only when instantiated.
from ractogateway import (
    anthropic_developer_kit,
    google_developer_kit,
    openai_developer_kit,
)
from ractogateway.adapters.base import LLMResponse
from ractogateway.gateway.runner import Gateway
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry, tool

__all__ = [
    "Gateway",
    "LLMResponse",
    "RactoPrompt",
    "ToolRegistry",
    "anthropic_developer_kit",
    "google_developer_kit",
    "openai_developer_kit",
    "tool",
]
__version__ = "0.1.0"
