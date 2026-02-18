"""RactoGateway — Unified AI SDK with anti-hallucination prompting via RACTO.

Developer Kits (primary API)::

    from ractogateway import openai_developer_kit as opd
    from ractogateway import google_developer_kit as god
    from ractogateway import anthropic_developer_kit as anth

Core utilities::

    from ractogateway import RactoPrompt, ToolRegistry, tool, Gateway
"""

from ractogateway.adapters.base import LLMResponse
from ractogateway.gateway.runner import Gateway
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry, tool

# Developer Kit subpackages — available as module-level imports.
# Each kit lazily imports its provider SDK only when instantiated.
from ractogateway import anthropic_developer_kit  # noqa: F401
from ractogateway import google_developer_kit  # noqa: F401
from ractogateway import openai_developer_kit  # noqa: F401

__all__ = [
    # Core
    "Gateway",
    "LLMResponse",
    "RactoPrompt",
    "ToolRegistry",
    "tool",
    # Developer Kits
    "openai_developer_kit",
    "google_developer_kit",
    "anthropic_developer_kit",
]
__version__ = "0.1.0"
