"""RactoGateway fine-tuning module.

Provides a production-grade multimodal training pipeline for OpenAI,
Google Gemini, and Anthropic Claude.

Quick start::

    from ractogateway.finetune import (
        RactoDataset,
        RactoTrainingExample,
        RactoTrainingMessage,
        OpenAIFineTuner,
        GeminiFineTuner,
        AnthropicFineTuner,
    )
"""

from ractogateway.finetune.anthropic_tuner import AnthropicFineTuner
from ractogateway.finetune.dataset import (
    RactoDataset,
    RactoTrainingExample,
    RactoTrainingMessage,
)
from ractogateway.finetune.gemini_tuner import GeminiFineTuner
from ractogateway.finetune.openai_tuner import OpenAIFineTuner

__all__ = [
    "AnthropicFineTuner",
    "GeminiFineTuner",
    "OpenAIFineTuner",
    "RactoDataset",
    "RactoTrainingExample",
    "RactoTrainingMessage",
]
