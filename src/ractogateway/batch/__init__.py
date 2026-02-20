"""Batch-processing subsystem for RactoGateway.

Submit thousands of non-urgent LLM requests through provider Batch APIs at
**~50 % lower cost** than synchronous calls.  Supported providers:

* **OpenAI** — :class:`OpenAIBatchProcessor` (uses Files API + Batch API)
* **Anthropic** — :class:`AnthropicBatchProcessor` (uses Message Batches API)

Quick start (OpenAI)::

    from ractogateway import openai_developer_kit as gpt

    processor = gpt.OpenAIBatchProcessor(model="gpt-4o-mini", default_prompt=my_prompt)
    results = processor.submit_and_wait([
        gpt.BatchItem(custom_id="q1", user_message="Summarise this text: …"),
        gpt.BatchItem(custom_id="q2", user_message="Classify the sentiment: …"),
    ])

    for r in results:
        if r.ok:
            print(r.custom_id, r.response.content)

Quick start (Anthropic)::

    from ractogateway import anthropic_developer_kit as claude

    processor = claude.AnthropicBatchProcessor(
        model="claude-haiku-4-5-20251001", default_prompt=my_prompt
    )
    results = processor.submit_and_wait(items)
"""

from ractogateway.batch._models import (
    BatchItem,
    BatchJobInfo,
    BatchResult,
    BatchStatus,
)
from ractogateway.batch.anthropic_batch import AnthropicBatchProcessor
from ractogateway.batch.openai_batch import OpenAIBatchProcessor

__all__ = [
    "AnthropicBatchProcessor",
    "BatchItem",
    "BatchJobInfo",
    "BatchResult",
    "BatchStatus",
    "OpenAIBatchProcessor",
]
