"""Automated token-truncation subsystem for RactoGateway.

Prevents context-window overflows by intelligently trimming conversation
history while preserving the beginning and most-recent turns.

Quick start::

    from ractogateway.truncation import TokenTruncator, TruncationConfig

    truncator = TokenTruncator(TruncationConfig(
        keep_first_n=2,
        keep_last_n=8,
        safety_margin=512,
    ))

    # Wire into any kit:
    kit = OpenAIDeveloperKit(model="gpt-4o", truncator=truncator)

For exact token counts (OpenAI only)::

    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o")
    cfg = TruncationConfig(token_counter=lambda t: len(enc.encode(t)))
    truncator = TokenTruncator(cfg)
"""

from ractogateway.truncation._models import (
    MODEL_CONTEXT_LIMITS,
    TruncationConfig,
)
from ractogateway.truncation.truncator import TokenTruncator

__all__ = [
    "MODEL_CONTEXT_LIMITS",
    "TokenTruncator",
    "TruncationConfig",
]
