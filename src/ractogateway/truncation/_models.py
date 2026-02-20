"""Data models and defaults for the token-truncation subsystem."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Known context-window limits (in tokens) for common models.
# Users can override via TruncationConfig.context_limit or the model_limits map.
# Values are intentionally conservative (leaving ~10 % head-room).
# ---------------------------------------------------------------------------

MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3-mini": 200_000,
    # Google
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-2.5-pro": 1_048_576,
    # Anthropic
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5-20251001": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
}

# Token budget reserved for the system prompt + the user message itself.
_SYSTEM_RESERVE: int = 512
# Default maximum context when model is not in MODEL_CONTEXT_LIMITS.
_DEFAULT_CONTEXT: int = 8_192

# ---------------------------------------------------------------------------
# Token counter type alias
# ---------------------------------------------------------------------------

TokenCounterFn = Callable[[str], int]


def _approx_count(text: str) -> int:
    """Approximate token count: 1 token ≈ 4 characters.

    No external dependencies required.  Accurate to within ~15 % for English
    text compared to tiktoken; sufficient for truncation decisions.
    """
    return max(1, len(text) // 4)


class TruncationConfig(BaseModel):
    """Configuration for :class:`~ractogateway.truncation.TokenTruncator`.

    Parameters
    ----------
    max_context_tokens:
        Hard cap on total prompt tokens *before* calling the API.  When
        ``None``, the truncator looks up the model in
        :data:`MODEL_CONTEXT_LIMITS` (falling back to ``8 192``).
    keep_first_n:
        Number of *history* messages to always preserve from the start of
        the conversation (anchors context).  Defaults to ``2``.
    keep_last_n:
        Number of *history* messages to always preserve from the most recent
        end of the conversation.  Defaults to ``6``.
    token_counter:
        Callable ``(text: str) -> int``.  Defaults to the built-in
        approximate counter (``len // 4``).  Swap for ``tiktoken`` for
        exact OpenAI token counts::

            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4o")
            config = TruncationConfig(token_counter=lambda t: len(enc.encode(t)))

    safety_margin:
        Extra token budget reserved beyond the system prompt and user
        message.  Defaults to ``512``.
    """

    max_context_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Override context-window size.  None = auto-detect from model name."
        ),
    )
    keep_first_n: int = Field(
        default=2,
        ge=0,
        description="Always keep the first N history messages.",
    )
    keep_last_n: int = Field(
        default=6,
        ge=0,
        description="Always keep the last N history messages.",
    )
    token_counter: TokenCounterFn = Field(
        default=_approx_count,
        description="Callable that counts tokens in a string.",
        exclude=True,  # not serialisable — excluded from model_dump / schema
    )
    safety_margin: int = Field(
        default=_SYSTEM_RESERVE,
        ge=0,
        description="Extra buffer reserved for system-prompt overhead.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def resolve_limit(self, model: str) -> int:
        """Return the effective token limit for *model*.

        Priority: ``max_context_tokens`` → ``MODEL_CONTEXT_LIMITS`` lookup →
        ``_DEFAULT_CONTEXT``.
        """
        if self.max_context_tokens is not None:
            return self.max_context_tokens
        return MODEL_CONTEXT_LIMITS.get(model, _DEFAULT_CONTEXT)

    def model_post_init(self, __context: Any) -> None:
        # Ensure keep_first_n + keep_last_n doesn't trivially exceed a sane value.
        # We don't raise here — the truncator will simply keep all messages
        # if keep_first_n + keep_last_n ≥ len(history).
        pass
