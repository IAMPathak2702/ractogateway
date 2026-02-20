"""Automated token truncation for long conversation histories.

When a conversation history is about to breach the model's context-window
limit, :class:`TokenTruncator` trims middle turns while preserving:

* The beginning of the conversation (``keep_first_n`` messages) — provides
  original task context.
* The most recent turns (``keep_last_n`` messages) — preserves conversational
  continuity.
* The current system prompt and user message — always present.

The truncator operates on :class:`~ractogateway._models.chat.ChatConfig` and
returns a *new* config object (Pydantic ``model_copy``), leaving the original
unchanged.

No external dependencies are required for the default approximation mode
(``len(text) // 4``).  Swap in ``tiktoken`` for exact OpenAI token counting.
"""

from __future__ import annotations

from ractogateway._models.chat import ChatConfig, Message
from ractogateway.truncation._models import TruncationConfig


class TokenTruncator:
    """Smart conversation-history trimmer.

    Parameters
    ----------
    config:
        :class:`~ractogateway.truncation.TruncationConfig` instance.  If
        omitted a default config is used (approximate counter, 8 k limit).

    Example::

        from ractogateway.truncation import TokenTruncator, TruncationConfig
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-4o")
        truncator = TokenTruncator(
            TruncationConfig(
                token_counter=lambda t: len(enc.encode(t)),
                keep_first_n=2,
                keep_last_n=8,
            )
        )
        kit = OpenAIDeveloperKit(model="gpt-4o", truncator=truncator)
    """

    def __init__(self, config: TruncationConfig | None = None) -> None:
        self._config = config or TruncationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def truncate(self, chat_config: ChatConfig, model: str) -> ChatConfig:
        """Return a copy of *chat_config* with trimmed history if necessary.

        If the total estimated token count (system prompt + history +
        user_message) fits within the model's context limit, the original
        ``ChatConfig`` is returned unchanged.

        Parameters
        ----------
        chat_config:
            The chat configuration to potentially truncate.
        model:
            The resolved model name used to look up the context-window limit.

        Returns
        -------
        ChatConfig
            A new ``ChatConfig`` instance with (possibly shorter) history.
            The ``user_message`` and all other fields are preserved verbatim.
        """
        cfg = self._config
        limit = cfg.resolve_limit(model)
        budget = max(0, limit - cfg.safety_margin)

        history = chat_config.history
        if not history:
            return chat_config

        # Estimate tokens for fixed parts (user message).
        fixed_tokens = cfg.token_counter(chat_config.user_message)

        # Include prompt if available (resolved outside truncator)
        # — callers should pass system_prompt_text when known; we skip here
        # since the truncator doesn't have access to the compiled prompt.
        # The safety_margin covers that overhead.

        # Total tokens for all history messages
        history_tokens = [cfg.token_counter(m.content) for m in history]
        total = fixed_tokens + sum(history_tokens)

        if total <= budget:
            return chat_config  # nothing to trim

        # Sliding-window truncation:
        # Always keep first keep_first_n + last keep_last_n.
        # Drop messages from the middle until we fit.
        first_n = cfg.keep_first_n
        last_n = cfg.keep_last_n

        # If keep_first_n + keep_last_n >= len(history), keep everything
        # and just return (we've already checked total > budget, so we
        # can't do better without dropping mandatory messages).
        if first_n + last_n >= len(history):
            return chat_config

        # Build the trimmed history list
        trimmed = list(
            self._sliding_window(
                history=history,
                history_tokens=history_tokens,
                budget=budget - fixed_tokens,
                first_n=first_n,
                last_n=last_n,
            )
        )

        return chat_config.model_copy(update={"history": trimmed})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sliding_window(
        history: list[Message],
        history_tokens: list[int],
        budget: int,
        first_n: int,
        last_n: int,
    ) -> list[Message]:
        """Yield history messages that fit within *budget* tokens.

        Strategy:
        1. Reserve slots for the first ``first_n`` and last ``last_n`` messages.
        2. Fill remaining budget with middle messages from newest to oldest.
        """
        n = len(history)
        # Indices of messages that are *always* kept
        first_indices = set(range(min(first_n, n)))
        last_indices = set(range(max(0, n - last_n), n))
        anchors = first_indices | last_indices

        # Token cost of anchors
        anchor_tokens = sum(history_tokens[i] for i in anchors)
        remaining_budget = budget - anchor_tokens

        # Middle indices (in reverse order — prefer keeping most recent middle)
        middle_indices = sorted(
            (i for i in range(n) if i not in anchors),
            reverse=True,
        )

        kept_middle: set[int] = set()
        for idx in middle_indices:
            cost = history_tokens[idx]
            if remaining_budget >= cost:
                kept_middle.add(idx)
                remaining_budget -= cost

        # Compose final list preserving original order
        kept = anchors | kept_middle
        return [history[i] for i in sorted(kept)]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """Convenience wrapper around the configured token counter."""
        return self._config.token_counter(text)

    def __repr__(self) -> str:  # pragma: no cover
        cfg = self._config
        return (
            f"TokenTruncator(keep_first={cfg.keep_first_n}, "
            f"keep_last={cfg.keep_last_n}, "
            f"safety_margin={cfg.safety_margin})"
        )
