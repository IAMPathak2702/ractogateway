"""Pydantic configuration models for the Redis infrastructure layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Configuration for :class:`RedisRateLimiter`.

    Parameters
    ----------
    max_tokens_per_minute:
        Maximum LLM tokens a single ``user_id`` may consume in any 60-second window.
    key_prefix:
        Redis key namespace.  All rate-limit keys are stored under
        ``"{key_prefix}:{user_id}:{unix_minute}"``.
    """

    max_tokens_per_minute: int = Field(default=10_000, gt=0)
    key_prefix: str = Field(default="ractogateway:ratelimit")


class ChatMemoryConfig(BaseModel):
    """Configuration for :class:`RedisChatMemory`.

    Parameters
    ----------
    max_turns:
        Maximum number of *conversation turns* to retain.  Each turn consists of
        one user message and one assistant message, so up to ``max_turns * 2``
        raw messages are stored per conversation.
    ttl_seconds:
        Optional TTL.  Every ``append()`` call refreshes the expiry on the
        underlying Redis list.  ``None`` disables expiry.
    key_prefix:
        Redis key namespace.  Each conversation is stored at
        ``"{key_prefix}:{conversation_id}"``.
    """

    max_turns: int = Field(default=10, gt=0)
    ttl_seconds: float | None = Field(default=None, gt=0)
    key_prefix: str = Field(default="ractogateway:memory")
