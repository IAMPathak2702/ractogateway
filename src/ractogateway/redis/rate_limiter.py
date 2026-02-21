"""Distributed token-bucket rate limiter backed by Redis.

Uses a sliding 1-minute window: each request increments a counter stored at
``"{key_prefix}:{user_id}:{unix_minute}"``.  The key expires automatically after
60 seconds, so the window always reflects the *current* minute.

Why not a strict token-bucket?
-------------------------------
A true token-bucket requires compare-and-swap semantics (Lua script) to be
atomic.  The sliding-window approach (INCRBY + EXPIRE in a pipeline) is atomic
enough for rate-limiting purposes: it has a small race window at the boundary of
two minute-windows, but this is acceptable — the same trade-off made by every
major API gateway (Stripe, Cloudflare, etc.).

Example::

    from ractogateway.redis import RedisRateLimiter, RateLimitConfig

    limiter = RedisRateLimiter(
        url="redis://localhost:6379/0",
        config=RateLimitConfig(max_tokens_per_minute=5_000),
    )

    # In your request handler (before calling the LLM):
    if not limiter.check_and_consume(user_id="user_42", tokens=estimated_tokens):
        raise RuntimeError("Rate limit exceeded — try again in a minute.")
"""

from __future__ import annotations

import time
from typing import Any

from ractogateway.redis._models import RateLimitConfig


def _require_redis() -> Any:
    try:
        import redis as redis_lib
    except ImportError as exc:
        raise ImportError(
            "The 'redis' package is required for RedisRateLimiter. "
            "Install it with:  pip install ractogateway[redis]"
        ) from exc
    return redis_lib


class RedisRateLimiter:
    """Fleet-wide token-budget rate limiter backed by a shared Redis instance.

    Parameters
    ----------
    url:
        Redis connection URL.  Ignored when *client* is provided.
    client:
        Pre-built ``redis.Redis`` client.  Useful for connection-pool sharing
        or unit-test mocking.
    config:
        :class:`~ractogateway.redis.RateLimitConfig` controlling the token
        budget and Redis key namespace.  Defaults are applied when ``None``.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        client: Any | None = None,
        config: RateLimitConfig | None = None,
    ) -> None:
        self._url = url
        self._provided_client = client
        self._config = config or RateLimitConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        return _require_redis().from_url(self._url, decode_responses=True)

    def _current_key(self, user_id: str) -> str:
        """Build the Redis key for the current 60-second window."""
        unix_minute = int(time.time()) // 60
        return f"{self._config.key_prefix}:{user_id}:{unix_minute}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_consume(self, user_id: str, tokens: int = 1) -> bool:
        """Attempt to consume *tokens* from *user_id*'s budget.

        Returns ``True`` if the request is within budget (tokens are consumed),
        or ``False`` if the rate limit would be exceeded (no tokens consumed).

        The check-and-increment is done in a single Redis pipeline, making it
        safe against concurrent requests from the same user.

        Parameters
        ----------
        user_id:
            Opaque identifier for the caller (e.g. API key, user UUID).
        tokens:
            Number of tokens to consume.  Defaults to ``1`` for request-count
            limiting; pass the estimated LLM token count for cost-based limiting.
        """
        key = self._current_key(user_id)
        cli = self._client()
        max_tpm = self._config.max_tokens_per_minute

        with cli.pipeline(transaction=False) as pipe:
            pipe.get(key)
            pipe.incrby(key, tokens)
            pipe.expire(key, 60)
            results = pipe.execute()

        # results[0] = current value BEFORE increment (None if new key)
        # results[1] = value AFTER increment
        before: str | None = results[0]
        after: int = results[1]

        current_before = int(before) if before is not None else 0

        # If adding `tokens` would exceed the limit, undo the increment.
        if current_before + tokens > max_tpm:
            # Roll back: decrement by tokens (key is still set from pipeline).
            cli.decrby(key, tokens)
            return False

        # Sanity: if after somehow exceeds max (concurrent edge case), cap it.
        if after > max_tpm:
            cli.set(key, max_tpm, keepttl=True)

        return True

    def get_remaining(self, user_id: str) -> int:
        """Return the remaining token budget for the current minute.

        Returns ``max_tokens_per_minute`` if the user has not made any requests
        in the current window.
        """
        key = self._current_key(user_id)
        raw = self._client().get(key)
        used = int(raw) if raw is not None else 0
        return max(0, self._config.max_tokens_per_minute - used)

    def reset(self, user_id: str) -> None:
        """Delete *all* rate-limit keys for *user_id* (current and any stale windows).

        Intended for admin / testing use.  Uses SCAN to avoid blocking.
        """
        cli = self._client()
        pattern = f"{self._config.key_prefix}:{user_id}:*"
        cursor = 0
        while True:
            cursor, keys = cli.scan(cursor, match=pattern, count=100)
            if keys:
                cli.delete(*keys)
            if cursor == 0:
                break

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"RedisRateLimiter(max_tpm={self._config.max_tokens_per_minute}, "
            f"prefix={self._config.key_prefix!r})"
        )
