"""Redis-backed exact-match cache — distributed drop-in for :class:`ExactMatchCache`.

Stores LLM responses in Redis so the cache is shared across every process and
server in a fleet.  The public API is **byte-for-byte identical** to
:class:`~ractogateway.cache.ExactMatchCache`, which means you can substitute
``RedisExactCache`` wherever ``ExactMatchCache`` is accepted (including all
developer-kit ``exact_cache=`` parameters) without changing any other code.

Thread-safety
-------------
Stats counters are guarded by ``threading.Lock`` (same pattern as the in-process
cache).  The Redis operations themselves are atomic at the command level; no
additional locking is required across processes.

Cache key
---------
``"{key_prefix}:{sha256_hex}"`` where the SHA-256 digest is computed from
``(user_message, system_prompt, model, temperature, max_tokens)`` — identical
hashing logic to :func:`~ractogateway.cache.exact_cache._make_key`.

Example::

    from ractogateway.redis import RedisExactCache
    from ractogateway import openai_developer_kit as gpt

    cache = RedisExactCache(
        url="redis://localhost:6379/0",
        ttl_seconds=3600,
    )
    kit = gpt.OpenAIDeveloperKit(model="gpt-4o", exact_cache=cache)
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from ractogateway.adapters.base import LLMResponse
from ractogateway.cache._models import CacheStats


def _require_redis() -> Any:
    try:
        import redis as redis_lib
    except ImportError as exc:
        raise ImportError(
            "The 'redis' package is required for RedisExactCache. "
            "Install it with:  pip install ractogateway[redis]"
        ) from exc
    return redis_lib


def _make_key(
    user_message: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Build a deterministic SHA-256 cache key from request parameters."""
    raw = "\x00".join(
        [user_message, system_prompt, model, str(temperature), str(max_tokens)]
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class RedisExactCache:
    """Distributed exact-match LRU cache backed by Redis.

    Parameters
    ----------
    url:
        Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        Ignored when *client* is provided.
    client:
        Pre-built ``redis.Redis`` (or compatible) client.  Useful when you
        manage the connection pool yourself or use a mock in tests.
    ttl_seconds:
        Optional TTL for each entry.  Passed directly to Redis ``SET EX``.
        ``None`` means entries never expire (Redis default).
    key_prefix:
        Namespace for all Redis keys managed by this instance.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        client: Any | None = None,
        ttl_seconds: float | None = None,
        key_prefix: str = "ractogateway:exact",
    ) -> None:
        self._url = url
        self._provided_client = client
        self._ttl_seconds = ttl_seconds
        self._key_prefix = key_prefix
        # Stats are tracked in-memory (ephemeral — acceptable; same pattern as
        # in-process ExactMatchCache).
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        """Return the Redis client, creating from URL if needed."""
        if self._provided_client is not None:
            return self._provided_client
        redis_lib = _require_redis()
        # decode_responses=False so we get raw bytes back (safe for JSON payloads)
        return redis_lib.from_url(self._url, decode_responses=False)

    def _full_key(self, digest: str) -> str:
        return f"{self._key_prefix}:{digest}"

    # ------------------------------------------------------------------
    # Public API (identical signatures to ExactMatchCache)
    # ------------------------------------------------------------------

    def get(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse | None:
        """Return a cached response or ``None`` on a miss.

        O(1) Redis GET.
        """
        digest = _make_key(user_message, system_prompt, model, temperature, max_tokens)
        raw = self._client().get(self._full_key(digest))
        with self._lock:
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
        return LLMResponse.model_validate_json(raw)

    def put(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response: LLMResponse,
    ) -> None:
        """Store a response in Redis.

        O(1) Redis SET [EX ttl].
        """
        digest = _make_key(user_message, system_prompt, model, temperature, max_tokens)
        payload = response.model_dump_json()
        cli = self._client()
        if self._ttl_seconds is not None:
            cli.set(self._full_key(digest), payload, ex=int(self._ttl_seconds))
        else:
            cli.set(self._full_key(digest), payload)

    def invalidate(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """Remove a specific entry.  Returns ``True`` if it was present."""
        digest = _make_key(user_message, system_prompt, model, temperature, max_tokens)
        deleted = self._client().delete(self._full_key(digest))
        return bool(deleted)

    def clear(self) -> None:
        """Delete all entries matching this instance's key prefix.

        Uses SCAN to iterate safely (no KEYS * in production).
        Also resets in-memory stats counters.
        """
        cli = self._client()
        pattern = f"{self._key_prefix}:*".encode()
        cursor = 0
        while True:
            cursor, keys = cli.scan(cursor, match=pattern, count=100)
            if keys:
                cli.delete(*keys)
            if cursor == 0:
                break
        with self._lock:
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> CacheStats:
        """Return a snapshot of hit/miss counters plus current Redis key count."""
        with self._lock:
            hits = self._hits
            misses = self._misses
        return CacheStats(hits=hits, misses=misses, size=len(self))

    def __len__(self) -> int:
        """Approximate number of entries via SCAN."""
        cli = self._client()
        pattern = f"{self._key_prefix}:*".encode()
        count = 0
        cursor = 0
        while True:
            cursor, keys = cli.scan(cursor, match=pattern, count=100)
            count += len(keys)
            if cursor == 0:
                break
        return count

    def __repr__(self) -> str:  # pragma: no cover
        s = self.stats
        return (
            f"RedisExactCache(prefix={self._key_prefix!r}, "
            f"ttl={self._ttl_seconds}s, "
            f"size={s.size}, hit_rate={s.hit_rate:.1%})"
        )
