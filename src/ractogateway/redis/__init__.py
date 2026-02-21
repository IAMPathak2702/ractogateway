"""Redis infrastructure layer for RactoGateway.

Three production-ready utilities that replace or complement the built-in
in-process modules when running across multiple servers:

* :class:`RedisExactCache` — distributed drop-in for
  :class:`~ractogateway.cache.ExactMatchCache`.  Pass it directly to any
  developer-kit ``exact_cache=`` parameter; no other code changes required.

* :class:`RedisRateLimiter` — fleet-wide token-budget rate limiter.  Uses a
  sliding 1-minute window so costs can never exceed ``max_tokens_per_minute``
  per ``user_id``, even across multiple server replicas.

* :class:`RedisChatMemory` — bounded sliding-window conversation history.
  Stores the last *N* message pairs in a Redis List so multi-turn conversations
  survive rolling deployments and are accessible to every replica.

Quick start::

    pip install ractogateway[redis]

    from ractogateway.redis import (
        RedisExactCache,
        RedisRateLimiter,
        RedisChatMemory,
        RateLimitConfig,
        ChatMemoryConfig,
    )

    REDIS_URL = "redis://localhost:6379/0"

    # 1. Distributed response cache — wire into any kit:
    cache = RedisExactCache(url=REDIS_URL, ttl_seconds=3600)
    kit = OpenAIDeveloperKit(model="gpt-4o", exact_cache=cache)

    # 2. Rate limiter — check before calling the LLM:
    limiter = RedisRateLimiter(
        url=REDIS_URL,
        config=RateLimitConfig(max_tokens_per_minute=5_000),
    )
    if not limiter.check_and_consume(user_id, tokens=estimated_tokens):
        raise RuntimeError("Rate limit exceeded.")

    # 3. Chat memory — persist and retrieve conversation history:
    memory = RedisChatMemory(
        url=REDIS_URL,
        config=ChatMemoryConfig(max_turns=20, ttl_seconds=1800),
    )
    memory.append(conv_id, "user", user_message)
    history = memory.get_history(conv_id)
"""

from ractogateway.redis._models import ChatMemoryConfig, RateLimitConfig
from ractogateway.redis.chat_memory import RedisChatMemory
from ractogateway.redis.exact_cache import RedisExactCache
from ractogateway.redis.rate_limiter import RedisRateLimiter

__all__ = [
    "ChatMemoryConfig",
    "RateLimitConfig",
    "RedisChatMemory",
    "RedisExactCache",
    "RedisRateLimiter",
]
