# Redis

The `redis` module provides distributed, production-ready utilities that work across multiple server replicas — drop-in replacements or complements for the built-in in-process modules.

## Installation

```bash
pip install ractogateway[redis]
```

## Distributed Exact Cache

`RedisExactCache` is a drop-in for the in-process `ExactMatchCache`. Pass it to any developer kit's `exact_cache=` parameter:

```python
from ractogateway.redis import RedisExactCache
from ractogateway import openai_developer_kit as gpt

cache = RedisExactCache(url="redis://localhost:6379/0", ttl_seconds=3600)
kit = gpt.OpenAIDeveloperKit(model="gpt-4o", exact_cache=cache)
```

## Rate Limiter

`RedisRateLimiter` enforces a sliding 1-minute token budget per `user_id` across all replicas:

```python
from ractogateway.redis import RedisRateLimiter, RateLimitConfig

limiter = RedisRateLimiter(
    url="redis://localhost:6379/0",
    config=RateLimitConfig(max_tokens_per_minute=5_000),
)

estimated_tokens = 200
if not limiter.check_and_consume(user_id="u-42", tokens=estimated_tokens):
    raise RuntimeError("Rate limit exceeded — try again later.")

remaining = limiter.get_remaining("u-42")
```

## Chat Memory

`RedisChatMemory` stores bounded conversation history in a Redis List so conversations survive rolling deployments:

```python
from ractogateway.redis import RedisChatMemory, ChatMemoryConfig

memory = RedisChatMemory(
    url="redis://localhost:6379/0",
    config=ChatMemoryConfig(max_turns=20, ttl_seconds=1800),
)

conv_id = "session-abc"
memory.append(conv_id, "user", "Hello!")
memory.append(conv_id, "assistant", "Hi, how can I help?")

history = memory.get_history(conv_id)
# [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi, …"}]

memory.clear(conv_id)
```

## Using a Pre-Built Redis Client

All three classes also accept a pre-built `redis.Redis` instance via the `client=` parameter instead of a URL:

```python
import redis as _redis
client = _redis.Redis.from_url("redis://localhost:6379/0")

cache = RedisExactCache(client=client, ttl_seconds=600)
```
