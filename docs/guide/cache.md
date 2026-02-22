# Caching

RactoGateway ships two complementary, thread-safe cache strategies that can be wired into any developer kit with zero friction.

## Exact Match Cache

`ExactMatchCache` uses a SHA-256 key over the serialised request. On a cache hit the stored response is returned instantly — no network call, no token cost.

```python
from ractogateway.cache import ExactMatchCache
from ractogateway import openai_developer_kit as gpt

cache = ExactMatchCache(max_size=1024)

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o",
    exact_cache=cache,
)

response = kit.chat(gpt.ChatConfig(user_message="What is 2+2?"))
# Second call with identical message hits cache immediately.
response = kit.chat(gpt.ChatConfig(user_message="What is 2+2?"))
```

## Semantic Cache

`SemanticCache` uses vector similarity so *semantically equivalent* queries (even with different wording) return the cached answer.

```python
from ractogateway.cache import SemanticCache

def my_embed_fn(text: str) -> list[float]:
    # Use any embedding function — OpenAI, Google, local model, etc.
    ...

semantic_cache = SemanticCache(embedder=my_embed_fn, similarity_threshold=0.92)

kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o",
    semantic_cache=semantic_cache,
)
```

## Using Both Together

```python
kit = gpt.OpenAIDeveloperKit(
    model="gpt-4o",
    exact_cache=ExactMatchCache(max_size=512),
    semantic_cache=SemanticCache(embedder=my_embed_fn),
)
```

Exact match is checked first (O(1)), then semantic similarity, then a live API call.

## Installation

Both caches are included in the base install. For precise token counting with tiktoken:

```bash
pip install ractogateway[cache]
```
