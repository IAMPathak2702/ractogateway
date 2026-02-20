"""Caching subsystem for RactoGateway.

Two complementary cache strategies:

* :class:`ExactMatchCache` — SHA-256 keyed LRU cache for byte-for-byte
  identical requests (zero latency on hit, no embedding cost).
* :class:`SemanticCache` — vector-similarity cache that returns cached
  answers for *semantically equivalent* queries even when the wording differs.

Both are optional and thread-safe.  Enable them by passing instances to any
developer kit constructor::

    from ractogateway.cache import ExactMatchCache, SemanticCache
    from ractogateway import openai_developer_kit as gpt

    kit = gpt.OpenAIDeveloperKit(
        model="gpt-4o",
        exact_cache=ExactMatchCache(max_size=1024),
        semantic_cache=SemanticCache(embedder=my_embed_fn),
    )
"""

from ractogateway.cache._models import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    SemanticCacheConfig,
    SemanticCacheEntry,
)
from ractogateway.cache.exact_cache import ExactMatchCache
from ractogateway.cache.semantic_cache import EmbedFn, SemanticCache

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "EmbedFn",
    "ExactMatchCache",
    "SemanticCache",
    "SemanticCacheConfig",
    "SemanticCacheEntry",
]
