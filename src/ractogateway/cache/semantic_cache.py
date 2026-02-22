"""Semantic similarity cache backed by any embedding function.

Caches LLM responses by semantic meaning rather than exact string match.
When a new query arrives, it is embedded and compared (cosine similarity)
against all stored query embeddings.  If the best match exceeds the
configured threshold, the cached response is returned without making a new
API call — saving cost and latency.

**Embedding protocol:** The cache accepts *any* callable
``(text: str) -> list[float]``.  Wire in the kit's own embed method, a RAG
embedder, or any other embedding service::

    from ractogateway.cache import SemanticCache

    def my_embedder(text: str) -> list[float]:
        # call your embedding API here
        return [0.1, 0.2, ...]

    cache = SemanticCache(embedder=my_embedder, threshold=0.95)

**Complexity:** O(n) per lookup where n = number of stored entries.  For
large caches (> 10 k entries) consider using a proper ANN index (e.g. FAISS)
as the embedder backend and reducing *max_size*.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

from ractogateway.adapters.base import LLMResponse
from ractogateway.cache._models import CacheStats, SemanticCacheConfig, SemanticCacheEntry

# Type alias for the embedding callable protocol.
EmbedFn = Callable[[str], list[float]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors in O(d) time.

    Returns a value in [-1.0, 1.0].  Returns 0.0 for zero-magnitude vectors
    to avoid division by zero.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    denom = mag_a * mag_b
    return dot / denom if denom > 0.0 else 0.0


class SemanticCache:
    """Vector-similarity cache — returns cached answers for semantically
    similar queries, costing $0 in API calls.

    Parameters
    ----------
    embedder:
        Any callable ``(text: str) -> list[float]``.  Called once per *new*
        query (cache miss) and once at ``put()`` time.
    threshold:
        Minimum cosine similarity to declare a hit.  Default ``0.95`` is
        intentionally strict to avoid incorrect responses.
    max_size:
        Maximum number of entries (LRU eviction).  ``0`` = unlimited.
    ttl_seconds:
        Optional per-entry TTL.  ``None`` disables expiry.

    Examples
    --------
    ::

        import ractogateway.openai_developer_kit as gpt

        kit = gpt.OpenAIDeveloperKit(model="gpt-4o")

        def embed(text: str) -> list[float]:
            import openai
            r = openai.OpenAI().embeddings.create(
                model="text-embedding-3-small", input=text
            )
            return r.data[0].embedding

        cache = SemanticCache(embedder=embed, threshold=0.95)
    """

    def __init__(
        self,
        embedder: EmbedFn,
        threshold: float = 0.95,
        max_size: int = 512,
        ttl_seconds: float | None = None,
    ) -> None:
        self._embedder = embedder
        self._config = SemanticCacheConfig(
            threshold=threshold,
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )
        # OrderedDict maps a unique key (insertion-time query text) to entry.
        # We iterate values for similarity search; keys for LRU management.
        self._store: OrderedDict[str, SemanticCacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> LLMResponse | None:
        """Embed *query* and return a cached response if cosine-sim ≥ threshold.

        Returns ``None`` on a cache miss (caller should make the real API call
        and then invoke :meth:`put`).

        Complexity: O(n·d) where n = number of entries, d = embedding dim.
        """
        query_vec = self._embedder(query)
        now = time.monotonic()
        ttl = self._config.ttl_seconds

        best_sim = -1.0
        best_key: str | None = None
        best_entry: SemanticCacheEntry | None = None

        with self._lock:
            # Lazy-evict expired entries and find best cosine match
            expired: list[str] = []
            for key, entry in self._store.items():
                if ttl is not None and (now - entry.created_at) > ttl:
                    expired.append(key)
                    continue
                sim = _cosine_similarity(query_vec, entry.vector)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
                    best_entry = entry

            for k in expired:
                del self._store[k]

            if best_key is None or best_entry is None or best_sim < self._config.threshold:
                self._misses += 1
                return None

            # Promote to MRU end
            self._store.move_to_end(best_key)
            best_entry.hit_count += 1
            self._hits += 1
            return best_entry.response

    def put(self, query: str, response: LLMResponse) -> None:
        """Embed *query* and store *response* for future similar queries.

        Evicts LRU entry when at capacity.
        """
        vector = self._embedder(query)
        entry = SemanticCacheEntry(
            vector=vector,
            response=response,
            created_at=time.monotonic(),
        )
        with self._lock:
            # Use query text as key (unique per distinct query stored)
            if query in self._store:
                self._store[query].response = response
                self._store[query].vector = vector
                self._store[query].created_at = entry.created_at
                self._store.move_to_end(query)
                return

            self._store[query] = entry
            self._store.move_to_end(query)

            cap = self._config.max_size
            if cap > 0:
                while len(self._store) > cap:
                    self._store.popitem(last=False)

    def clear(self) -> None:
        """Remove all entries and reset counters."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> CacheStats:
        """Return a snapshot of hit/miss/size counters."""
        with self._lock:
            return CacheStats(hits=self._hits, misses=self._misses, size=len(self._store))

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:  # pragma: no cover
        s = self.stats
        return (
            f"SemanticCache(threshold={self._config.threshold}, "
            f"max_size={self._config.max_size}, "
            f"size={s.size}, hit_rate={s.hit_rate:.1%})"
        )
