"""Exact-match key-value cache with LRU eviction and optional TTL.

Uses ``collections.OrderedDict`` for O(1) get / put / evict — a standard
least-recently-used (LRU) cache pattern.  No external dependencies.

Thread-safety is provided by a ``threading.Lock`` so the cache is safe to
share across threads without any external synchronisation.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict

from ractogateway.adapters.base import LLMResponse
from ractogateway.cache._models import CacheConfig, CacheEntry, CacheStats


def _make_key(
    user_message: str,
    system_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Build a deterministic SHA-256 cache key from request parameters.

    Hashing avoids key-size bloat while remaining collision-resistant for
    practical workloads.  The digest is hex-encoded (64 chars).
    """
    raw = "\x00".join(
        [user_message, system_prompt, model, str(temperature), str(max_tokens)]
    )
    return hashlib.sha256(raw.encode()).hexdigest()


class ExactMatchCache:
    """Ultra-low-latency key-value cache for identical LLM requests.

    Parameters
    ----------
    max_size:
        LRU capacity.  ``0`` = unlimited (no eviction).
    ttl_seconds:
        Entries older than *ttl_seconds* are treated as misses and
        transparently evicted.  ``None`` disables expiry.

    Example::

        from ractogateway.cache import ExactMatchCache

        cache = ExactMatchCache(max_size=512, ttl_seconds=3600)

        # Wire into a kit:
        kit = OpenAIDeveloperKit(model="gpt-4o", exact_cache=cache)
    """

    def __init__(
        self,
        max_size: int = 1024,
        ttl_seconds: float | None = None,
    ) -> None:
        self._config = CacheConfig(max_size=max_size, ttl_seconds=ttl_seconds)
        # OrderedDict: insertion order = LRU order; move_to_end(key) on hit.
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
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

        O(1) — dictionary lookup + optional move-to-end.
        """
        key = _make_key(user_message, system_prompt, model, temperature, max_tokens)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            # TTL check
            if self._config.ttl_seconds is not None:
                age = time.monotonic() - entry.created_at
                if age > self._config.ttl_seconds:
                    del self._store[key]
                    self._misses += 1
                    return None

            # Cache hit — promote to most-recently-used position
            self._store.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1
            return entry.response

    def put(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        response: LLMResponse,
    ) -> None:
        """Store a response.  Evicts LRU entry when at capacity.

        O(1) amortised — dictionary insert + optional popitem(last=False).
        """
        key = _make_key(user_message, system_prompt, model, temperature, max_tokens)
        with self._lock:
            if key in self._store:
                # Update in place; promote to MRU end.
                self._store[key].response = response
                self._store[key].created_at = time.monotonic()
                self._store.move_to_end(key)
                return

            entry = CacheEntry(response=response, created_at=time.monotonic())
            self._store[key] = entry
            self._store.move_to_end(key)

            # Evict least-recently-used entry when over capacity
            cap = self._config.max_size
            if cap > 0:
                while len(self._store) > cap:
                    self._store.popitem(last=False)  # O(1) — pop from LRU end

    def invalidate(
        self,
        user_message: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """Remove a specific entry.  Returns ``True`` if it was present."""
        key = _make_key(user_message, system_prompt, model, temperature, max_tokens)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Evict all cached entries and reset counters."""
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
            f"ExactMatchCache(max_size={self._config.max_size}, "
            f"ttl={self._config.ttl_seconds}s, "
            f"size={s.size}, hit_rate={s.hit_rate:.1%})"
        )
