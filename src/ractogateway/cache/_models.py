"""Shared data models for caching subsystem."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ractogateway.adapters.base import LLMResponse


class CacheConfig(BaseModel):
    """Configuration for cache instances.

    Parameters
    ----------
    max_size:
        Maximum number of entries to hold.  When full, the least-recently-used
        entry is evicted (LRU policy).  ``0`` means unlimited.
    ttl_seconds:
        Time-to-live in seconds.  Entries older than this are treated as
        misses and evicted lazily.  ``None`` disables TTL.
    """

    max_size: int = Field(default=1024, ge=0)
    ttl_seconds: float | None = Field(default=None, gt=0)


class CacheEntry(BaseModel):
    """A single cached LLM response."""

    response: LLMResponse
    created_at: float = Field(description="Monotonic timestamp of insertion (time.monotonic()).")
    hit_count: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}


class CacheStats(BaseModel):
    """Snapshot of cache performance counters."""

    hits: int = Field(default=0, ge=0, description="Requests served from cache.")
    misses: int = Field(default=0, ge=0, description="Requests that bypassed the cache.")
    size: int = Field(default=0, ge=0, description="Current number of stored entries.")

    @property
    def total(self) -> int:
        """Total requests seen by the cache."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of requests that were cache hits (0.0-1.0)."""
        return self.hits / self.total if self.total else 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"size={self.size}, hit_rate={self.hit_rate:.1%})"
        )


class SemanticCacheConfig(BaseModel):
    """Configuration for the semantic similarity cache.

    Parameters
    ----------
    threshold:
        Minimum cosine similarity (0.0-1.0) required to declare a cache hit.
        Defaults to ``0.95`` (very strict — avoids false positives).
    max_size:
        Maximum entries before LRU eviction.  ``0`` means unlimited.
    ttl_seconds:
        Optional TTL; ``None`` disables expiry.
    """

    threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    max_size: int = Field(default=512, ge=0)
    ttl_seconds: float | None = Field(default=None, gt=0)


class SemanticCacheEntry(BaseModel):
    """One entry in the semantic cache, pairing an embedding with a response."""

    vector: list[float] = Field(description="Embedding of the original query.")
    response: LLMResponse
    created_at: float
    hit_count: int = Field(default=0, ge=0)

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context: Any) -> None:
        # Store vector as plain list for easy serialisation / comparison.
        self.vector = list(self.vector)
