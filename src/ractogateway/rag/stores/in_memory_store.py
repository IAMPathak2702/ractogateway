"""In-memory vector store — pure Python, zero extra dependencies.

Uses brute-force cosine similarity over a list of stored vectors.
Suitable for development, testing, and small corpora (< 10k chunks).
"""

from __future__ import annotations

import math
from typing import Any

from ractogateway.rag._models.document import Chunk
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore(BaseVectorStore):
    """Pure-Python brute-force vector store — no extra dependencies.

    This store keeps all chunks and their embeddings in memory.  It is not
    suitable for production-scale corpora but requires no installation.

    Parameters
    ----------
    similarity:
        Similarity function to use.  Currently only ``"cosine"`` is supported.
    """

    def __init__(self, similarity: str = "cosine") -> None:
        if similarity != "cosine":
            raise ValueError(
                f"Unsupported similarity: {similarity!r}. Only 'cosine' is supported."
            )
        self._chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        self._chunks.extend(chunks)

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        candidates = self._chunks
        if filters:
            candidates = [
                c
                for c in candidates
                if all(
                    c.metadata.extra.get(k) == v or getattr(c.metadata, k, None) == v
                    for k, v in filters.items()
                )
            ]

        scored: list[tuple[Chunk, float]] = []
        for chunk in candidates:
            chunk_embedding = chunk.embedding
            if chunk_embedding is None:
                continue
            raw_score = _cosine_similarity(embedding, chunk_embedding)
            # Normalise cosine similarity from [-1, 1] to a non-negative relevance score.
            scored.append((chunk, max(raw_score, 0.0)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank + 1)
            for rank, (chunk, score) in enumerate(top)
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        id_set = set(chunk_ids)
        self._chunks = [c for c in self._chunks if c.chunk_id not in id_set]

    def clear(self) -> None:
        self._chunks = []

    def count(self) -> int:
        return len(self._chunks)
