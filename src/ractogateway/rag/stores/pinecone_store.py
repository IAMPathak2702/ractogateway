"""Pinecone vector store (lazy import).

Install with:  pip install ractogateway[rag-pinecone]
"""

from __future__ import annotations

from typing import Any


def _require_pinecone() -> Any:
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise ImportError(
            "PineconeStore requires the 'pinecone-client' package. "
            "Install it with:  pip install ractogateway[rag-pinecone]"
        ) from exc
    return Pinecone


from ractogateway.rag._models.document import Chunk, ChunkMetadata
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


class PineconeStore(BaseVectorStore):
    """Vector store backed by Pinecone cloud.

    Parameters
    ----------
    index_name:
        Name of the Pinecone index (must already exist).
    api_key:
        Pinecone API key.  Falls back to ``PINECONE_API_KEY`` env var.
    namespace:
        Pinecone namespace for logical data isolation.
    environment:
        Deprecated Pinecone environment string (for legacy pod-based indexes).
    batch_size:
        Number of vectors per upsert batch.
    """

    def __init__(
        self,
        index_name: str,
        *,
        api_key: str | None = None,
        namespace: str = "",
        batch_size: int = 100,
    ) -> None:
        import os

        self._index_name = index_name
        self._api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self._namespace = namespace
        self._batch_size = batch_size
        self._index: Any = None

    def _init(self) -> None:
        if self._index is not None:
            return
        pinecone_cls = _require_pinecone()
        kw: dict[str, Any] = {}
        if self._api_key:
            kw["api_key"] = self._api_key
        pc = pinecone_cls(**kw)
        self._index = pc.Index(self._index_name)

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        self._init()
        vectors = [
            {
                "id": c.chunk_id,
                "values": c.embedding,
                "metadata": {
                    "doc_id": c.doc_id,
                    "content": c.content[:1000],  # Pinecone metadata size limit
                    "source": c.metadata.source,
                    "chunk_index": c.metadata.chunk_index,
                },
            }
            for c in chunks
        ]
        for i in range(0, len(vectors), self._batch_size):
            self._index.upsert(
                vectors=vectors[i : i + self._batch_size],
                namespace=self._namespace,
            )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self._init()
        kw: dict[str, Any] = {
            "vector": embedding,
            "top_k": top_k,
            "namespace": self._namespace,
            "include_metadata": True,
            "include_values": True,
        }
        if filters:
            kw["filter"] = filters
        raw = self._index.query(**kw)

        results: list[RetrievalResult] = []
        for rank, match in enumerate(raw.matches, start=1):
            meta = match.metadata or {}
            chunk = Chunk(
                chunk_id=match.id,
                doc_id=meta.get("doc_id", ""),
                content=meta.get("content", ""),
                embedding=list(match.values) if match.values else None,
                metadata=ChunkMetadata(
                    source=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=0,
                    start_char=0,
                    end_char=len(meta.get("content", "")),
                    doc_id=meta.get("doc_id", ""),
                ),
            )
            results.append(RetrievalResult(chunk=chunk, score=float(match.score), rank=rank))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        self._init()
        self._index.delete(ids=chunk_ids, namespace=self._namespace)

    def clear(self) -> None:
        self._init()
        self._index.delete(delete_all=True, namespace=self._namespace)

    def count(self) -> int:
        self._init()
        stats = self._index.describe_index_stats()
        ns_stats = stats.namespaces.get(self._namespace)
        return ns_stats.vector_count if ns_stats else 0
