"""Qdrant vector store (lazy import).

Install with:  pip install ractogateway[rag-qdrant]
"""

from __future__ import annotations

from typing import Any


def _require_qdrant() -> Any:
    try:
        from qdrant_client import QdrantClient  # noqa: PLC0415
        from qdrant_client.models import (  # noqa: PLC0415
            Distance,
            PointStruct,
            VectorParams,
        )
    except ImportError as exc:
        raise ImportError(
            "QdrantStore requires the 'qdrant-client' package. "
            "Install it with:  pip install ractogateway[rag-qdrant]"
        ) from exc
    return QdrantClient, Distance, PointStruct, VectorParams


from ractogateway.rag._models.document import Chunk, ChunkMetadata
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


class QdrantStore(BaseVectorStore):
    """Vector store backed by Qdrant.

    Parameters
    ----------
    collection:
        Qdrant collection name.
    url:
        Qdrant server URL.  ``None`` = in-memory.
    api_key:
        Qdrant cloud API key (optional).
    distance:
        ``"cosine"``, ``"euclid"``, or ``"dot"``.
    dimension:
        Vector dimension.  Inferred on first add if ``None``.
    batch_size:
        Points per upsert batch.
    """

    def __init__(
        self,
        collection: str = "ractogateway",
        *,
        url: str | None = None,
        api_key: str | None = None,
        distance: str = "cosine",
        dimension: int | None = None,
        batch_size: int = 100,
    ) -> None:
        self._collection = collection
        self._url = url
        self._api_key = api_key
        self._distance_str = distance
        self._dim = dimension
        self._batch_size = batch_size
        self._client: Any = None

    def _init(self, dim: int | None = None) -> None:
        QdrantClient, Distance, PointStruct, VectorParams = _require_qdrant()
        if self._client is None:
            kw: dict[str, Any] = {}
            if self._url:
                kw["url"] = self._url
            else:
                kw["location"] = ":memory:"
            if self._api_key:
                kw["api_key"] = self._api_key
            self._client = QdrantClient(**kw)

        if dim is not None and self._dim is None:
            self._dim = dim

        dist_map = {
            "cosine": Distance.COSINE,
            "euclid": Distance.EUCLID,
            "dot": Distance.DOT,
        }
        dist = dist_map.get(self._distance_str, Distance.COSINE)

        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing and self._dim:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=dist),
            )

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        dim = len(chunks[0].embedding)  # type: ignore[arg-type]
        self._init(dim)
        _, _, PointStruct, _ = _require_qdrant()
        points = [
            PointStruct(
                id=c.chunk_id,
                vector=c.embedding,
                payload={
                    "doc_id": c.doc_id,
                    "content": c.content,
                    "source": c.metadata.source,
                    "chunk_index": c.metadata.chunk_index,
                },
            )
            for c in chunks
        ]
        for i in range(0, len(points), self._batch_size):
            self._client.upsert(
                collection_name=self._collection,
                points=points[i : i + self._batch_size],
            )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self._init()
        kw: dict[str, Any] = {
            "collection_name": self._collection,
            "query_vector": embedding,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": True,
        }
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue  # noqa: PLC0415

            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
            ]
            kw["query_filter"] = Filter(must=conditions)

        hits = self._client.search(**kw)
        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            chunk = Chunk(
                chunk_id=str(hit.id),
                doc_id=payload.get("doc_id", ""),
                content=payload.get("content", ""),
                embedding=list(hit.vector) if hit.vector else None,
                metadata=ChunkMetadata(
                    source=payload.get("source", ""),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    total_chunks=0,
                    start_char=0,
                    end_char=len(payload.get("content", "")),
                    doc_id=payload.get("doc_id", ""),
                ),
            )
            results.append(RetrievalResult(chunk=chunk, score=float(hit.score), rank=rank))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        self._init()
        from qdrant_client.models import PointIdsList  # noqa: PLC0415

        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=chunk_ids),
        )

    def clear(self) -> None:
        self._init()
        self._client.delete_collection(self._collection)
        self._dim = None  # reset so it will be recreated on next add
        self._init()

    def count(self) -> int:
        self._init()
        info = self._client.get_collection(self._collection)
        return info.points_count or 0
