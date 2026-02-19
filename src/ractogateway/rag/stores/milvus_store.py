"""Milvus / Zilliz vector store (lazy import).

Install with:  pip install ractogateway[rag-milvus]
"""

from __future__ import annotations

from typing import Any


def _require_pymilvus() -> Any:
    try:
        from pymilvus import (  # noqa: PLC0415
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            MilvusException,
            connections,
            utility,
        )
    except ImportError as exc:
        raise ImportError(
            "MilvusStore requires the 'pymilvus' package. "
            "Install it with:  pip install ractogateway[rag-milvus]"
        ) from exc
    return Collection, CollectionSchema, DataType, FieldSchema, MilvusException, connections, utility


from ractogateway.rag._models.document import Chunk, ChunkMetadata
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


class MilvusStore(BaseVectorStore):
    """Vector store backed by Milvus or Zilliz Cloud.

    Parameters
    ----------
    collection:
        Milvus collection name.
    host:
        Milvus server host (default ``"localhost"``).
    port:
        Milvus server port (default ``19530``).
    uri:
        Zilliz Cloud URI (overrides host/port when set).
    token:
        Zilliz Cloud API token.
    dimension:
        Embedding dimension.  Inferred on first add.
    metric_type:
        ``"IP"`` (inner product / cosine) or ``"L2"``.
    batch_size:
        Vectors per insert batch.
    """

    def __init__(
        self,
        collection: str = "ractogateway",
        *,
        host: str = "localhost",
        port: int = 19530,
        uri: str | None = None,
        token: str | None = None,
        dimension: int | None = None,
        metric_type: str = "IP",
        batch_size: int = 100,
    ) -> None:
        self._collection_name = collection
        self._host = host
        self._port = port
        self._uri = uri
        self._token = token
        self._dim = dimension
        self._metric_type = metric_type
        self._batch_size = batch_size
        self._collection: Any = None
        self._connected = False

    def _connect(self) -> None:
        if self._connected:
            return
        _, _, _, _, _, connections, _ = _require_pymilvus()
        if self._uri:
            connections.connect("default", uri=self._uri, token=self._token or "")
        else:
            connections.connect("default", host=self._host, port=self._port)
        self._connected = True

    def _init(self, dim: int | None = None) -> None:
        self._connect()
        Collection, CollectionSchema, DataType, FieldSchema, _, _, utility = _require_pymilvus()

        if dim is not None:
            self._dim = dim

        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            self._collection.load()
            return

        if self._dim is None:
            return  # will be created on first add

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="chunk_index", dtype=DataType.INT32),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
            ],
            description="RactoGateway RAG chunks",
        )
        self._collection = Collection(self._collection_name, schema)
        self._collection.create_index(
            field_name="embedding",
            index_params={"metric_type": self._metric_type, "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )
        self._collection.load()

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        dim = len(chunks[0].embedding)  # type: ignore[arg-type]
        self._init(dim)
        for i in range(0, len(chunks), self._batch_size):
            batch = chunks[i : i + self._batch_size]
            self._collection.insert([
                [c.chunk_id for c in batch],
                [c.doc_id for c in batch],
                [c.content[:65535] for c in batch],
                [c.metadata.source[:512] for c in batch],
                [c.metadata.chunk_index for c in batch],
                [c.embedding for c in batch],
            ])
        self._collection.flush()

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if self._collection is None:
            self._init()
        if self._collection is None:
            return []
        expr = None
        if filters:
            parts = [f'{k} == "{v}"' if isinstance(v, str) else f"{k} == {v}" for k, v in filters.items()]
            expr = " && ".join(parts)
        search_params = {"metric_type": self._metric_type, "params": {"nprobe": 10}}
        hits = self._collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["chunk_id", "doc_id", "content", "source", "chunk_index"],
        )
        results: list[RetrievalResult] = []
        for rank, hit in enumerate(hits[0], start=1):
            chunk = Chunk(
                chunk_id=hit.entity.get("chunk_id", str(hit.id)),
                doc_id=hit.entity.get("doc_id", ""),
                content=hit.entity.get("content", ""),
                metadata=ChunkMetadata(
                    source=hit.entity.get("source", ""),
                    chunk_index=int(hit.entity.get("chunk_index", 0)),
                    total_chunks=0,
                    start_char=0,
                    end_char=len(hit.entity.get("content", "")),
                    doc_id=hit.entity.get("doc_id", ""),
                ),
            )
            results.append(RetrievalResult(chunk=chunk, score=float(hit.distance), rank=rank))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        if self._collection is None:
            self._init()
        if self._collection is None:
            return
        ids_str = ", ".join(f'"{cid}"' for cid in chunk_ids)
        self._collection.delete(f"chunk_id in [{ids_str}]")

    def clear(self) -> None:
        if self._collection is None:
            self._init()
        if self._collection is not None:
            self._collection.drop()
        self._collection = None
        self._dim = None

    def count(self) -> int:
        if self._collection is None:
            self._init()
        if self._collection is None:
            return 0
        return int(self._collection.num_entities)
