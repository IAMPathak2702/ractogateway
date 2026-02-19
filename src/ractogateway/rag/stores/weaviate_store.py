"""Weaviate vector store (lazy import).

Install with:  pip install ractogateway[rag-weaviate]
"""

from __future__ import annotations

from typing import Any


def _require_weaviate() -> Any:
    try:
        import weaviate  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "WeaviateStore requires the 'weaviate-client' package. "
            "Install it with:  pip install ractogateway[rag-weaviate]"
        ) from exc
    return weaviate


from ractogateway.rag._models.document import Chunk, ChunkMetadata
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


class WeaviateStore(BaseVectorStore):
    """Vector store backed by Weaviate.

    Supports embedded (local, no server needed), local server, and
    Weaviate Cloud (WCS) connections.

    Parameters
    ----------
    class_name:
        Weaviate class (collection) name.
    url:
        Weaviate server URL.  ``None`` = use embedded Weaviate.
    api_key:
        Weaviate Cloud API key.
    additional_headers:
        Extra HTTP headers (e.g. for OpenAI API key pass-through to Weaviate).
    distance_metric:
        ``"cosine"`` or ``"l2-squared"``.
    batch_size:
        Objects per batch import.
    """

    def __init__(
        self,
        class_name: str = "RactoChunk",
        *,
        url: str | None = None,
        api_key: str | None = None,
        additional_headers: dict[str, str] | None = None,
        distance_metric: str = "cosine",
        batch_size: int = 100,
    ) -> None:
        self._class_name = class_name
        self._url = url
        self._api_key = api_key
        self._headers = additional_headers or {}
        self._distance_metric = distance_metric
        self._batch_size = batch_size
        self._client: Any = None

    def _init(self) -> None:
        if self._client is not None:
            return
        weaviate = _require_weaviate()
        if self._url:
            auth = weaviate.auth.AuthApiKey(self._api_key) if self._api_key else None
            self._client = weaviate.connect_to_custom(
                http_host=self._url.split("://")[-1].split(":")[0],
                http_port=int(self._url.split(":")[-1])
                if ":" in self._url.split("://")[-1]
                else 80,
                http_secure=self._url.startswith("https"),
                grpc_host=self._url.split("://")[-1].split(":")[0],
                grpc_port=50051,
                grpc_secure=False,
                auth_credentials=auth,
                headers=self._headers,
            )
        else:
            self._client = weaviate.connect_to_embedded(headers=self._headers)

        # Create class if needed
        if not self._client.collections.exists(self._class_name):
            from weaviate.classes.config import Configure, Property, DataType  # noqa: PLC0415

            self._client.collections.create(
                name=self._class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                ],
            )

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        self._init()
        collection = self._client.collections.get(self._class_name)
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                batch.add_object(
                    properties={
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "source": chunk.metadata.source,
                        "chunk_index": chunk.metadata.chunk_index,
                    },
                    vector=chunk.embedding,
                    uuid=chunk.chunk_id,
                )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self._init()
        from weaviate.classes.query import MetadataQuery  # noqa: PLC0415

        collection = self._client.collections.get(self._class_name)
        kw: dict[str, Any] = {
            "near_vector": embedding,
            "limit": top_k,
            "return_metadata": MetadataQuery(distance=True),
            "include_vector": True,
        }
        if filters:
            from weaviate.classes.query import Filter  # noqa: PLC0415

            weaviate_filters = None
            for k, v in filters.items():
                f = Filter.by_property(k).equal(v)
                weaviate_filters = f if weaviate_filters is None else weaviate_filters & f
            if weaviate_filters:
                kw["filters"] = weaviate_filters

        response = collection.query.near_vector(**kw)
        results: list[RetrievalResult] = []
        for rank, obj in enumerate(response.objects, start=1):
            props = obj.properties
            dist = obj.metadata.distance or 0.0
            score = 1.0 - dist
            chunk = Chunk(
                chunk_id=str(obj.uuid),
                doc_id=str(props.get("doc_id", "")),
                content=str(props.get("content", "")),
                embedding=list(obj.vector.get("default", [])) if obj.vector else None,
                metadata=ChunkMetadata(
                    source=str(props.get("source", "")),
                    chunk_index=int(props.get("chunk_index", 0)),
                    total_chunks=0,
                    start_char=0,
                    end_char=len(str(props.get("content", ""))),
                    doc_id=str(props.get("doc_id", "")),
                ),
            )
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        self._init()
        import uuid as _uuid  # noqa: PLC0415

        collection = self._client.collections.get(self._class_name)
        for cid in chunk_ids:
            collection.data.delete_by_id(_uuid.UUID(cid))

    def clear(self) -> None:
        self._init()
        self._client.collections.delete(self._class_name)
        self._client = None
        self._init()

    def count(self) -> int:
        self._init()
        collection = self._client.collections.get(self._class_name)
        agg = collection.aggregate.over_all(total_count=True)
        return agg.total_count or 0
