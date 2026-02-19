"""ChromaDB vector store (lazy import).

Install with:  pip install ractogateway[rag-chroma]
"""

from __future__ import annotations

from typing import Any


def _require_chromadb() -> Any:
    try:
        import chromadb  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ChromaStore requires the 'chromadb' package. "
            "Install it with:  pip install ractogateway[rag-chroma]"
        ) from exc
    return chromadb


from ractogateway.rag._models.document import Chunk
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


class ChromaStore(BaseVectorStore):
    """Vector store backed by ChromaDB.

    Supports both in-process (``path`` or ``None`` for ephemeral) and
    HTTP-client modes (``host`` + ``port``).

    Parameters
    ----------
    collection:
        Name of the ChromaDB collection.
    path:
        Persist directory for a local persistent client.  ``None`` = ephemeral.
    host:
        ChromaDB server host (enables HTTP client mode).
    port:
        ChromaDB server port (default 8000).
    distance_function:
        ``"cosine"``, ``"l2"``, or ``"ip"`` (inner product).
    """

    def __init__(
        self,
        collection: str = "ractogateway",
        *,
        path: str | None = None,
        host: str | None = None,
        port: int = 8000,
        distance_function: str = "cosine",
    ) -> None:
        self._collection_name = collection
        self._path = path
        self._host = host
        self._port = port
        self._distance = distance_function
        self._client: Any = None
        self._collection: Any = None

    def _init(self) -> None:
        if self._collection is not None:
            return
        chromadb = _require_chromadb()
        if self._host:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
        elif self._path:
            self._client = chromadb.PersistentClient(path=self._path)
        else:
            self._client = chromadb.EphemeralClient()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._distance},
        )

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        self._init()
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.content for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "source": c.metadata.source,
                    "chunk_index": c.metadata.chunk_index,
                    **{k: str(v) for k, v in c.metadata.extra.items()},
                }
                for c in chunks
            ],
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        self._init()
        kw: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if filters:
            kw["where"] = filters
        raw = self._collection.query(**kw)

        results: list[RetrievalResult] = []
        for rank, (doc_id, doc, meta, dist, emb) in enumerate(
            zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["metadatas"][0],
                raw["distances"][0],
                raw["embeddings"][0],
            ),
            start=1,
        ):
            score = 1.0 - dist if self._distance == "cosine" else -dist
            from ractogateway.rag._models.document import ChunkMetadata  # noqa: PLC0415

            chunk = Chunk(
                chunk_id=doc_id,
                doc_id=meta.get("doc_id", ""),
                content=doc,
                embedding=list(emb),
                metadata=ChunkMetadata(
                    source=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    total_chunks=0,
                    start_char=0,
                    end_char=len(doc),
                    doc_id=meta.get("doc_id", ""),
                ),
            )
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        self._init()
        self._collection.delete(ids=chunk_ids)

    def clear(self) -> None:
        self._init()
        self._client.delete_collection(self._collection_name)
        self._collection = None
        self._init()

    def count(self) -> int:
        self._init()
        return int(self._collection.count())
