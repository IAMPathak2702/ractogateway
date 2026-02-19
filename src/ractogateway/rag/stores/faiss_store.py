"""FAISS vector store (lazy import).

Install with:  pip install ractogateway[rag-faiss]
"""

from __future__ import annotations

from typing import Any


def _require_faiss() -> Any:
    try:
        import faiss  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "FAISSStore requires the 'faiss-cpu' package. "
            "Install it with:  pip install ractogateway[rag-faiss]"
        ) from exc
    return faiss


from ractogateway.rag._models.document import Chunk, ChunkMetadata
from ractogateway.rag._models.retrieval import RetrievalResult
from ractogateway.rag.stores.base import BaseVectorStore


class FAISSStore(BaseVectorStore):
    """Vector store backed by Facebook AI Similarity Search (FAISS).

    Stores embeddings in a flat L2 or cosine (Inner Product) index.
    All data is in-memory; call :meth:`save` / :meth:`load` to persist.

    Parameters
    ----------
    dimension:
        Embedding dimension.  Inferred from the first :meth:`add` call if
        ``None``.
    index_type:
        ``"flat_l2"`` or ``"flat_ip"`` (inner product / cosine when normalised).
    """

    def __init__(
        self,
        dimension: int | None = None,
        index_type: str = "flat_ip",
    ) -> None:
        self._dim = dimension
        self._index_type = index_type
        self._index: Any = None
        self._chunks: list[Chunk] = []  # parallel list to FAISS index

    # ------------------------------------------------------------------
    # Lazy index init
    # ------------------------------------------------------------------

    def _init_index(self, dim: int) -> None:
        if self._index is not None:
            return
        faiss = _require_faiss()
        self._dim = dim
        if self._index_type == "flat_l2":
            self._index = faiss.IndexFlatL2(dim)
        else:
            self._index = faiss.IndexFlatIP(dim)

    def _to_numpy(self, vectors: list[list[float]]) -> Any:
        import numpy as np  # noqa: PLC0415

        return np.array(vectors, dtype="float32")

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk]) -> None:
        self._require_embeddings(chunks)
        first_embedding = chunks[0].embedding
        if first_embedding is None:
            raise ValueError("Chunks must have embeddings before adding to FAISSStore.")
        dim = len(first_embedding)
        self._init_index(dim)
        vectors: list[list[float]] = []
        for chunk in chunks:
            chunk_embedding = chunk.embedding
            if chunk_embedding is None:
                raise ValueError(
                    "Chunks must have embeddings before adding to FAISSStore."
                )
            vectors.append(chunk_embedding)
        self._index.add(self._to_numpy(vectors))
        self._chunks.extend(chunks)

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if self._index is None or self._index.ntotal == 0:
            return []
        import numpy as np  # noqa: PLC0415

        query = np.array([embedding], dtype="float32")
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query, k)

        results: list[RetrievalResult] = []
        rank = 1
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = self._chunks[idx]
            if filters:
                match = all(
                    chunk.metadata.extra.get(fk) == fv
                    or getattr(chunk.metadata, fk, None) == fv
                    for fk, fv in filters.items()
                )
                if not match:
                    continue
            score = float(dist)
            results.append(RetrievalResult(chunk=chunk, score=score, rank=rank))
            rank += 1
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        id_set = set(chunk_ids)
        keep_idx = [i for i, c in enumerate(self._chunks) if c.chunk_id not in id_set]
        kept_chunks = [self._chunks[i] for i in keep_idx]
        # Rebuild index
        self._index = None
        self._chunks = []
        if kept_chunks:
            self.add(kept_chunks)

    def clear(self) -> None:
        self._index = None
        self._chunks = []

    def count(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the FAISS index to *path*.index and chunks to *path*.chunks."""
        import json  # noqa: PLC0415
        import pickle  # noqa: S403,PLC0415

        faiss = _require_faiss()
        faiss.write_index(self._index, f"{path}.index")
        with open(f"{path}.chunks", "wb") as f:
            pickle.dump(self._chunks, f)  # noqa: S301

    def load(self, path: str) -> None:
        """Load a previously saved index from *path*."""
        import pickle  # noqa: S403,PLC0415

        faiss = _require_faiss()
        self._index = faiss.read_index(f"{path}.index")
        with open(f"{path}.chunks", "rb") as f:
            self._chunks = pickle.load(f)  # noqa: S301
