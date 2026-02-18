"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ractogateway.rag._models.document import Chunk
from ractogateway.rag._models.retrieval import RetrievalResult


class BaseVectorStore(ABC):
    """Persist and search embedding vectors.

    All vector stores share the same interface: :meth:`add`, :meth:`search`,
    :meth:`delete`, :meth:`clear`, and :meth:`count`.  The underlying storage
    backend is determined by the concrete subclass.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add *chunks* (with embeddings) to the store.

        Parameters
        ----------
        chunks:
            Chunks to index.  Each chunk must have a non-``None`` ``embedding``.

        Raises
        ------
        ValueError
            If any chunk has ``embedding=None``.
        """

    @abstractmethod
    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for the *top_k* most similar chunks.

        Parameters
        ----------
        embedding:
            Query embedding vector.
        top_k:
            Number of results to return.
        filters:
            Optional metadata filters (store-specific format).

        Returns
        -------
        list[RetrievalResult]
            Ranked list of results (rank 1 = most similar).
        """

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks with the given IDs from the store."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of indexed chunks."""

    # ------------------------------------------------------------------
    # Shared validation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _require_embeddings(chunks: list[Chunk]) -> None:
        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"Chunks must have embeddings before adding to a vector store. "
                f"Missing embeddings on chunk_ids: {missing[:5]}"
                + (" (and more)" if len(missing) > 5 else "")
            )
