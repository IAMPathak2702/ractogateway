"""Abstract base class for text chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ractogateway.rag._models.document import Chunk, Document


class BaseChunker(ABC):
    """Split a :class:`~ractogateway.rag._models.document.Document` into
    a list of :class:`~ractogateway.rag._models.document.Chunk` objects.

    Each chunk preserves provenance (``doc_id``, ``chunk_index``,
    ``start_char``, ``end_char``) in its ``ChunkMetadata``.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split *document* into chunks.

        Parameters
        ----------
        document:
            The fully-loaded document to split.

        Returns
        -------
        list[Chunk]
            Ordered list of non-overlapping (or slightly overlapping) chunks.
        """
