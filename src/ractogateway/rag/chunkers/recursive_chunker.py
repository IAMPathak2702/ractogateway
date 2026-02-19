"""Recursive character text splitter (LangChain-style).

Tries progressively finer separators (``"\\n\\n"``, ``"\\n"``, ``". "``,
``" "`` and finally character-by-character) until every piece fits within
``chunk_size``.
"""

from __future__ import annotations

from ractogateway.rag._models.document import Chunk, ChunkMetadata, Document
from ractogateway.rag.chunkers.base import BaseChunker

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """Split text recursively using a priority list of separators.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters of overlap between consecutive chunks.
    separators:
        Ordered list of separator strings to try.  The first separator that
        produces pieces within *chunk_size* is used.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or _DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.content, self.separators)

        # Merge small pieces into windows of chunk_size with overlap
        merged = self._merge(pieces)

        total = len(merged)
        chunks: list[Chunk] = []
        cursor = 0
        for idx, text in enumerate(merged):
            start = document.content.find(text, cursor)
            if start == -1:
                start = cursor
            end = start + len(text)
            cursor = max(start, end - self.overlap)
            chunks.append(
                Chunk(
                    doc_id=document.doc_id,
                    content=text,
                    metadata=ChunkMetadata(
                        source=document.source,
                        chunk_index=idx,
                        total_chunks=total,
                        start_char=start,
                        end_char=end,
                        doc_id=document.doc_id,
                        extra=dict(document.metadata),
                    ),
                )
            )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text until all pieces fit within chunk_size."""
        if not separators or len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        sep = separators[0]
        remaining = separators[1:]

        if sep == "":
            # Character-level fallback
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.overlap)
                if text[i : i + self.chunk_size].strip()
            ]

        parts = text.split(sep)
        result: list[str] = []
        for raw_part in parts:
            part_text = raw_part.strip()
            if not part_text:
                continue
            if len(part_text) <= self.chunk_size:
                result.append(part_text)
            else:
                result.extend(self._split(part_text, remaining))
        return result

    def _merge(self, pieces: list[str]) -> list[str]:
        """Merge small pieces into chunks up to chunk_size, with overlap."""
        merged: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len + 1 > self.chunk_size and current_parts:
                merged.append(" ".join(current_parts))
                # Keep overlap
                overlap_text = " ".join(current_parts)[-self.overlap :]
                current_parts = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)
            current_parts.append(piece)
            current_len += piece_len + 1

        if current_parts:
            merged.append(" ".join(current_parts))
        return [m for m in merged if m.strip()]
