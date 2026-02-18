"""Fixed-size character chunker with configurable overlap."""

from __future__ import annotations

from ractogateway.rag._models.document import Chunk, ChunkMetadata, Document
from ractogateway.rag.chunkers.base import BaseChunker


class FixedChunker(BaseChunker):
    """Split text into fixed-size character windows with overlap.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters to repeat at the start of the next chunk.
        Must be less than *chunk_size*.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50) -> None:
        if overlap >= chunk_size:
            raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        step = self.chunk_size - self.overlap
        positions: list[int] = list(range(0, max(1, len(text)), step))
        raw_chunks = [(pos, text[pos : pos + self.chunk_size]) for pos in positions]

        # Drop empty trailing chunks
        raw_chunks = [(s, t) for s, t in raw_chunks if t.strip()]

        total = len(raw_chunks)
        return [
            Chunk(
                doc_id=document.doc_id,
                content=chunk_text,
                metadata=ChunkMetadata(
                    source=document.source,
                    chunk_index=idx,
                    total_chunks=total,
                    start_char=start,
                    end_char=min(start + self.chunk_size, len(text)),
                    doc_id=document.doc_id,
                    extra=dict(document.metadata),
                ),
            )
            for idx, (start, chunk_text) in enumerate(raw_chunks)
        ]
