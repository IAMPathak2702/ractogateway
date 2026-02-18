"""Core document and chunk models for RAG.

Every piece of content in the RAG pipeline is represented as a ``Document``
(raw, as loaded from a file) or a ``Chunk`` (a processed, embeddable slice of
a document).  Both are strict Pydantic models with no unvalidated fields.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid.uuid4())


class ChunkMetadata(BaseModel):
    """Provenance and positional data attached to every chunk."""

    source: str = Field(description="Absolute path or URL of the source file.")
    page: int | None = Field(default=None, description="Page number (1-based) for PDFs/docs.")
    chunk_index: int = Field(description="0-based index of this chunk within the document.")
    total_chunks: int = Field(description="Total number of chunks in the parent document.")
    start_char: int = Field(description="Character offset in the *processed* document where this chunk begins.")
    end_char: int = Field(description="Character offset where this chunk ends (exclusive).")
    doc_id: str = Field(description="UUID of the parent Document.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Caller-supplied metadata pass-through.")


class Document(BaseModel):
    """A raw document loaded from a file or supplied as plain text.

    Parameters
    ----------
    content:
        The full extracted text of the document.
    source:
        Absolute file path, URL, or a descriptive label (e.g. ``"manual"``).
    metadata:
        Free-form key/value pairs (file size, author, MIME type, …).
    doc_id:
        Auto-generated UUID; override only when you need stable IDs.
    """

    doc_id: str = Field(default_factory=_new_id)
    content: str = Field(description="Full extracted text content of the document.")
    source: str = Field(description="Absolute path, URL, or label that identifies the source.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A single embeddable slice of a document.

    Produced by a :class:`~ractogateway.rag.chunkers.base.BaseChunker` and
    enriched with an embedding vector by a
    :class:`~ractogateway.rag.embedders.base.BaseEmbedder`.
    """

    chunk_id: str = Field(default_factory=_new_id)
    doc_id: str = Field(description="UUID of the parent Document.")
    content: str = Field(description="Text content of this chunk (post-processing).")
    embedding: list[float] | None = Field(
        default=None,
        description="Dense embedding vector; populated after embed() is called.",
    )
    metadata: ChunkMetadata
