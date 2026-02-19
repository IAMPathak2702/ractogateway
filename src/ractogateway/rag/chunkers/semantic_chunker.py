"""Semantic chunker — splits at embedding-space boundaries.

Uses cosine similarity between adjacent sentence embeddings to detect
topic shifts.  Requires an :class:`~ractogateway.rag.embedders.base.BaseEmbedder`
and NLTK ``sent_tokenize``.

Install with:  pip install ractogateway[rag-nlp]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ractogateway.rag.embedders.base import BaseEmbedder


def _require_nltk() -> Any:
    try:
        import nltk
    except ImportError as exc:
        raise ImportError(
            "SemanticChunker requires the 'nltk' package. "
            "Install it with:  pip install ractogateway[rag-nlp]"
        ) from exc
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    return nltk


from ractogateway.rag._models.document import Chunk, ChunkMetadata, Document
from ractogateway.rag.chunkers.base import BaseChunker


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(BaseChunker):
    """Split documents where the semantic similarity between adjacent
    sentences drops below a threshold.

    Parameters
    ----------
    embedder:
        Any :class:`~ractogateway.rag.embedders.base.BaseEmbedder` instance.
    threshold:
        Cosine similarity below which a split is inserted (default: ``0.5``).
    min_chunk_size:
        Minimum number of sentences per chunk (prevents ultra-fine splits).
    language:
        NLTK sentence tokenizer language.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float = 0.5,
        min_chunk_size: int = 2,
        language: str = "english",
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.min_chunk_size = min_chunk_size
        self.language = language

    def chunk(self, document: Document) -> list[Chunk]:
        nltk = _require_nltk()
        sentences = nltk.sent_tokenize(document.content, language=self.language)

        if len(sentences) <= self.min_chunk_size:
            return self._make_single_chunk(document, sentences)

        # Embed all sentences (batched)
        embeddings = self.embedder.embed(sentences)

        # Detect split points
        split_indices: list[int] = [0]
        current_group_size = 1
        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            if sim < self.threshold and current_group_size >= self.min_chunk_size:
                split_indices.append(i)
                current_group_size = 1
            else:
                current_group_size += 1
        split_indices.append(len(sentences))  # sentinel

        # Build chunks from split points
        groups = [
            sentences[split_indices[i] : split_indices[i + 1]]
            for i in range(len(split_indices) - 1)
        ]
        total = len(groups)
        chunks: list[Chunk] = []
        cursor = 0
        for idx, group in enumerate(groups):
            text = " ".join(group)
            start = document.content.find(group[0], cursor)
            if start == -1:
                start = cursor
            end = start + len(text)
            cursor = start
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

    def _make_single_chunk(self, document: Document, sentences: list[str]) -> list[Chunk]:
        text = " ".join(sentences)
        return [
            Chunk(
                doc_id=document.doc_id,
                content=text,
                metadata=ChunkMetadata(
                    source=document.source,
                    chunk_index=0,
                    total_chunks=1,
                    start_char=0,
                    end_char=len(text),
                    doc_id=document.doc_id,
                    extra=dict(document.metadata),
                ),
            )
        ]
