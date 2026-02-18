"""Sentence-aware chunker — uses NLTK ``sent_tokenize`` (lazy import).

Install with:  pip install ractogateway[rag-nlp]
"""

from __future__ import annotations

from typing import Any


def _require_nltk() -> Any:
    try:
        import nltk  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "SentenceChunker requires the 'nltk' package. "
            "Install it with:  pip install ractogateway[rag-nlp]"
        ) from exc
    # Download punkt tokenizer data silently if not present
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


class SentenceChunker(BaseChunker):
    """Split text into groups of sentences using NLTK.

    Parameters
    ----------
    sentences_per_chunk:
        Number of sentences per chunk.
    overlap_sentences:
        Number of sentences to repeat at the start of the next chunk.
    language:
        Language for the NLTK sentence tokenizer (default: ``"english"``).
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        overlap_sentences: int = 1,
        language: str = "english",
    ) -> None:
        if overlap_sentences >= sentences_per_chunk:
            raise ValueError(
                f"overlap_sentences ({overlap_sentences}) must be "
                f"< sentences_per_chunk ({sentences_per_chunk})"
            )
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap_sentences = overlap_sentences
        self.language = language

    def chunk(self, document: Document) -> list[Chunk]:
        nltk = _require_nltk()
        sentences: list[str] = nltk.sent_tokenize(document.content, language=self.language)

        step = self.sentences_per_chunk - self.overlap_sentences
        groups: list[list[str]] = []
        i = 0
        while i < len(sentences):
            groups.append(sentences[i : i + self.sentences_per_chunk])
            i += step

        total = len(groups)
        chunks: list[Chunk] = []
        cursor = 0
        for idx, group in enumerate(groups):
            text = " ".join(group)
            start = document.content.find(group[0], cursor)
            if start == -1:
                start = cursor
            end = start + len(text)
            cursor = start + len(" ".join(group[: self.overlap_sentences]))
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
