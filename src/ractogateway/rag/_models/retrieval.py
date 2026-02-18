"""Retrieval and RAG response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ractogateway.adapters.base import LLMResponse
from ractogateway.rag._models.document import Chunk


class RetrievalConfig(BaseModel):
    """Input parameters for a vector-store search."""

    query: str = Field(description="Natural-language query to search for.")
    top_k: int = Field(default=5, ge=1, description="Maximum number of results to return.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters (store-dependent format).",
    )


class RetrievalResult(BaseModel):
    """A single retrieved chunk together with its relevance score."""

    chunk: Chunk
    score: float = Field(description="Similarity / relevance score (higher = more relevant).")
    rank: int = Field(description="1-based rank among all results.")


class RAGResponse(BaseModel):
    """Combined output from a RAG query (retrieval + generation).

    Attributes
    ----------
    answer:
        The LLM-generated response, as a standard ``LLMResponse``.
    sources:
        The retrieved chunks that were injected as context.
    query:
        The original user question.
    context_used:
        The verbatim context string that was injected into the prompt.
    """

    answer: LLMResponse
    sources: list[RetrievalResult]
    query: str
    context_used: str = Field(description="Context string injected into the LLM prompt.")

    model_config = {"arbitrary_types_allowed": True}
