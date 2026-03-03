"""Pydantic models for the PageIndexRAG pipeline."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field

from ractogateway.adapters.base import LLMResponse


def _new_id() -> str:
    return str(uuid.uuid4())


class PageEntry(BaseModel):
    """A single page (or fixed-size window) extracted from a document.

    Produced by :class:`~ractogateway.rag.page_index.pipeline.PageIndexRAG`
    during ingestion and stored in the in-process index.
    """

    entry_id: str = Field(default_factory=_new_id, description="Auto-generated UUID.")
    page_number: int | None = Field(
        default=None,
        description=(
            "1-based page number for page-aware sources (PDFs, Word docs). "
            "``None`` for sliding-window entries created from plain-text files."
        ),
    )
    content: str = Field(description="Full text of the page (post-processing).")
    source: str = Field(description="Absolute file path or descriptive label.")
    section_title: str | None = Field(
        default=None,
        description="First Markdown-style heading detected on the page, if any.",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "Top-N TF-weighted content terms extracted from the page. "
            "Used by the decision index for first-stage candidate selection."
        ),
    )
    doc_id: str = Field(description="UUID of the parent document.")
    char_count: int = Field(description="Length of ``content`` in characters.")
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Caller-supplied metadata forwarded from ``ingest()`` kwargs.",
    )


class PageIndexResult(BaseModel):
    """A single retrieved page together with its BM25 relevance score."""

    entry: PageEntry
    score: float = Field(description="Okapi BM25 score (higher = more relevant).")
    rank: int = Field(description="1-based rank within the current result list.")
    matched_terms: list[str] = Field(
        default_factory=list,
        description="Query tokens that matched this page's content.",
    )


class PageIndexResponse(BaseModel):
    """Full response from :meth:`PageIndexRAG.query` / :meth:`PageIndexRAG.aquery`."""

    answer: LLMResponse | None = Field(
        default=None,
        description=(
            "Generated answer from the LLM kit. "
            "``None`` when no ``llm_kit`` is configured on the pipeline."
        ),
    )
    sources: list[PageIndexResult] = Field(
        default_factory=list,
        description="Retrieved pages ranked by BM25 relevance.",
    )
    query: str = Field(description="The original question / query string.")
    context_used: str = Field(
        default="",
        description="Formatted context block that was supplied to the LLM.",
    )

    model_config = {"arbitrary_types_allowed": True}
