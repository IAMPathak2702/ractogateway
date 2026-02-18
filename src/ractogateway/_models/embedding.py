"""Typed input / output models for embedding calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Validated input for ``embed`` / ``aembed`` calls.

    Example::

        config = EmbeddingConfig(texts=["Hello world", "Goodbye world"])
        response = kit.embed(config)
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        description="One or more texts to embed.",
    )
    model: str | None = Field(
        default=None,
        description="Override the kit-level embedding model for this call.",
    )
    dimensions: int | None = Field(
        default=None,
        description="Output dimensionality (text-embedding-3-* and Gemini).",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific pass-through kwargs.",
    )


class EmbeddingVector(BaseModel):
    """A single embedding result."""

    index: int
    text: str
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    """Unified response from an embedding call."""

    vectors: list[EmbeddingVector]
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    raw: Any = Field(default=None)

    model_config = {"arbitrary_types_allowed": True}
