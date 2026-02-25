"""Telemetry data models — shared across tracer, metrics, and pricing."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """USD cost per 1 million tokens for a specific model.

    Parameters
    ----------
    input_per_million:
        Price in USD for 1 million **input** (prompt) tokens.
    output_per_million:
        Price in USD for 1 million **output** (completion) tokens.
    """

    input_per_million: float = Field(gt=0, description="USD per 1M input tokens.")
    output_per_million: float = Field(gt=0, description="USD per 1M output tokens.")


class SpanRecord(BaseModel):
    """In-memory span record — captured when ``RactoTracer(in_memory=True)``.

    Useful for unit tests: inspect ``.spans`` after a call and assert on
    attributes without requiring an external OTEL backend.

    Parameters
    ----------
    name:
        OTEL span name (``"llm.chat"`` or ``"llm.embed"``).
    provider:
        Provider string — ``"openai"``, ``"google"``, or ``"anthropic"``.
    model:
        Model identifier as passed to the kit (e.g. ``"gpt-4o"``).
    operation:
        Operation type — ``"chat"``, ``"stream"``, or ``"embed"``.
    latency_ms:
        Total wall-clock latency of the LLM call in milliseconds.
    input_tokens:
        Number of prompt / input tokens consumed.
    output_tokens:
        Number of completion / output tokens produced.
    cost_usd:
        Estimated cost in USD derived from the built-in pricing table.
    cache_hit:
        Which cache served the result: ``"exact"``, ``"semantic"``,
        or ``"miss"`` when the LLM API was actually called.
    tool_calls:
        Number of tool calls present in the response.
    status:
        ``"ok"`` on success, ``"error"`` on exception.
    error_type:
        Exception class name when ``status == "error"``, else ``None``.
    timestamp:
        Unix timestamp (``time.time()``) when the span was recorded.
    """

    name: str
    provider: str
    model: str
    operation: str
    latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    cache_hit: str = Field(default="miss")
    tool_calls: int = Field(default=0, ge=0)
    status: str = Field(default="ok")
    error_type: str | None = Field(default=None)
    timestamp: float = Field(default_factory=time.time)
