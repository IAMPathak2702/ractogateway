"""Shared data models for the batch-processing subsystem."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ractogateway.adapters.base import LLMResponse


class BatchStatus(str, Enum):
    """Processing state of a batch job.

    Maps to the union of OpenAI and Anthropic batch status strings.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


class BatchItem(BaseModel):
    """A single request within a batch job.

    Parameters
    ----------
    custom_id:
        User-supplied identifier used to correlate results.  Must be unique
        within a batch.
    user_message:
        The end-user's query string (equivalent to ``ChatConfig.user_message``).
    temperature:
        Sampling temperature.  Defaults to ``0.0``.
    max_tokens:
        Maximum tokens for the completion.  Defaults to ``4096``.
    extra:
        Provider-specific pass-through kwargs.
    """

    custom_id: str = Field(min_length=1, description="Unique identifier for this request.")
    user_message: str = Field(min_length=1, description="The end-user's query.")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    extra: dict[str, Any] = Field(default_factory=dict)


class BatchJobInfo(BaseModel):
    """Metadata about a submitted batch job.

    Returned by ``submit_batch()`` and ``poll_status()``.
    """

    job_id: str = Field(description="Provider-assigned batch job identifier.")
    provider: str = Field(description="Provider name (e.g. 'openai', 'anthropic').")
    status: BatchStatus
    created_at: float = Field(description="Unix timestamp of job creation.")
    request_count: int = Field(default=0, ge=0)
    raw: Any = Field(default=None, description="Raw provider response object.")

    model_config = {"arbitrary_types_allowed": True}


class BatchResult(BaseModel):
    """The outcome of a single :class:`BatchItem`.

    A result is always present in the ``results`` list returned by
    ``get_results()``; check ``error`` to detect failures.
    """

    custom_id: str = Field(description="Matches the ``BatchItem.custom_id``.")
    response: LLMResponse | None = Field(
        default=None,
        description="Parsed LLM response.  ``None`` when the request failed.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error message when the request failed.",
    )
    raw: Any = Field(default=None, description="Raw provider result object.")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def ok(self) -> bool:
        """``True`` when the request succeeded (no error, response present)."""
        return self.error is None and self.response is not None
