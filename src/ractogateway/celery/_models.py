"""Pydantic models for the Celery task-queue subsystem."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle state of a Celery task.

    Maps directly to Celery's native task states so you can compare them without
    importing Celery's own constants.
    """

    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


class TaskResult(BaseModel):
    """Unified result returned by :meth:`~RactoCeleryWorker.wait` and
    :meth:`~RactoCeleryWorker.get_result`.

    Parameters
    ----------
    task_id:
        The Celery task UUID.
    status:
        Current :class:`TaskStatus`.
    result:
        Deserialised task output on success:

        * For :meth:`~RactoCeleryWorker.generate` — a ``dict`` matching
          :class:`~ractogateway.adapters.base.LLMResponse`.model_dump()`.
        * For :meth:`~RactoCeleryWorker.ingest_document` — a ``list`` of
          :class:`~ractogateway.rag._models.Chunk`.model_dump()` dicts.
    error:
        Exception message string on failure; ``None`` on success.
    """

    task_id: str = Field(description="Celery task UUID.")
    status: TaskStatus
    result: Any | None = Field(
        default=None,
        description="Deserialised task output (LLMResponse dict or list of Chunk dicts).",
    )
    error: str | None = Field(default=None, description="Exception message on failure.")

    model_config = {"arbitrary_types_allowed": True}

    @property
    def ok(self) -> bool:
        """``True`` when the task succeeded and produced a result."""
        return self.status == TaskStatus.SUCCESS and self.error is None


class RetryConfig(BaseModel):
    """Exponential-backoff retry policy for :class:`RactoCeleryWorker` tasks.

    The backoff delay for attempt *n* (0-based) is::

        delay = min(initial_delay_s * backoff_factor ** n, max_delay_s)

    With the defaults this gives: 2 s → 4 s → 8 s (then stops at max_retries=3).

    Parameters
    ----------
    max_retries:
        Maximum number of retry attempts after the first failure.
        ``0`` disables retries entirely.
    initial_delay_s:
        Countdown (seconds) before the first retry.
    backoff_factor:
        Multiplier applied on each successive retry.  Must be > 1.
    max_delay_s:
        Upper bound on the countdown (seconds).  Prevents extremely long waits
        after many retries.
    """

    max_retries: int = Field(default=3, ge=0)
    initial_delay_s: float = Field(default=2.0, gt=0)
    backoff_factor: float = Field(default=2.0, gt=1.0)
    max_delay_s: float = Field(default=300.0, gt=0)
