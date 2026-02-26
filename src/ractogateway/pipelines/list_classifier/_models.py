"""Data models for ListClassifierPipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ClassifierRateLimitExceededError(RuntimeError):
    """Raised when the rate limiter denies a request for a given user."""


class ClassifierUsage(BaseModel):
    """Token usage and retry statistics for a single pipeline call.

    Properties
    ----------
    total_tokens:
        Sum of *input_tokens* and *output_tokens* across all LLM attempts,
        including automatic retries triggered by invalid LLM responses.
        Zero on a cache hit (no LLM call was made).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class AuditEntry(BaseModel):
    """Immutable audit record emitted to the ``audit_logger`` after every call.

    Emitted regardless of whether the call was served from cache, hit an error,
    or was a live LLM classification.  Provides a complete picture of every
    request for compliance, debugging, and analytics.

    Fields
    ------
    timestamp:
        ISO 8601 UTC timestamp of when the call was made
        (e.g. ``"2026-02-26T14:23:01.456789Z"``).
    user_query:
        Original natural-language query.
    options_provided:
        Full candidate list shown to the LLM (including ``uncertain_label``
        if one was configured).
    selected:
        Option(s) chosen by the LLM, or empty on error.
    confidences:
        Per-selection confidence scores, or ``None``.
    all_scores:
        Score for every option (when ``score_all=True``), or ``None``.
    reasoning:
        LLM explanation (when ``include_reasoning=True``), or ``None``.
    fuzzy_corrected:
        ``True`` when the LLM returned a near-miss that was fuzzy-matched.
    uncertain:
        ``True`` when the LLM selected the ``uncertain_label`` option.
    cache_hit:
        ``"exact"`` or ``"semantic"`` when the result was served from cache;
        ``None`` when a live LLM call was made.
    user_id:
        User identifier passed to the pipeline (for rate limiting / audit).
    session_id:
        Conversation session identifier (for memory context).
    latency_ms:
        Wall-clock latency of the entire pipeline call in milliseconds
        (near-zero for cache hits).
    usage:
        Token usage dict — keys: ``input_tokens``, ``output_tokens``,
        ``total_tokens``, ``retry_count``.  All zero on cache hits.
    error:
        Non-``None`` when ``safe_mode=True`` and an exception occurred.
    """

    timestamp: str
    user_query: str
    options_provided: list[str]
    selected: list[str] = Field(default_factory=list)
    confidences: list[float] | None = None
    all_scores: dict[str, float] | None = None
    reasoning: str | None = None
    fuzzy_corrected: bool = False
    uncertain: bool = False
    cache_hit: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    latency_ms: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class ClassifierResult(BaseModel):
    """Result returned by :class:`~ractogateway.pipelines.ListClassifierPipeline`.

    All fields except ``user_query`` and ``options_provided`` have sensible
    defaults so that a partial result can be returned when ``safe_mode=True``
    and an error occurs mid-pipeline.

    Fields
    ------
    user_query:
        The original natural-language query passed to the pipeline.
    options_provided:
        The full list of candidate strings presented to the LLM (including
        the injected ``uncertain_label`` option if one was configured).
    selected:
        The option(s) chosen by the LLM, ordered by descending confidence.
        Empty when an error occurred and ``safe_mode=True``.
    confidences:
        Per-selection confidence scores in [0.0, 1.0], aligned with
        ``selected``.  ``None`` when ``include_confidence=False``.
    all_scores:
        Confidence score for *every* option in the list, keyed by option
        string.  ``None`` when ``score_all=False`` (the default).
    reasoning:
        Brief natural-language explanation produced by the LLM.
        ``None`` when ``include_reasoning=False``.
    fuzzy_corrected:
        ``True`` when the LLM returned a near-miss that was corrected by the
        built-in fuzzy matcher without consuming a retry.
    uncertain:
        ``True`` when the LLM selected the ``uncertain_label`` option,
        indicating no real option matched the query well enough.
    cache_hit:
        ``"exact"`` or ``"semantic"`` when served from cache; ``None`` for a
        live LLM call.
    usage:
        Aggregated token counts and retry statistics for this call.
    error:
        Non-``None`` only when ``safe_mode=True`` and an exception occurred.
        When ``error`` is set, ``selected`` will be empty.

    Examples
    --------
    >>> result.first                         # "Billing"
    >>> result.top_confidence                # 0.95
    >>> result.as_string()                   # "Billing, Account"
    >>> result.as_dict()                     # {"selected": ["Billing"], ...}
    >>> result.as_enum()["Billing"].value    # "Billing"
    >>> result.uncertain                     # False
    >>> result.cache_hit                     # "exact" | "semantic" | None
    """

    user_query: str
    options_provided: list[str]
    selected: list[str] = Field(default_factory=list)
    confidences: list[float] | None = None
    all_scores: dict[str, float] | None = None
    reasoning: str | None = None
    fuzzy_corrected: bool = False
    uncertain: bool = False
    cache_hit: str | None = None
    usage: ClassifierUsage = Field(default_factory=ClassifierUsage)
    error: str | None = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def first(self) -> str | None:
        """The first (highest-priority) selected option, or ``None`` if empty."""
        return self.selected[0] if self.selected else None

    @property
    def top_confidence(self) -> float | None:
        """Confidence score for the first selected option, or ``None``."""
        if self.confidences and self.selected:
            return self.confidences[0]
        return None

    @property
    def is_empty(self) -> bool:
        """``True`` when no options were selected (including error cases)."""
        return len(self.selected) == 0

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def as_string(self, separator: str = ", ") -> str:
        """Return selected options as a single joined string.

        Parameters
        ----------
        separator:
            Delimiter placed between items.  Default: ``", "``.

        Returns
        -------
        str
            E.g. ``"Billing, Account"`` for two selections.
        """
        return separator.join(self.selected)

    def as_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` with selected options and optional metadata.

        Always contains ``"selected"``.  ``"confidences"``, ``"all_scores"``,
        and ``"reasoning"`` are included only when they are non-``None``.

        Returns
        -------
        dict[str, Any]
            Example::

                {
                    "selected":    ["Billing", "Account"],
                    "confidences": [0.95, 0.82],
                    "all_scores":  {"Billing": 0.95, "Account": 0.82, "Sales": 0.12},
                    "reasoning":   "Both topics are mentioned explicitly.",
                }
        """
        d: dict[str, Any] = {"selected": self.selected}
        if self.confidences is not None:
            d["confidences"] = self.confidences
        if self.all_scores is not None:
            d["all_scores"] = self.all_scores
        if self.reasoning is not None:
            d["reasoning"] = self.reasoning
        return d

    def as_enum(self, name: str = "SelectedOptions") -> type[Enum]:
        """Return a dynamic Python :class:`enum.Enum` of the selected options.

        Parameters
        ----------
        name:
            Class name for the generated Enum.  Default: ``"SelectedOptions"``.

        Returns
        -------
        type[Enum]
            An Enum whose members have the option string as both name and value.

        Example
        -------
        >>> E = result.as_enum()
        >>> E["Billing"].value   # "Billing"
        """
        return Enum(name, {opt: opt for opt in self.selected})

    def top_n(self, n: int) -> list[str]:
        """Return the top-*n* selected options.

        Parameters
        ----------
        n:
            Maximum number of options to return.
        """
        return self.selected[:n]

    def score_for(self, option: str) -> float | None:
        """Return the confidence score for a specific option, or ``None``.

        Searches ``all_scores`` first (all options, when ``score_all=True``),
        then ``confidences`` for selected items.

        Parameters
        ----------
        option:
            The option string to look up.
        """
        if self.all_scores is not None:
            return self.all_scores.get(option)
        if self.confidences and option in self.selected:
            idx = self.selected.index(option)
            return self.confidences[idx]
        return None

    def to_audit_entry(
        self,
        *,
        timestamp: str,
        user_id: str | None = None,
        session_id: str | None = None,
        latency_ms: float = 0.0,
    ) -> AuditEntry:
        """Build an :class:`AuditEntry` from this result.

        Called automatically by the pipeline — exposed here so that users can
        reconstruct audit entries from stored results if needed.
        """
        return AuditEntry(
            timestamp=timestamp,
            user_query=self.user_query,
            options_provided=self.options_provided,
            selected=self.selected,
            confidences=self.confidences,
            all_scores=self.all_scores,
            reasoning=self.reasoning,
            fuzzy_corrected=self.fuzzy_corrected,
            uncertain=self.uncertain,
            cache_hit=self.cache_hit,
            user_id=user_id,
            session_id=session_id,
            latency_ms=latency_ms,
            usage={
                "input_tokens":  self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens":  self.usage.total_tokens,
                "retry_count":   self.usage.retry_count,
            },
            error=self.error,
        )
