"""RactoGateway Prebuilt Pipelines.

Prebuilt end-to-end pipelines for real-world LLM workflows.
Each pipeline is a self-contained class with ``run()`` / ``arun()`` methods,
full observability hooks, per-step model control, and optional per-call overrides.

Available pipelines
-------------------
- :class:`SQLAnalystPipeline` / :class:`AsyncSQLAnalystPipeline` —
  NL → SQL → pandas → Markdown answer + optional Plotly chart.
  Requires: ``pip install ractogateway[pipelines-sql]``
  Charts:   ``pip install ractogateway[pipelines-sql-viz]``

- :class:`ListClassifierPipeline` / :class:`AsyncListClassifierPipeline` —
  NL query → best-matching item(s) from a ``list[str]``.
  Uses dynamic ``Enum`` + Pydantic validation; supports single/multi selection,
  confidence scores, reasoning, retries, memory, rate limiting, and telemetry.
  No extra dependencies — works with any installed provider kit.

Usage::

    from ractogateway.pipelines import SQLAnalystPipeline, ListClassifierPipeline
    from ractogateway.openai_developer_kit import Chat

    # SQL Analyst
    pipeline = SQLAnalystPipeline(kit=Chat(model="gpt-4o"))
    result = pipeline.run(
        user_query="Top 5 products by quantity sold?",
        connection_string="postgresql://user:pass@localhost/db",
    )
    print(result.answer)

    # List Classifier
    classifier = ListClassifierPipeline(
        kit=Chat(model="gpt-4o-mini"),
        options=["Billing", "Technical Support", "Sales", "Account Management"],
        selection_mode="single",
        include_confidence=True,
        include_reasoning=True,
    )
    result = classifier.run("I can't log into my account")
    print(result.first)           # "Account Management"
    print(result.top_confidence)  # 0.94
    print(result.as_dict())       # {"selected": [...], "confidences": [...], ...}
"""

from ractogateway.pipelines.list_classifier import (
    AsyncListClassifierPipeline,
    AuditEntry,
    ClassifierRateLimitExceededError,
    ClassifierResult,
    ClassifierUsage,
    ListClassifierPipeline,
)
from ractogateway.pipelines.sql_analyst import (
    AsyncSQLAnalystPipeline,
    ChartSpec,
    PipelineUsage,
    RateLimitExceededError,
    ReadOnlySQLGuard,
    ReadOnlyViolationError,
    SQLAnalystPipeline,
    SQLAnalystResult,
    clear_schema_cache,
)

__all__ = [
    # SQL Analyst
    "AsyncSQLAnalystPipeline",
    "ChartSpec",
    "PipelineUsage",
    "RateLimitExceededError",
    "ReadOnlySQLGuard",
    "ReadOnlyViolationError",
    "SQLAnalystPipeline",
    "SQLAnalystResult",
    "clear_schema_cache",
    # List Classifier
    "AsyncListClassifierPipeline",
    "AuditEntry",
    "ClassifierRateLimitExceededError",
    "ClassifierResult",
    "ClassifierUsage",
    "ListClassifierPipeline",
]
