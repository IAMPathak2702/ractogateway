"""Data models for SQLAnalystPipeline."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ractogateway.pipelines.sql_analyst._schema import (
    _rows_to_csv,
    _rows_to_csv_file,
)


class ReadOnlyViolationError(ValueError):
    """Raised when generated SQL contains a write operation in force_read_only mode."""


class RateLimitExceededError(RuntimeError):
    """Raised when the rate limiter denies a request for a given user."""


class PipelineUsage(BaseModel):
    """Aggregated token usage across all LLM calls in the pipeline.

    Tracks each step (SQL generation, pandas code generation, markdown answer
    generation) separately so you can see exactly where tokens are consumed.

    Properties
    ----------
    total_input_tokens:
        Sum of all prompt tokens across every LLM step.
    total_output_tokens:
        Sum of all completion tokens across every LLM step.
    total_tokens:
        Grand total of every token consumed by the pipeline.
    """

    sql_input_tokens: int = 0
    sql_output_tokens: int = 0
    pandas_input_tokens: int = 0
    pandas_output_tokens: int = 0
    answer_input_tokens: int = 0
    answer_output_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.sql_input_tokens + self.pandas_input_tokens + self.answer_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.sql_output_tokens + self.pandas_output_tokens + self.answer_output_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class SQLAnalystResult(BaseModel):
    """Result returned by :class:`~ractogateway.pipelines.SQLAnalystPipeline`.

    All fields except ``user_query`` have sensible defaults so that a partial
    result can be returned when ``safe_mode=True`` and an error occurs.

    Fields
    ------
    user_query:
        The original natural-language question.
    schema_used:
        The database schema string that was passed to (or fetched for) the LLM.
    sql_query:
        The generated (and possibly LIMIT-injected) SQL SELECT statement.
    columns:
        Column names returned by the SQL query.
    row_count:
        Number of rows in ``raw_rows``.
    raw_rows:
        All rows from the SQL result as a list of dicts.
    pandas_code:
        The LLM-generated pandas analysis code (``None`` if ``run_pandas=False``).
    pandas_result:
        Output of executing ``pandas_code`` — DataFrame, scalar, or any value
        assigned to ``result`` inside the code.  ``None`` if ``run_pandas=False``.
    answer:
        Rich Markdown answer written by the LLM, including a results table and
        key insights.  ``None`` if ``run_answer=False``.
    chart_spec:
        The :class:`~ractogateway.pipelines.ChartSpec` dict used to build the
        Plotly figure.  ``None`` if no chart was requested.
    plotly_figure:
        A ``plotly.graph_objects.Figure`` object ready to call ``.show()`` or
        ``.to_html()``.  ``None`` if no chart was requested or plotly is not
        installed.
    usage:
        Aggregated token counts for all LLM steps in the pipeline.
    error:
        Set when ``safe_mode=True`` and an exception occurs.  ``None`` means
        the pipeline completed successfully.
    """

    user_query: str
    schema_used: str = ""
    sql_query: str = ""
    columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    raw_rows: list[dict[str, Any]] = Field(default_factory=list)
    pandas_code: str | None = None
    pandas_result: Any | None = None
    answer: str | None = None
    chart_spec: dict[str, Any] | None = None
    plotly_figure: Any | None = None
    usage: PipelineUsage = Field(default_factory=PipelineUsage)
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_csv(self, path: str | None = None) -> str | None:
        """Export the raw SQL result rows to CSV.

        Does **not** require pandas — uses the standard-library ``csv`` module.

        Parameters
        ----------
        path:
            File path to write to.  When ``None`` the CSV string is returned.

        Returns
        -------
        str | None
            CSV string when *path* is ``None``; otherwise ``None`` (file written).
        """
        if not self.raw_rows:
            return "" if path is None else None
        if path is None:
            return _rows_to_csv(self.raw_rows, self.columns)
        _rows_to_csv_file(self.raw_rows, self.columns, path)
        return None

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str | None:
        """Export the raw SQL result rows to JSON.

        Parameters
        ----------
        path:
            File path to write to.  When ``None`` the JSON string is returned.
        indent:
            JSON indentation level (default: ``2``).

        Returns
        -------
        str | None
            JSON string when *path* is ``None``; otherwise ``None`` (file written).
        """
        data = json.dumps(self.raw_rows, default=str, indent=indent)
        if path is None:
            return data
        with open(path, "w", encoding="utf-8") as f:  # noqa: PTH123
            f.write(data)
        return None

    def to_excel(self, path: str, *, sheet_name: str = "Results") -> None:
        """Export the raw SQL result rows to an Excel file.

        Requires ``pandas`` and ``openpyxl``::

            pip install ractogateway[pipelines-sql] openpyxl

        Parameters
        ----------
        path:
            File path to write to (must end in ``.xlsx``).
        sheet_name:
            Excel sheet name (default: ``"Results"``).
        """
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_excel(). "
                "Install it with:  pip install ractogateway[pipelines-sql]"
            ) from exc
        pd.DataFrame(self.raw_rows, columns=self.columns or None).to_excel(
            path, index=False, sheet_name=sheet_name
        )
