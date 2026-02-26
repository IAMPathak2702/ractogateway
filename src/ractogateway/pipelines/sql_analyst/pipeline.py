"""SQLAnalystPipeline — natural-language → SQL → pandas → answer + chart.

Two classes are exported:

- :class:`SQLAnalystPipeline` — ``run()`` is **sync**, ``arun()`` is **async**.
- :class:`AsyncSQLAnalystPipeline` — ``run()`` is **async** (no sync variant).
  Designed for FastAPI / async frameworks where you just ``await pipeline.run(...)``.

Per-step model control
----------------------
You can assign a different LLM kit (provider + model) to each pipeline step::

    pipeline = SQLAnalystPipeline(
        kit=Chat(model="gpt-4o"),                       # default fallback
        sql_kit=Chat(model="gpt-4o"),                   # SQL generation
        pandas_kit=Chat(model="gpt-3.5-turbo"),         # cheaper for pandas
        answer_kit=Chat(model="gpt-4o"),                # rich markdown answer
    )

When a per-step kit is ``None`` the default ``kit`` is used for that step.

Production features
-------------------
- **SQL retry loop** — on DB execution error the LLM is re-prompted with the
  error message to auto-fix the SQL (up to ``max_sql_retries`` times).
- **Pre-built engine** — pass a pre-configured SQLAlchemy ``Engine`` (with
  connection pooling) directly via the ``engine`` parameter.
- **Row limit cap** — ``max_rows`` auto-injects ``LIMIT`` to prevent unbounded
  result sets from overwhelming memory.
- **Schema cache** — schema introspection results are cached in-process for
  ``schema_cache_ttl`` seconds to avoid repeated DB round-trips.
- **Table RBAC** — ``allowed_tables`` hides all other tables from the LLM.
- **Column blocking** — ``blocked_columns`` removes sensitive columns from the
  schema shown to the LLM.
- **Data masking** — ``mask_columns`` replaces values with ``"***MASKED***"``
  in result rows before passing them to the LLM or returning to the caller.
- **Custom docs** — ``table_docs`` / ``column_docs`` annotate the schema with
  business descriptions so the LLM generates more accurate SQL.
- **Conversation memory** — ``memory`` (any object with ``get_history`` /
  ``append`` methods, e.g. :class:`~ractogateway.redis.RedisChatMemory`) +
  ``session_id`` appends prior Q&A context to each SQL prompt.
- **Rate limiting** — ``rate_limiter`` (any object with ``check_and_consume``
  / ``get_remaining`` methods, e.g.
  :class:`~ractogateway.redis.RedisRateLimiter`) + ``user_id`` gates requests.
- **Safe mode** — ``safe_mode=True`` catches all exceptions and returns a
  :class:`~ractogateway.pipelines.SQLAnalystResult` with the ``error`` field
  set instead of raising.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from ractogateway._models.chat import ChatConfig
from ractogateway.pipelines.sql_analyst._guard import ReadOnlySQLGuard
from ractogateway.pipelines.sql_analyst._models import (
    PipelineUsage,
    RateLimitExceededError,
    SQLAnalystResult,
)
from ractogateway.pipelines.sql_analyst._schema import (
    SchemaFetcher,
    _build_connection_string,
    _execute_pandas,
    _execute_polars,
    _extract_code,
    _extract_sql,
    _filter_schema,
    _get_cached_schema,
    _inject_limit,
    _mask_rows,
    _put_cached_schema,
    _require_pandas,
    _require_polars,
    _require_sqlalchemy,
)
from ractogateway.pipelines.sql_analyst._viz import (
    ChartSpec,
    build_figure,
    infer_spec,
)
from ractogateway.prompts.engine import RactoPrompt

# ---------------------------------------------------------------------------
# Sentinel — distinguishes "not passed" from "explicitly None"
# ---------------------------------------------------------------------------

_UNSET: Any = object()

# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

_DEFAULT_SQL_PROMPT = RactoPrompt(
    role=(
        "You are a senior database engineer who writes precise, optimised SQL "
        "queries that do all heavy lifting inside the database."
    ),
    aim=(
        "Analyse the user's question and the provided database schema, then "
        "produce a single SQL SELECT query that DIRECTLY returns the exact data "
        "needed — with every filter, join, aggregation, and sort handled in SQL "
        "itself.  The result set must be small, focused, and immediately usable."
    ),
    constraints=[
        # ── Security ────────────────────────────────────────────────────
        "SECURITY: Only produce SELECT statements — never INSERT, UPDATE, DELETE, "
        "DROP, CREATE, ALTER, TRUNCATE, EXEC, MERGE, CALL, GRANT, REVOKE, "
        "REPLACE, or any other DDL/DML statement.",
        # ── Schema fidelity ─────────────────────────────────────────────
        "SCHEMA: Use column and table names EXACTLY as they appear in the schema — "
        "no invented names, no assumptions about columns that are not listed.",
        # ── Filtering ───────────────────────────────────────────────────
        "FILTERING: Apply ALL row filters as SQL WHERE conditions. "
        "Never return unfiltered rows expecting Python/pandas to filter them later. "
        "If the question says 'active customers', add WHERE status = 'active' in SQL. "
        "If it says 'last 30 days', add WHERE created_at >= NOW() - INTERVAL '30 days'.",
        # ── Aggregation ─────────────────────────────────────────────────
        "AGGREGATION: Perform all counting, summing, averaging, and other "
        "aggregations with SQL GROUP BY + aggregate functions (SUM, COUNT, AVG, "
        "MAX, MIN, etc.). Never return raw rows for aggregation in Python.",
        # ── Post-group filtering ─────────────────────────────────────────
        "HAVING: Use HAVING to filter on aggregated values "
        "(e.g. customers with more than 5 orders → HAVING COUNT(*) > 5).",
        # ── Joins ───────────────────────────────────────────────────────
        "JOINS: Use SQL JOINs to combine tables when the question spans multiple "
        "tables. Never return separate tables for Python to merge. "
        "Use the foreign key relationships shown in the schema as join hints.",
        # ── Ordering & ranking ───────────────────────────────────────────
        "ORDERING: Always add ORDER BY for 'top N', 'highest', 'lowest', 'most', "
        "'least', 'newest', 'oldest', or any ranking question. "
        "Combine ORDER BY with LIMIT to return ONLY the required rows.",
        # ── Column selection ─────────────────────────────────────────────
        "COLUMNS: Select ONLY the columns the question needs. "
        "Never use SELECT * — name each column explicitly.",
        # ── CTEs and subqueries ──────────────────────────────────────────
        "READABILITY: Use Common Table Expressions (CTEs with WITH …) or "
        "subqueries when the logic involves multiple steps — they make the query "
        "correct and easy to verify.",
        # ── NULL handling ────────────────────────────────────────────────
        "NULLS: Handle NULLs explicitly where relevant "
        "(e.g. use IS NOT NULL, COALESCE, or NULLIF as needed).",
        # ── Output format ────────────────────────────────────────────────
        "OUTPUT: Return the SQL query inside a ```sql code block and NOTHING else "
        "outside that block.",
        # ── Unsolvable questions ─────────────────────────────────────────
        "UNSOLVABLE: If the question cannot be answered with the available schema, "
        "return a SQL comment inside the code block explaining why "
        "(e.g. -- Cannot answer: no 'email' column exists in the schema).",
    ],
    tone="precise, methodical, and technically rigorous",
    output_format=(
        "A single SQL SELECT query wrapped in a ```sql code block. "
        "The query must apply all filters, aggregations, joins, and ordering "
        "at the SQL level so the result set is minimal and ready to use directly."
    ),
    anti_hallucination=True,
)

_DEFAULT_PANDAS_PROMPT = RactoPrompt(
    role="You are an expert Python data analyst.",
    aim=(
        "Write concise pandas code to answer the user's question using "
        "the provided DataFrame `df`."
    ),
    constraints=[
        "Use only `df` (the DataFrame) and `pd` (pandas) — no other imports.",
        "Assign the final answer to a variable named `result`.",
        "Return code inside a ```python code block and nothing else outside it.",
        "Keep the code concise and efficient.",
    ],
    tone="precise and technical",
    output_format=(
        "Python pandas code wrapped in a ```python code block. "
        "The final answer must be assigned to a variable called `result`."
    ),
    anti_hallucination=True,
)

_DEFAULT_ANSWER_PROMPT = RactoPrompt(
    role="You are a skilled data analyst and communicator.",
    aim=(
        "Write a clear, insightful Markdown answer to the user's question "
        "based on the provided query results."
    ),
    constraints=[
        "Format your response in Markdown — use headers, bullet points, and a results table.",
        "Always include a well-formatted Markdown table showing the key data.",
        "Highlight the top 3 most important insights from the data.",
        "Keep the answer focused — directly address the user's question.",
        "Do not mention SQL, DataFrames, or any technical implementation details.",
    ],
    tone="clear, insightful, and professional",
    output_format=(
        "Markdown with a ## Summary section, a Markdown results table, "
        "and a ## Key Insights bullet list."
    ),
    anti_hallucination=True,
)


_DEFAULT_POLARS_PROMPT = RactoPrompt(
    role="You are an expert Python data analyst specialising in Polars.",
    aim=(
        "Write concise Polars code to answer the user's question using "
        "the provided DataFrame `df`."
    ),
    constraints=[
        "Use only `df` (the Polars DataFrame) and `pl` (the polars module) — no other imports.",
        "Assign the final answer to a variable named `result`.",
        "Return code inside a ```python code block and nothing else outside it.",
        "Use Polars-native operations — do NOT use pandas syntax or methods.",
        "Keep the code concise and efficient.",
    ],
    tone="precise and technical",
    output_format=(
        "Polars code wrapped in a ```python code block. "
        "The final answer must be assigned to a variable called `result`."
    ),
    anti_hallucination=True,
)


# ---------------------------------------------------------------------------
# SQL message builder — chain-of-thought prompt that precedes the query
# ---------------------------------------------------------------------------


def _build_sql_message(
    user_query: str,
    schema_text: str,
    memory_ctx: str = "",
    prior_sql: str = "",
    error_msg: str = "",
) -> str:
    """Build the full user-turn message for SQL generation.

    Uses a structured chain-of-thought format so the LLM reasons through
    tables → filters → aggregations → ordering BEFORE writing any SQL.
    On retries, the previous SQL and its error are appended so the LLM
    can self-correct.

    Parameters
    ----------
    user_query:
        The plain-English question from the user.
    schema_text:
        The (possibly filtered/annotated) schema string.
    memory_ctx:
        Formatted prior conversation turns, or empty string.
    prior_sql:
        Non-empty only on retry — the SQL attempt that failed.
    error_msg:
        Non-empty only on retry — the database error message.
    """
    retry_block = ""
    if prior_sql and error_msg:
        retry_block = (
            f"\n\n---\n"
            f"**Previous SQL attempt (FAILED — do not reuse):**\n"
            f"```sql\n{prior_sql}\n```\n"
            f"**Database error:**\n{error_msg}\n\n"
            f"Identify exactly what caused the error, fix it, and produce a "
            f"corrected SQL query below.\n---"
        )

    return (
        f"{memory_ctx}"
        f"## Database Schema\n\n"
        f"{schema_text}\n\n"
        f"## User Question\n\n"
        f"{user_query}\n\n"
        f"## Reasoning Steps\n\n"
        f"Work through these steps before writing SQL:\n\n"
        f"1. **Relevant tables** — Which tables contain the data this question needs?\n"
        f"2. **Joins** — Which foreign-key relationships connect those tables? "
        f"Write explicit JOIN conditions (never cross-join).\n"
        f"3. **Row filters (WHERE)** — What exact conditions narrow the rows to "
        f"only what the question asks about? Apply ALL filters in SQL — "
        f"do NOT return unfiltered rows.\n"
        f"4. **Aggregation (GROUP BY)** — Does the question need counts, sums, "
        f"averages, or other aggregates? If yes, push them into SQL.\n"
        f"5. **Post-group filter (HAVING)** — Does the question restrict an "
        f"aggregated value (e.g. 'customers with > 5 orders')? Use HAVING.\n"
        f"6. **Ordering & ranking (ORDER BY + LIMIT)** — Does the question ask "
        f"for top/bottom N, most/least, newest/oldest? Add ORDER BY + LIMIT.\n"
        f"7. **Columns** — Select ONLY the columns the question needs. No SELECT *.\n\n"
        f"Now write the complete SQL SELECT query that applies ALL of the above "
        f"inside SQL itself, returning the minimal, focused result set."
        f"{retry_block}"
    )


# ---------------------------------------------------------------------------
# SQLAnalystPipeline
# ---------------------------------------------------------------------------


class SQLAnalystPipeline:
    """Natural-language to SQL + pandas + Markdown answer + chart pipeline.

    Converts a plain-English question into:

    1. A read-only SQL query (LLM step — *sql_kit*)
    2. Pandas analysis code executed against the SQL result (LLM step — *pandas_kit*)
    3. A rich Markdown answer with table + insights (LLM step — *answer_kit*)
    4. An optional Plotly figure built deterministically from a ``ChartSpec``
       (zero LLM calls — pure dtype heuristics or user-provided spec)

    Two variants
    ------------
    - :class:`SQLAnalystPipeline` — ``run()`` sync, ``arun()`` async.
    - :class:`AsyncSQLAnalystPipeline` — ``run()`` is async (same as ``arun()``).

    Parameters
    ----------
    kit:
        Default LLM kit used for any step that doesn't have its own kit.
    sql_kit:
        Override kit for SQL generation.  Falls back to *kit*.
    pandas_kit:
        Override kit for pandas code generation.  Falls back to *kit*.
    answer_kit:
        Override kit for Markdown answer generation.  Falls back to *kit*.
    sql_prompt / pandas_prompt / answer_prompt:
        Override default system prompts for each step.
    sql_temperature / sql_max_tokens:
        LLM settings for the SQL step (default: 0.0 / 1024).
    pandas_temperature / pandas_max_tokens:
        LLM settings for the pandas step (default: 0.0 / 2048).
    answer_temperature / answer_max_tokens:
        LLM settings for the answer step (default: 0.3 / 2048).
    run_pandas:
        Run pandas analysis step by default (default: ``True``).
    run_answer:
        Run Markdown answer step by default (default: ``True``).
    chart:
        Default chart behaviour: ``"auto"`` (infer from data), a
        :class:`~ractogateway.pipelines.ChartSpec`, a plain ``dict``, or
        ``None`` to skip charts.  Default: ``"auto"``.
    force_read_only:
        Block any non-SELECT SQL (default: ``True``).
    tracer:
        Optional :class:`~ractogateway.telemetry.RactoTracer` instance.
    metrics:
        Optional :class:`~ractogateway.telemetry.GatewayMetricsMiddleware` instance.
    engine:
        Optional pre-built SQLAlchemy ``Engine`` (e.g. with connection pooling).
        When provided, ``connection_string`` / ``host`` / ``port`` / etc. params
        in ``run()`` are ignored.
    max_sql_retries:
        Number of times to retry SQL generation when a DB execution error occurs.
        Each retry re-sends the LLM the original question plus the error message
        so it can self-correct.  Default: ``2``.
    max_rows:
        Safety cap on returned rows — auto-injects ``LIMIT {max_rows}`` into the
        SQL if no LIMIT is already present.  Set to ``0`` to disable.
        Default: ``10_000``.
    schema_cache_ttl:
        Seconds to cache the schema introspection result in-process.  Set to
        ``0`` to disable caching.  Default: ``3600`` (1 hour).
    allowed_tables:
        Allowlist of table names shown to the LLM.  All other tables are hidden,
        preventing the LLM from generating SQL that touches them.
    blocked_columns:
        Column names to strip from the schema shown to the LLM (case-insensitive).
        Useful for hiding PII columns like ``ssn`` or ``credit_card_number``.
    mask_columns:
        Column names whose *values* are replaced with ``"***MASKED***"`` in result
        rows before they are returned or passed to the answer LLM.
    table_docs:
        ``{table_name: description}`` — appended as inline schema comments so
        the LLM understands table business meaning.
    column_docs:
        ``{table_name: {column_name: description}}`` — per-column inline comments.
    safe_mode:
        When ``True``, all exceptions are caught and returned as
        ``SQLAnalystResult(error=...)`` instead of being raised.
        Default: ``False``.
    memory:
        Optional conversation memory object (e.g.
        :class:`~ractogateway.redis.RedisChatMemory`).  Must implement
        ``get_history(session_id) -> list[dict]`` and
        ``append(session_id, role, content)``.
    rate_limiter:
        Optional rate-limiter object (e.g.
        :class:`~ractogateway.redis.RedisRateLimiter`).  Must implement
        ``check_and_consume(user_id, tokens) -> bool`` and
        ``get_remaining(user_id) -> int``.
    user_id:
        Default user identifier used for rate limiting and audit.  Can be
        overridden per-call in ``run()`` / ``arun()``.

    Example::

        from ractogateway.openai_developer_kit import Chat
        from ractogateway.pipelines import SQLAnalystPipeline, ChartSpec

        pipeline = SQLAnalystPipeline(
            kit=Chat(model="gpt-4o"),
            pandas_kit=Chat(model="gpt-3.5-turbo"),  # cheaper for pandas
            max_rows=5_000,
            allowed_tables=["orders", "customers", "products"],
            mask_columns=["email", "phone"],
            safe_mode=True,
        )
        result = pipeline.run(
            user_query="Top 5 products by quantity sold?",
            connection_string="postgresql://user:pass@localhost/shop",
        )
        if result.error:
            print("Pipeline error:", result.error)
        else:
            print(result.answer)
            result.plotly_figure.show()
            print(result.usage.total_tokens)
            result.to_csv("output.csv")
    """

    def __init__(  # noqa: PLR0913
        self,
        kit: Any,
        *,
        # Per-step kit overrides
        sql_kit: Any | None = None,
        pandas_kit: Any | None = None,
        answer_kit: Any | None = None,
        # Prompts
        sql_prompt: RactoPrompt | None = None,
        pandas_prompt: RactoPrompt | None = None,
        answer_prompt: RactoPrompt | None = None,
        # SQL step settings
        sql_temperature: float = 0.0,
        sql_max_tokens: int = 1024,
        # Pandas step settings
        pandas_temperature: float = 0.0,
        pandas_max_tokens: int = 2048,
        # Answer step settings
        answer_temperature: float = 0.3,
        answer_max_tokens: int = 2048,
        # Step toggles
        run_pandas: bool = True,
        run_answer: bool = True,
        chart: ChartSpec | dict[str, Any] | str | None = "auto",
        # Security
        force_read_only: bool = True,
        # Observability
        tracer: Any | None = None,
        metrics: Any | None = None,
        # Production: connection
        engine: Any | None = None,
        # Production: reliability
        max_sql_retries: int = 2,
        max_rows: int = 10_000,
        # Production: schema management
        schema_cache_ttl: float = 3600.0,
        schema_include_indexes: bool = True,
        schema_include_row_counts: bool = False,
        schema_include_sample_values: bool = False,
        schema_sample_value_limit: int = 8,
        allowed_tables: list[str] | None = None,
        blocked_columns: list[str] | None = None,
        # Production: data access control
        mask_columns: list[str] | None = None,
        table_docs: dict[str, str] | None = None,
        column_docs: dict[str, dict[str, str]] | None = None,
        # Production: error handling
        safe_mode: bool = False,
        # Production: analysis engine  ("pandas" | "polars")
        analysis_engine: str = "pandas",
        # Production: conversation memory
        memory: Any | None = None,
        # Production: rate limiting
        rate_limiter: Any | None = None,
        user_id: str | None = None,
    ) -> None:
        self._kit = kit
        self._sql_kit = sql_kit
        self._pandas_kit = pandas_kit
        self._answer_kit = answer_kit
        self._sql_prompt = sql_prompt or _DEFAULT_SQL_PROMPT
        self._pandas_prompt = pandas_prompt or _DEFAULT_PANDAS_PROMPT
        self._answer_prompt = answer_prompt or _DEFAULT_ANSWER_PROMPT
        self._sql_temperature = sql_temperature
        self._sql_max_tokens = sql_max_tokens
        self._pandas_temperature = pandas_temperature
        self._pandas_max_tokens = pandas_max_tokens
        self._answer_temperature = answer_temperature
        self._answer_max_tokens = answer_max_tokens
        self._run_pandas = run_pandas
        self._run_answer = run_answer
        self._chart = chart
        self._force_read_only = force_read_only
        self._tracer = tracer
        self._metrics = metrics
        # Production
        self._engine = engine
        self._max_sql_retries = max_sql_retries
        self._max_rows = max_rows
        self._schema_cache_ttl = schema_cache_ttl
        self._schema_include_indexes = schema_include_indexes
        self._schema_include_row_counts = schema_include_row_counts
        self._schema_include_sample_values = schema_include_sample_values
        self._schema_sample_value_limit = schema_sample_value_limit
        self._allowed_tables = allowed_tables
        self._blocked_columns = blocked_columns
        self._mask_columns = mask_columns
        self._table_docs = table_docs
        self._column_docs = column_docs
        self._safe_mode = safe_mode
        _valid_engines = {"pandas", "polars"}
        if analysis_engine not in _valid_engines:
            raise ValueError(
                f"analysis_engine must be one of {sorted(_valid_engines)}, "
                f"got {analysis_engine!r}"
            )
        self._analysis_engine = analysis_engine
        self._memory = memory
        self._rate_limiter = rate_limiter
        self._user_id = user_id

    # ------------------------------------------------------------------
    # Public sync API
    # ------------------------------------------------------------------

    def run(  # noqa: PLR0913
        self,
        user_query: str,
        *,
        # Connection
        connection_string: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        driver: str = "postgresql",
        # Pre-built engine override (per-call)
        engine: Any = _UNSET,
        # Schema
        schema: str | None = None,
        # Per-call step overrides (None = use pipeline default)
        run_pandas: bool | None = None,
        run_answer: bool | None = None,
        chart: Any = _UNSET,
        force_read_only: bool | None = None,
        # Per-call LLM setting overrides
        sql_temperature: float | None = None,
        sql_max_tokens: int | None = None,
        pandas_temperature: float | None = None,
        pandas_max_tokens: int | None = None,
        answer_temperature: float | None = None,
        answer_max_tokens: int | None = None,
        # Per-call production overrides
        max_rows: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> SQLAnalystResult:
        """Run the full pipeline synchronously.

        Parameters
        ----------
        user_query:
            Plain-English question to answer from the database.
        connection_string:
            Full SQLAlchemy URI.  Ignored when ``engine`` is provided.
        host / port / database / username / password / driver:
            Individual connection params used when both *connection_string* and
            *engine* are omitted.
        engine:
            Per-call pre-built SQLAlchemy ``Engine``.  Overrides the
            pipeline-level ``engine`` and all connection params.
        schema:
            Pre-computed schema string.  ``None`` → fetched automatically
            (with optional cache).
        run_pandas / run_answer / chart / force_read_only:
            Override the corresponding pipeline-level defaults for this call.
        sql_temperature / sql_max_tokens / pandas_temperature / pandas_max_tokens /
        answer_temperature / answer_max_tokens:
            Per-call LLM setting overrides.
        max_rows:
            Per-call row limit override.  Overrides the pipeline-level ``max_rows``.
        user_id:
            Per-call user ID for rate limiting and audit.
        session_id:
            Conversation session ID used to fetch and save memory context.

        Returns
        -------
        SQLAnalystResult
        """
        cfg = self._resolve_config(
            connection_string=connection_string,
            engine_override=engine,
            host=host, port=port, database=database,
            username=username, password=password, driver=driver,
            run_pandas=run_pandas, run_answer=run_answer, chart=chart,
            force_read_only=force_read_only,
            sql_temp=sql_temperature, sql_tok=sql_max_tokens,
            pd_temp=pandas_temperature, pd_tok=pandas_max_tokens,
            ans_temp=answer_temperature, ans_tok=answer_max_tokens,
            max_rows=max_rows, user_id=user_id, session_id=session_id,
        )
        if self._safe_mode:
            try:
                return self._execute_pipeline(user_query, schema, cfg)
            except Exception as exc:  # noqa: BLE001
                return SQLAnalystResult(user_query=user_query, error=str(exc))
        return self._execute_pipeline(user_query, schema, cfg)

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def arun(  # noqa: PLR0913
        self,
        user_query: str,
        *,
        connection_string: str | None = None,
        host: str = "localhost",
        port: int = 5432,
        database: str | None = None,
        username: str | None = None,
        password: str | None = None,
        driver: str = "postgresql",
        engine: Any = _UNSET,
        schema: str | None = None,
        run_pandas: bool | None = None,
        run_answer: bool | None = None,
        chart: Any = _UNSET,
        force_read_only: bool | None = None,
        sql_temperature: float | None = None,
        sql_max_tokens: int | None = None,
        pandas_temperature: float | None = None,
        pandas_max_tokens: int | None = None,
        answer_temperature: float | None = None,
        answer_max_tokens: int | None = None,
        max_rows: int | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> SQLAnalystResult:
        """Async variant of :meth:`run` — identical parameters.

        Blocking SQLAlchemy calls run in a thread executor.
        LLM calls use each kit's async ``achat()`` method.
        """
        cfg = self._resolve_config(
            connection_string=connection_string,
            engine_override=engine,
            host=host, port=port, database=database,
            username=username, password=password, driver=driver,
            run_pandas=run_pandas, run_answer=run_answer, chart=chart,
            force_read_only=force_read_only,
            sql_temp=sql_temperature, sql_tok=sql_max_tokens,
            pd_temp=pandas_temperature, pd_tok=pandas_max_tokens,
            ans_temp=answer_temperature, ans_tok=answer_max_tokens,
            max_rows=max_rows, user_id=user_id, session_id=session_id,
        )
        if self._safe_mode:
            try:
                return await self._aexecute_pipeline(user_query, schema, cfg)
            except Exception as exc:  # noqa: BLE001
                return SQLAnalystResult(user_query=user_query, error=str(exc))
        return await self._aexecute_pipeline(user_query, schema, cfg)

    # ------------------------------------------------------------------
    # Core sync pipeline logic
    # ------------------------------------------------------------------

    def _execute_pipeline(  # noqa: PLR0912, PLR0915
        self, user_query: str, schema_override: str | None, cfg: dict[str, Any]
    ) -> SQLAnalystResult:
        sa = _require_sqlalchemy()
        use_polars = cfg["analysis_engine"] == "polars"
        pd_module: Any = None
        pl_module: Any = None

        if use_polars:
            pl_module = _require_polars()
        else:
            pd_module = _require_pandas()

        # ── Rate limiting ──────────────────────────────────────────────
        self._check_rate_limit(cfg["user_id"])

        # ── Build / resolve engine ─────────────────────────────────────
        db_engine = cfg["engine"] or sa.create_engine(cfg["conn_str"])

        # ── Schema (cache → fetch → filter) ───────────────────────────
        schema_text = self._get_schema(
            schema_override, db_engine, cfg["conn_str"], cfg["schema_cache_ttl"]
        )

        # ── Memory context ─────────────────────────────────────────────
        memory_ctx = self._get_memory_context(cfg["session_id"])

        usage = PipelineUsage()

        # ── Step 1: SQL generation with retry loop ─────────────────────
        sql_query, columns, rows = self._sql_with_retry(
            user_query, schema_text, memory_ctx, db_engine, sa, cfg, usage
        )

        # ── Masking ────────────────────────────────────────────────────
        rows = _mask_rows(rows, self._mask_columns)

        # ── Build DataFrame (pandas or Polars) ─────────────────────────
        if use_polars:
            df = pl_module.DataFrame(rows)
        else:
            df = pd_module.DataFrame(rows, columns=columns)  # type: ignore[union-attr]

        # ── Step 2: Analysis (pandas or Polars) ───────────────────────
        pandas_code: str | None = None
        pandas_result: Any | None = None

        if cfg["run_pandas"]:
            if use_polars:
                pandas_code, pd_usg = self._generate_analysis_code(
                    user_query, columns, df, cfg["pd_temp"], cfg["pd_tok"],
                    engine="polars",
                )
                usage.pandas_input_tokens = pd_usg.get("prompt_tokens", 0)
                usage.pandas_output_tokens = pd_usg.get("completion_tokens", 0)
                pandas_result = _execute_polars(pandas_code, df, pl_module)
            else:
                pandas_code, pd_usg = self._generate_analysis_code(
                    user_query, columns, df, cfg["pd_temp"], cfg["pd_tok"],
                    engine="pandas",
                )
                usage.pandas_input_tokens = pd_usg.get("prompt_tokens", 0)
                usage.pandas_output_tokens = pd_usg.get("completion_tokens", 0)
                pandas_result = _execute_pandas(pandas_code, df, pd_module)

        # ── Step 3: Markdown answer ────────────────────────────────────
        answer: str | None = None

        if cfg["run_answer"]:
            result_str = self._result_to_str_any(pandas_result, df, use_polars, pd_module)
            answer, ans_usg = self._generate_answer(
                user_query, columns, result_str, cfg["ans_temp"], cfg["ans_tok"]
            )
            usage.answer_input_tokens = ans_usg.get("prompt_tokens", 0)
            usage.answer_output_tokens = ans_usg.get("completion_tokens", 0)

        # ── Step 4: Chart (zero LLM calls) ────────────────────────────
        # Plotly Express works best with pandas — convert Polars if needed
        if use_polars:
            pd_module = self._try_get_pandas()
            chart_spec_dict, plotly_figure = self._build_chart_any(
                cfg["chart"], pandas_result, df, pd_module, use_polars
            )
        else:
            chart_spec_dict, plotly_figure = self._build_chart(
                cfg["chart"], pandas_result, df, pd_module
            )

        # ── Save memory + emit metrics ─────────────────────────────────
        self._save_memory(cfg["session_id"], user_query, answer)
        self._emit_metrics(usage)

        return SQLAnalystResult(
            user_query=user_query,
            schema_used=schema_text,
            sql_query=sql_query,
            columns=columns,
            row_count=len(rows),
            raw_rows=rows,
            pandas_code=pandas_code,
            pandas_result=pandas_result,
            answer=answer,
            chart_spec=chart_spec_dict,
            plotly_figure=plotly_figure,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # Core async pipeline logic
    # ------------------------------------------------------------------

    async def _aexecute_pipeline(  # noqa: PLR0912, PLR0915
        self, user_query: str, schema_override: str | None, cfg: dict[str, Any]
    ) -> SQLAnalystResult:
        sa = _require_sqlalchemy()
        use_polars = cfg["analysis_engine"] == "polars"
        pd_module: Any = None
        pl_module: Any = None

        if use_polars:
            pl_module = _require_polars()
        else:
            pd_module = _require_pandas()

        loop = asyncio.get_event_loop()

        # ── Rate limiting ──────────────────────────────────────────────
        self._check_rate_limit(cfg["user_id"])

        # ── Build / resolve engine ─────────────────────────────────────
        if cfg["engine"] is not None:
            db_engine = cfg["engine"]
        else:
            db_engine = await loop.run_in_executor(
                None, lambda: sa.create_engine(cfg["conn_str"])
            )

        # ── Schema (cache → fetch → filter) ───────────────────────────
        schema_text = await loop.run_in_executor(
            None,
            lambda: self._get_schema(
                schema_override, db_engine, cfg["conn_str"], cfg["schema_cache_ttl"]
            ),
        )

        # ── Memory context ─────────────────────────────────────────────
        memory_ctx = self._get_memory_context(cfg["session_id"])

        usage = PipelineUsage()

        # ── Step 1: SQL generation with retry loop (async) ────────────
        sql_query, columns, rows = await self._asql_with_retry(
            user_query, schema_text, memory_ctx, db_engine, sa, cfg, usage, loop
        )

        # ── Masking ────────────────────────────────────────────────────
        rows = _mask_rows(rows, self._mask_columns)

        # ── Build DataFrame (pandas or Polars) ─────────────────────────
        if use_polars:
            df = pl_module.DataFrame(rows)
        else:
            df = pd_module.DataFrame(rows, columns=columns)  # type: ignore[union-attr]

        # ── Step 2: Analysis (async LLM) ──────────────────────────────
        pandas_code: str | None = None
        pandas_result: Any | None = None

        if cfg["run_pandas"]:
            if use_polars:
                pandas_code, pd_usg = await self._agenerate_analysis_code(
                    user_query, columns, df, cfg["pd_temp"], cfg["pd_tok"],
                    engine="polars",
                )
                usage.pandas_input_tokens = pd_usg.get("prompt_tokens", 0)
                usage.pandas_output_tokens = pd_usg.get("completion_tokens", 0)
                pandas_result = await loop.run_in_executor(
                    None,
                    lambda: _execute_polars(pandas_code, df, pl_module),  # type: ignore[arg-type]
                )
            else:
                pandas_code, pd_usg = await self._agenerate_analysis_code(
                    user_query, columns, df, cfg["pd_temp"], cfg["pd_tok"],
                    engine="pandas",
                )
                usage.pandas_input_tokens = pd_usg.get("prompt_tokens", 0)
                usage.pandas_output_tokens = pd_usg.get("completion_tokens", 0)
                pandas_result = await loop.run_in_executor(
                    None,
                    lambda: _execute_pandas(pandas_code, df, pd_module),  # type: ignore[arg-type]
                )

        # ── Step 3: Markdown answer (async LLM) ───────────────────────
        answer: str | None = None

        if cfg["run_answer"]:
            result_str = self._result_to_str_any(pandas_result, df, use_polars, pd_module)
            answer, ans_usg = await self._agenerate_answer(
                user_query, columns, result_str, cfg["ans_temp"], cfg["ans_tok"]
            )
            usage.answer_input_tokens = ans_usg.get("prompt_tokens", 0)
            usage.answer_output_tokens = ans_usg.get("completion_tokens", 0)

        # ── Step 4: Chart in thread (Plotly is blocking) ──────────────
        if use_polars:
            pd_mod_for_chart = self._try_get_pandas()
            chart_spec_dict, plotly_figure = await loop.run_in_executor(
                None,
                lambda: self._build_chart_any(
                    cfg["chart"], pandas_result, df, pd_mod_for_chart, use_polars
                ),
            )
        else:
            chart_spec_dict, plotly_figure = await loop.run_in_executor(
                None,
                lambda: self._build_chart(cfg["chart"], pandas_result, df, pd_module),
            )

        # ── Save memory + emit metrics ─────────────────────────────────
        self._save_memory(cfg["session_id"], user_query, answer)
        self._emit_metrics(usage)

        return SQLAnalystResult(
            user_query=user_query,
            schema_used=schema_text,
            sql_query=sql_query,
            columns=columns,
            row_count=len(rows),
            raw_rows=rows,
            pandas_code=pandas_code,
            pandas_result=pandas_result,
            answer=answer,
            chart_spec=chart_spec_dict,
            plotly_figure=plotly_figure,
            usage=usage,
        )

    # ------------------------------------------------------------------
    # SQL retry helpers (sync + async)
    # ------------------------------------------------------------------

    def _sql_with_retry(
        self,
        user_query: str,
        schema_text: str,
        memory_ctx: str,
        db_engine: Any,
        sa: Any,
        cfg: dict[str, Any],
        usage: PipelineUsage,
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        """Generate SQL and execute it, retrying on DB error up to max_sql_retries."""
        last_error: str | None = None
        sql: str = ""

        for attempt in range(self._max_sql_retries + 1):
            # On retries, the error + prior SQL are embedded inside the message
            sql, sql_usg = self._generate_sql(
                user_query, schema_text, memory_ctx, cfg["sql_temp"], cfg["sql_tok"],
                prior_sql=sql if attempt > 0 else "",
                error_msg=last_error or "",
            )
            # Accumulate usage across retries
            usage.sql_input_tokens += sql_usg.get("prompt_tokens", 0)
            usage.sql_output_tokens += sql_usg.get("completion_tokens", 0)

            if cfg["force_ro"]:
                ReadOnlySQLGuard.check(sql)

            final_sql = _inject_limit(sql, cfg["max_rows"])

            try:
                with db_engine.connect() as conn:
                    rp = conn.execute(sa.text(final_sql))
                    columns = list(rp.keys())
                    rows = [dict(zip(columns, row)) for row in rp.fetchall()]
                return final_sql, columns, rows
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self._max_sql_retries:
                    raise

        raise RuntimeError("unreachable")  # pragma: no cover

    async def _asql_with_retry(
        self,
        user_query: str,
        schema_text: str,
        memory_ctx: str,
        db_engine: Any,
        sa: Any,
        cfg: dict[str, Any],
        usage: PipelineUsage,
        loop: Any,
    ) -> tuple[str, list[str], list[dict[str, Any]]]:
        """Async version of :meth:`_sql_with_retry`."""
        last_error: str | None = None
        sql: str = ""

        for attempt in range(self._max_sql_retries + 1):
            sql, sql_usg = await self._agenerate_sql(
                user_query, schema_text, memory_ctx, cfg["sql_temp"], cfg["sql_tok"],
                prior_sql=sql if attempt > 0 else "",
                error_msg=last_error or "",
            )
            usage.sql_input_tokens += sql_usg.get("prompt_tokens", 0)
            usage.sql_output_tokens += sql_usg.get("completion_tokens", 0)

            if cfg["force_ro"]:
                ReadOnlySQLGuard.check(sql)

            final_sql = _inject_limit(sql, cfg["max_rows"])

            try:
                def _exec(engine: Any = db_engine, fsql: str = final_sql) -> (
                    tuple[list[str], list[dict[str, Any]]]
                ):
                    with engine.connect() as conn:
                        rp = conn.execute(sa.text(fsql))
                        cols = list(rp.keys())
                        rws = [dict(zip(cols, row)) for row in rp.fetchall()]
                    return cols, rws

                columns, rows = await loop.run_in_executor(None, _exec)
                return final_sql, columns, rows
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self._max_sql_retries:
                    raise

        raise RuntimeError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Internal: config resolution
    # ------------------------------------------------------------------

    def _resolve_config(  # noqa: PLR0913
        self,
        *,
        connection_string: str | None,
        engine_override: Any,
        host: str, port: int, database: str | None,
        username: str | None, password: str | None, driver: str,
        run_pandas: bool | None, run_answer: bool | None,
        chart: Any, force_read_only: bool | None,
        sql_temp: float | None, sql_tok: int | None,
        pd_temp: float | None, pd_tok: int | None,
        ans_temp: float | None, ans_tok: int | None,
        max_rows: int | None, user_id: str | None, session_id: str | None,
    ) -> dict[str, Any]:
        """Merge pipeline defaults with per-call overrides into a flat config dict."""
        # Resolve engine: per-call → pipeline-level → None (build from conn_str)
        resolved_engine: Any | None = None
        if engine_override is not _UNSET:
            resolved_engine = engine_override
        elif self._engine is not None:
            resolved_engine = self._engine

        # Build conn_str only when no engine is available
        conn_str = ""
        if resolved_engine is None:
            conn_str = connection_string or self._resolve_connection_string(
                host, port, database, username, password, driver
            )

        return {
            "conn_str":         conn_str,
            "engine":           resolved_engine,
            "run_pandas":       run_pandas       if run_pandas       is not None else self._run_pandas,
            "run_answer":       run_answer       if run_answer       is not None else self._run_answer,
            "chart":            chart            if chart            is not _UNSET else self._chart,
            "force_ro":         force_read_only  if force_read_only  is not None else self._force_read_only,
            "sql_temp":         sql_temp         if sql_temp         is not None else self._sql_temperature,
            "sql_tok":          sql_tok          if sql_tok          is not None else self._sql_max_tokens,
            "pd_temp":          pd_temp          if pd_temp          is not None else self._pandas_temperature,
            "pd_tok":           pd_tok           if pd_tok           is not None else self._pandas_max_tokens,
            "ans_temp":         ans_temp         if ans_temp         is not None else self._answer_temperature,
            "ans_tok":          ans_tok          if ans_tok          is not None else self._answer_max_tokens,
            "max_rows":         max_rows         if max_rows         is not None else self._max_rows,
            "schema_cache_ttl": self._schema_cache_ttl,
            "analysis_engine":  self._analysis_engine,
            "user_id":          user_id          if user_id          is not None else self._user_id,
            "session_id":       session_id,
        }

    # ------------------------------------------------------------------
    # Internal: kit selector
    # ------------------------------------------------------------------

    def _get_kit(self, step: str) -> Any:
        """Return the kit for *step*, falling back to the default kit."""
        override = {
            "sql":    self._sql_kit,
            "pandas": self._pandas_kit,
            "answer": self._answer_kit,
        }.get(step)
        return override if override is not None else self._kit

    # ------------------------------------------------------------------
    # Internal: schema management
    # ------------------------------------------------------------------

    def _get_schema(
        self,
        schema_override: str | None,
        db_engine: Any,
        conn_str: str,
        cache_ttl: float,
    ) -> str:
        """Fetch schema with cache support, then apply RBAC + doc filtering."""
        if schema_override:
            schema_blocked = list(
                set(self._blocked_columns or []) | set(self._mask_columns or [])
            )
            return _filter_schema(
                schema_override,
                self._allowed_tables, schema_blocked or None,
                self._table_docs, self._column_docs,
            )

        # Try cache (keyed by conn_str; skip if engine was passed directly)
        if conn_str:
            cached = _get_cached_schema(conn_str, cache_ttl)
            if cached is not None:
                return cached

        raw_schema = SchemaFetcher.fetch(
            db_engine,
            include_indexes=self._schema_include_indexes,
            include_row_counts=self._schema_include_row_counts,
            include_sample_values=self._schema_include_sample_values,
            sample_value_limit=self._schema_sample_value_limit,
        )
        # mask_columns values should also be hidden from the schema shown to the LLM
        schema_blocked = list(
            set(self._blocked_columns or []) | set(self._mask_columns or [])
        )
        filtered = _filter_schema(
            raw_schema,
            self._allowed_tables, schema_blocked or None,
            self._table_docs, self._column_docs,
        )

        if conn_str:
            _put_cached_schema(conn_str, filtered)

        return filtered

    # ------------------------------------------------------------------
    # Internal: rate limiting
    # ------------------------------------------------------------------

    def _check_rate_limit(self, user_id: str | None) -> None:
        """Raise :exc:`RateLimitExceededError` if the rate limiter denies the request."""
        if self._rate_limiter is None or not user_id:
            return
        try:
            allowed: bool = self._rate_limiter.check_and_consume(user_id, 1000)
        except Exception as exc:
            raise RateLimitExceededError(
                f"Rate limiter error for user '{user_id}': {exc}"
            ) from exc
        if not allowed:
            try:
                remaining = self._rate_limiter.get_remaining(user_id)
            except Exception:  # noqa: BLE001
                remaining = 0
            raise RateLimitExceededError(
                f"Rate limit exceeded for user '{user_id}'. "
                f"Remaining quota: {remaining} tokens."
            )

    # ------------------------------------------------------------------
    # Internal: memory helpers
    # ------------------------------------------------------------------

    def _get_memory_context(self, session_id: str | None) -> str:
        """Return a formatted string of prior conversation turns, or empty string."""
        if self._memory is None or not session_id:
            return ""
        try:
            history: list[dict[str, Any]] = self._memory.get_history(session_id)
        except Exception:  # noqa: BLE001
            return ""
        if not history:
            return ""
        parts = ["Prior conversation context (use for follow-up questions):"]
        for turn in history[-6:]:  # last 3 Q&A pairs maximum
            role = str(turn.get("role", "unknown")).upper()
            content = str(turn.get("content", ""))[:500]
            parts.append(f"  {role}: {content}")
        return "\n".join(parts) + "\n\n"

    def _save_memory(
        self, session_id: str | None, user_query: str, answer: str | None
    ) -> None:
        """Append this turn to the conversation memory (non-fatal on error)."""
        if self._memory is None or not session_id:
            return
        try:
            self._memory.append(session_id, "user", user_query)
            if answer:
                self._memory.append(session_id, "assistant", answer[:1000])
        except Exception:  # noqa: BLE001
            pass  # Memory persistence errors are non-fatal

    # ------------------------------------------------------------------
    # Internal: sync step helpers
    # ------------------------------------------------------------------

    def _generate_sql(
        self,
        user_query: str,
        schema_text: str,
        memory_ctx: str,
        temp: float,
        max_tok: int,
        *,
        prior_sql: str = "",
        error_msg: str = "",
    ) -> tuple[str, dict[str, int]]:
        start = time.perf_counter()
        config = ChatConfig(
            user_message=_build_sql_message(
                user_query, schema_text, memory_ctx, prior_sql, error_msg
            ),
            prompt=self._sql_prompt,
            temperature=temp,
            max_tokens=max_tok,
        )
        response = self._get_kit("sql").chat(config)
        latency_ms = (time.perf_counter() - start) * 1000
        usage: dict[str, int] = response.usage or {}
        sql = _extract_sql(response.content or "")
        self._record_span("sql_generation", latency_ms, usage)
        return sql, usage

    def _generate_analysis_code(
        self,
        user_query: str,
        columns: list[str],
        df: Any,
        temp: float,
        max_tok: int,
        engine: str = "pandas",
    ) -> tuple[str, dict[str, int]]:
        """Generate analysis code for either pandas or Polars engine."""
        start = time.perf_counter()
        # Build a concise sample regardless of DataFrame type
        try:
            sample = df.head(5).to_string(index=False)  # pandas
        except AttributeError:
            sample = str(df.head(5))  # polars

        lib_name = "polars (`pl`)" if engine == "polars" else "pandas (`pd`)"
        prompt = (
            _DEFAULT_POLARS_PROMPT if engine == "polars" else self._pandas_prompt
        )
        config = ChatConfig(
            user_message=(
                f"DataFrame columns: {columns}\n"
                f"First rows sample:\n{sample}\n\n"
                f"User question: {user_query}\n\n"
                f"Write {lib_name} code to answer the question. "
                "Assign the final result to a variable named `result`."
            ),
            prompt=prompt,
            temperature=temp,
            max_tokens=max_tok,
        )
        response = self._get_kit("pandas").chat(config)
        latency_ms = (time.perf_counter() - start) * 1000
        usage: dict[str, int] = response.usage or {}
        code = _extract_code(response.content or "")
        self._record_span("pandas_generation", latency_ms, usage)
        return code, usage

    # kept as an alias for backward-compat
    def _generate_pandas_code(
        self,
        user_query: str,
        columns: list[str],
        df: Any,
        temp: float,
        max_tok: int,
    ) -> tuple[str, dict[str, int]]:
        return self._generate_analysis_code(user_query, columns, df, temp, max_tok, "pandas")

    def _generate_answer(
        self,
        user_query: str,
        columns: list[str],
        result_str: str,
        temp: float,
        max_tok: int,
    ) -> tuple[str, dict[str, int]]:
        start = time.perf_counter()
        config = ChatConfig(
            user_message=(
                f"Columns: {columns}\n\n"
                f"Query result:\n{result_str}\n\n"
                f"User question: {user_query}\n\n"
                "Write a clear Markdown answer with a results table and key insights."
            ),
            prompt=self._answer_prompt,
            temperature=temp,
            max_tokens=max_tok,
        )
        response = self._get_kit("answer").chat(config)
        latency_ms = (time.perf_counter() - start) * 1000
        usage: dict[str, int] = response.usage or {}
        answer = response.content or ""
        self._record_span("answer_generation", latency_ms, usage)
        return answer, usage

    # ------------------------------------------------------------------
    # Internal: async step helpers
    # ------------------------------------------------------------------

    async def _agenerate_sql(
        self,
        user_query: str,
        schema_text: str,
        memory_ctx: str,
        temp: float,
        max_tok: int,
        *,
        prior_sql: str = "",
        error_msg: str = "",
    ) -> tuple[str, dict[str, int]]:
        start = time.perf_counter()
        config = ChatConfig(
            user_message=_build_sql_message(
                user_query, schema_text, memory_ctx, prior_sql, error_msg
            ),
            prompt=self._sql_prompt,
            temperature=temp,
            max_tokens=max_tok,
        )
        response = await self._get_kit("sql").achat(config)
        latency_ms = (time.perf_counter() - start) * 1000
        usage: dict[str, int] = response.usage or {}
        sql = _extract_sql(response.content or "")
        self._record_span("sql_generation", latency_ms, usage)
        return sql, usage

    async def _agenerate_analysis_code(
        self,
        user_query: str,
        columns: list[str],
        df: Any,
        temp: float,
        max_tok: int,
        engine: str = "pandas",
    ) -> tuple[str, dict[str, int]]:
        """Async version of :meth:`_generate_analysis_code`."""
        start = time.perf_counter()
        try:
            sample = df.head(5).to_string(index=False)
        except AttributeError:
            sample = str(df.head(5))

        lib_name = "polars (`pl`)" if engine == "polars" else "pandas (`pd`)"
        prompt = (
            _DEFAULT_POLARS_PROMPT if engine == "polars" else self._pandas_prompt
        )
        config = ChatConfig(
            user_message=(
                f"DataFrame columns: {columns}\n"
                f"First rows sample:\n{sample}\n\n"
                f"User question: {user_query}\n\n"
                f"Write {lib_name} code to answer the question. "
                "Assign the final result to a variable named `result`."
            ),
            prompt=prompt,
            temperature=temp,
            max_tokens=max_tok,
        )
        response = await self._get_kit("pandas").achat(config)
        latency_ms = (time.perf_counter() - start) * 1000
        usage: dict[str, int] = response.usage or {}
        code = _extract_code(response.content or "")
        self._record_span("pandas_generation", latency_ms, usage)
        return code, usage

    async def _agenerate_pandas_code(
        self,
        user_query: str,
        columns: list[str],
        df: Any,
        temp: float,
        max_tok: int,
    ) -> tuple[str, dict[str, int]]:
        return await self._agenerate_analysis_code(
            user_query, columns, df, temp, max_tok, "pandas"
        )

    async def _agenerate_answer(
        self,
        user_query: str,
        columns: list[str],
        result_str: str,
        temp: float,
        max_tok: int,
    ) -> tuple[str, dict[str, int]]:
        start = time.perf_counter()
        config = ChatConfig(
            user_message=(
                f"Columns: {columns}\n\n"
                f"Query result:\n{result_str}\n\n"
                f"User question: {user_query}\n\n"
                "Write a clear Markdown answer with a results table and key insights."
            ),
            prompt=self._answer_prompt,
            temperature=temp,
            max_tokens=max_tok,
        )
        response = await self._get_kit("answer").achat(config)
        latency_ms = (time.perf_counter() - start) * 1000
        usage: dict[str, int] = response.usage or {}
        answer = response.content or ""
        self._record_span("answer_generation", latency_ms, usage)
        return answer, usage

    # ------------------------------------------------------------------
    # Internal: chart builder
    # ------------------------------------------------------------------

    def _build_chart(
        self,
        chart_cfg: Any,
        pandas_result: Any,
        df: Any,
        pd_module: Any,
    ) -> tuple[dict[str, Any] | None, Any]:
        """Resolve chart config and build a Plotly figure deterministically."""
        if chart_cfg is None:
            return None, None

        # Prefer pandas_result if it's a DataFrame; otherwise fall back to raw df
        viz_df = (
            pandas_result
            if isinstance(pandas_result, pd_module.DataFrame)
            else df
        )

        spec: ChartSpec | None

        if isinstance(chart_cfg, str) and chart_cfg == "auto":
            spec = infer_spec(viz_df, pd_module)
        elif isinstance(chart_cfg, dict):
            spec = ChartSpec(**chart_cfg)
        elif isinstance(chart_cfg, ChartSpec):
            spec = chart_cfg
        else:
            return None, None

        if spec is None:
            return None, None

        try:
            fig = build_figure(viz_df, spec)
        except ImportError:
            # plotly not installed — return spec-only, no figure
            return spec.model_dump(), None

        return spec.model_dump(), fig

    # ------------------------------------------------------------------
    # Internal: utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _result_to_str(pandas_result: Any, df: Any, pd_module: Any) -> str:
        """Convert the pandas result (or raw df) to a compact string for the LLM."""
        target = (
            pandas_result
            if isinstance(pandas_result, pd_module.DataFrame)
            else df
        )
        if isinstance(target, pd_module.DataFrame):
            return target.to_string(index=False, max_rows=50)
        return str(target)

    @staticmethod
    def _result_to_str_any(
        analysis_result: Any, df: Any, use_polars: bool, pd_module: Any
    ) -> str:
        """Convert analysis result to string for the answer LLM.

        Works for both pandas and Polars DataFrames as well as scalars.
        """
        target = analysis_result if analysis_result is not None else df
        if use_polars:
            # Polars DataFrames have a clean __str__; slice to avoid huge outputs
            try:
                return str(target.head(50))
            except AttributeError:
                return str(target)
        # pandas path
        if pd_module is not None:
            try:
                if isinstance(target, pd_module.DataFrame):
                    return target.to_string(index=False, max_rows=50)
            except Exception:  # noqa: BLE001
                pass
        return str(target)

    @staticmethod
    def _try_get_pandas() -> Any | None:
        """Return the pandas module, or ``None`` if not installed."""
        try:
            import pandas as pd  # noqa: PLC0415
            return pd
        except ImportError:
            return None

    def _build_chart_any(
        self,
        chart_cfg: Any,
        analysis_result: Any,
        df: Any,
        pd_module: Any | None,
        use_polars: bool,
    ) -> tuple[dict[str, Any] | None, Any]:
        """Build a Plotly chart from a pandas or Polars DataFrame.

        For Polars DataFrames, converts to pandas via Arrow before building.
        When pandas is not available, returns the spec dict with no figure.
        """
        if chart_cfg is None or pd_module is None:
            return None, None

        # Convert Polars result → pandas for Plotly Express
        def _to_pd(frame: Any) -> Any:
            try:
                return frame.to_pandas()
            except Exception:  # noqa: BLE001
                return None

        pd_result: Any = None
        if use_polars and analysis_result is not None:
            pd_result = _to_pd(analysis_result)
        elif not use_polars:
            pd_result = analysis_result

        pd_df: Any = _to_pd(df) if use_polars else df

        return self._build_chart(chart_cfg, pd_result, pd_df, pd_module)

    @staticmethod
    def _resolve_connection_string(
        host: str, port: int, database: str | None,
        username: str | None, password: str | None, driver: str,
    ) -> str:
        if not database:
            raise ValueError(
                "Either 'connection_string', 'engine', or 'database' must be provided."
            )
        return _build_connection_string(host, port, database, username, password, driver)

    def _record_span(
        self, operation: str, latency_ms: float, usage: dict[str, int]
    ) -> None:
        if self._tracer is not None:
            self._tracer.record_chat_span(
                provider="pipeline",
                model=f"sql_analyst.{operation}",
                latency_ms=latency_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        if self._metrics is not None:
            self._metrics.record_request(
                provider="pipeline",
                model=f"sql_analyst.{operation}",
                operation="chat",
                status="ok",
                latency_s=latency_ms / 1000,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

    def _emit_metrics(self, usage: PipelineUsage) -> None:
        if self._metrics is not None:
            self._metrics.record_request(
                provider="pipeline",
                model="sql_analyst",
                operation="pipeline",
                status="ok",
                latency_s=0.0,
                input_tokens=usage.total_input_tokens,
                output_tokens=usage.total_output_tokens,
            )


# ---------------------------------------------------------------------------
# AsyncSQLAnalystPipeline — run() is async
# ---------------------------------------------------------------------------


class AsyncSQLAnalystPipeline(SQLAnalystPipeline):
    """Async-first variant of :class:`SQLAnalystPipeline`.

    ``run()`` is a coroutine — use ``await pipeline.run(...)`` directly.
    Designed for FastAPI, aiohttp, and other async frameworks.

    All constructor parameters and ``run()`` parameters are identical to
    :class:`SQLAnalystPipeline`.

    Example::

        from ractogateway.pipelines import AsyncSQLAnalystPipeline
        from ractogateway.openai_developer_kit import Chat

        pipeline = AsyncSQLAnalystPipeline(
            kit=Chat(model="gpt-4o"),
            pandas_kit=Chat(model="gpt-3.5-turbo"),
            max_rows=5_000,
            safe_mode=True,
        )

        # In an async context:
        result = await pipeline.run(
            user_query="Top 5 products by quantity sold?",
            connection_string="postgresql://user:pass@localhost/shop",
        )
        if result.error:
            print("Error:", result.error)
        else:
            print(result.answer)
            result.plotly_figure.show()
    """

    async def run(  # type: ignore[override]
        self,
        user_query: str,
        **kwargs: Any,
    ) -> SQLAnalystResult:
        """Async ``run()`` — delegates to :meth:`SQLAnalystPipeline.arun`."""
        return await self.arun(user_query, **kwargs)
