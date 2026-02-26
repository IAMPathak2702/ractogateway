"""Helpers for schema fetching, connection building, code extraction, and pandas execution."""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
import re
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "len", "range", "enumerate", "zip", "list", "dict", "set", "tuple",
        "str", "int", "float", "bool", "sum", "min", "max", "round", "sorted",
        "print", "abs", "any", "all", "isinstance", "type", "hasattr", "getattr",
        "map", "filter", "reversed", "next", "iter",
    )
    if hasattr(builtins, name)
}


def _require_sqlalchemy() -> Any:
    try:
        import sqlalchemy
    except ImportError as exc:
        raise ImportError(
            "The 'sqlalchemy' package is required for SQLAnalystPipeline. "
            "Install it with:  pip install ractogateway[pipelines-sql]"
        ) from exc
    return sqlalchemy


def _require_pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:
        raise ImportError(
            "The 'pandas' package is required for SQLAnalystPipeline. "
            "Install it with:  pip install ractogateway[pipelines-sql]"
        ) from exc
    return pandas


def _require_polars() -> Any:
    try:
        import polars
    except ImportError as exc:
        raise ImportError(
            "The 'polars' package is required when analysis_engine='polars'. "
            "Install it with:  pip install ractogateway[pipelines-sql-polars]"
        ) from exc
    return polars


# ---------------------------------------------------------------------------
# Connection string builder
# ---------------------------------------------------------------------------


def _build_connection_string(
    host: str,
    port: int,
    database: str,
    username: str | None,
    password: str | None,
    driver: str,
) -> str:
    """Build a SQLAlchemy connection URI from individual parameters.

    For SQLite pass ``driver="sqlite"``; ``host`` and ``port`` are ignored.

    Parameters
    ----------
    host:
        Database host (e.g. ``"localhost"``).
    port:
        Database port (e.g. ``5432``).
    database:
        Database name or, for SQLite, the file path.
    username:
        Optional username.
    password:
        Optional password (only used when *username* is set).
    driver:
        SQLAlchemy dialect string, e.g. ``"postgresql"``, ``"mysql"``,
        ``"sqlite"``, ``"mssql+pyodbc"``.
    """
    if driver.startswith("sqlite"):
        return f"sqlite:///{database}"
    creds = ""
    if username:
        creds = f"{username}:{password}@" if password else f"{username}@"
    return f"{driver}://{creds}{host}:{port}/{database}"


# ---------------------------------------------------------------------------
# SQL / code extraction from LLM output
# ---------------------------------------------------------------------------

_SQL_FENCE_RE: re.Pattern[str] = re.compile(
    r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)
_CODE_FENCE_RE: re.Pattern[str] = re.compile(
    r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)


def _extract_sql(text: str) -> str:
    """Extract the SQL query from a fenced code block, or return the text as-is."""
    m = _SQL_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_code(text: str) -> str:
    """Extract Python code from a fenced code block, or return the text as-is."""
    m = _CODE_FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Row-limit injection
# ---------------------------------------------------------------------------

_LIMIT_RE: re.Pattern[str] = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def _inject_limit(sql: str, max_rows: int) -> str:
    """Append ``LIMIT {max_rows}`` to *sql* if it has no existing LIMIT clause.

    Parameters
    ----------
    sql:
        The SQL query string.
    max_rows:
        Maximum rows to return.  Pass ``0`` or negative to skip injection.

    Returns
    -------
    str
        SQL with LIMIT clause appended if needed.
    """
    if max_rows <= 0 or _LIMIT_RE.search(sql):
        return sql
    return sql.rstrip().rstrip(";") + f" LIMIT {max_rows}"


# ---------------------------------------------------------------------------
# Result row masking
# ---------------------------------------------------------------------------

_MASK_VALUE: str = "***MASKED***"


def _mask_rows(
    rows: list[dict[str, Any]],
    mask_columns: list[str] | None,
) -> list[dict[str, Any]]:
    """Replace values in *mask_columns* with a placeholder string in every row.

    Parameters
    ----------
    rows:
        List of result rows (dicts of column → value).
    mask_columns:
        Column names whose values should be masked.  Case-insensitive match.
        Pass ``None`` or an empty list to skip masking.

    Returns
    -------
    list[dict[str, Any]]
        New list of rows with sensitive columns replaced by ``"***MASKED***"``.
    """
    if not mask_columns:
        return rows
    mask_set = {c.lower() for c in mask_columns}
    return [
        {k: (_MASK_VALUE if k.lower() in mask_set else v) for k, v in row.items()}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Schema cache  (TTL, process-scoped, thread-safe)
# ---------------------------------------------------------------------------

_SCHEMA_CACHE: dict[str, tuple[str, float]] = {}
_SCHEMA_LOCK: threading.Lock = threading.Lock()


def _schema_cache_key(conn_str: str) -> str:
    """Return a stable hex digest for *conn_str* suitable as a cache key."""
    return hashlib.sha256(conn_str.encode()).hexdigest()


def _get_cached_schema(conn_str: str, ttl: float) -> str | None:
    """Return the cached schema for *conn_str*, or ``None`` on miss/expiry.

    Parameters
    ----------
    conn_str:
        The SQLAlchemy connection URI (used as cache key).
    ttl:
        Cache time-to-live in seconds.  Pass ``0`` or negative to disable.
    """
    if ttl <= 0:
        return None
    key = _schema_cache_key(conn_str)
    with _SCHEMA_LOCK:
        entry = _SCHEMA_CACHE.get(key)
        if entry is None:
            return None
        schema, ts = entry
        if time.time() - ts > ttl:
            del _SCHEMA_CACHE[key]
            return None
        return schema


def _put_cached_schema(conn_str: str, schema: str) -> None:
    """Store *schema* in the process-level cache keyed by *conn_str*."""
    key = _schema_cache_key(conn_str)
    with _SCHEMA_LOCK:
        _SCHEMA_CACHE[key] = (schema, time.time())


def clear_schema_cache() -> None:
    """Evict all entries from the in-process schema cache.

    Useful in tests or when the database schema changes at runtime.
    """
    with _SCHEMA_LOCK:
        _SCHEMA_CACHE.clear()


# ---------------------------------------------------------------------------
# Schema RBAC + documentation filtering
# ---------------------------------------------------------------------------


def _filter_schema(
    schema: str,
    allowed_tables: list[str] | None,
    blocked_columns: list[str] | None,
    table_docs: dict[str, str] | None,
    column_docs: dict[str, dict[str, str]] | None,
) -> str:
    """Filter and annotate the schema string for RBAC and custom documentation.

    Operates on the plain-text schema produced by :class:`SchemaFetcher`.
    All filtering is case-insensitive.

    Parameters
    ----------
    schema:
        Raw schema string from :class:`SchemaFetcher`.
    allowed_tables:
        If set, only these tables appear in the output.  All others are hidden
        from the LLM so it cannot generate SQL that references them.
    blocked_columns:
        Column names to suppress from every table.  Useful for hiding PII or
        sensitive fields from the LLM.
    table_docs:
        ``{table_name: description}`` — appended as inline comments so the
        LLM understands table semantics.
    column_docs:
        ``{table_name: {column_name: description}}`` — per-column inline
        comments for additional LLM context.

    Returns
    -------
    str
        Filtered and annotated schema string.
    """
    if not any([allowed_tables, blocked_columns, table_docs, column_docs]):
        return schema

    allowed_set = {t.lower() for t in allowed_tables} if allowed_tables else None
    blocked_set = {c.lower() for c in blocked_columns} if blocked_columns else None

    blocks: list[str] = []
    for block in schema.split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue

        first = lines[0]
        if not first.startswith("Table: "):
            # Non-table block (unlikely, but preserve it)
            blocks.append(block)
            continue

        table_name = first[len("Table: "):].strip()

        # RBAC: skip tables not in allowed_tables
        if allowed_set is not None and table_name.lower() not in allowed_set:
            continue

        # Annotate table header with doc comment
        header = first
        if table_docs and table_name in table_docs:
            header = f"{first}  -- {table_docs[table_name]}"

        new_lines = [header]
        tbl_col_docs: dict[str, str] = (column_docs or {}).get(table_name, {})

        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue

            # Preserve constraint lines (PRIMARY KEY / FOREIGN KEY) as-is
            su = stripped.upper()
            if su.startswith("PRIMARY KEY") or su.startswith("FOREIGN KEY"):
                new_lines.append(line)
                continue

            # Extract column name (first token before the type)
            parts = stripped.split()
            col_name = parts[0] if parts else ""

            # RBAC: skip blocked columns
            if blocked_set and col_name.lower() in blocked_set:
                continue

            # Annotate with column doc comment
            annotated = line
            if col_name and col_name in tbl_col_docs:
                annotated = f"{line}  -- {tbl_col_docs[col_name]}"
            new_lines.append(annotated)

        blocks.append("\n".join(new_lines))

    return "\n\n".join(blocks) if blocks else "(no accessible tables)"


# ---------------------------------------------------------------------------
# Pandas execution sandbox
# ---------------------------------------------------------------------------


def _execute_pandas(code: str, df: Any, pd_module: Any) -> Any:
    """Execute LLM-generated pandas code in a restricted namespace.

    The LLM is instructed to assign its final answer to ``result``.
    If the variable is not set, the original DataFrame is returned.

    Parameters
    ----------
    code:
        Python source code string to execute.
    df:
        The pandas ``DataFrame`` produced by the SQL query.
    pd_module:
        The pandas module (passed in to avoid re-importing).

    Returns
    -------
    Any
        Whatever was assigned to ``result`` inside *code*, or *df* as fallback.
    """
    namespace: dict[str, Any] = {
        "df": df,
        "pd": pd_module,
        "result": df,
        "__builtins__": _SAFE_BUILTINS,
    }
    exec(compile(code, "<pandas_code>", "exec"), namespace)  # noqa: S102
    return namespace.get("result", df)


def _execute_polars(code: str, df: Any, pl_module: Any) -> Any:
    """Execute LLM-generated Polars code in a restricted namespace.

    Identical sandbox pattern to :func:`_execute_pandas` but exposes ``pl``
    (the polars module) instead of ``pd``.

    Parameters
    ----------
    code:
        Python source code string to execute.
    df:
        A Polars ``DataFrame`` produced by the SQL query.
    pl_module:
        The polars module (passed in to avoid re-importing).

    Returns
    -------
    Any
        Whatever was assigned to ``result`` inside *code*, or *df* as fallback.
    """
    namespace: dict[str, Any] = {
        "df": df,
        "pl": pl_module,
        "result": df,
        "__builtins__": _SAFE_BUILTINS,
    }
    exec(compile(code, "<polars_code>", "exec"), namespace)  # noqa: S102
    return namespace.get("result", df)


# ---------------------------------------------------------------------------
# CSV export helper (no pandas required)
# ---------------------------------------------------------------------------


def _rows_to_csv(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Serialize *rows* to a CSV string without requiring pandas."""
    buf = io.StringIO()
    fieldnames = columns or (list(rows[0].keys()) if rows else [])
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _rows_to_csv_file(rows: list[dict[str, Any]], columns: list[str], path: str) -> None:
    """Write *rows* to a CSV file at *path* without requiring pandas."""
    fieldnames = columns or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:  # noqa: PTH123
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Schema fetcher helpers
# ---------------------------------------------------------------------------

# Column type prefixes that are likely to hold categorical / enum values,
# making it useful to include sample distinct values in the schema.
_CATEGORICAL_PREFIXES: frozenset[str] = frozenset(
    {"varchar", "text", "char", "nvarchar", "nchar", "string", "enum", "tinytext"}
)


def _is_categorical(col_type: str) -> bool:
    """Return True when *col_type* looks like a low-cardinality string column."""
    return col_type.lower().split("(")[0].strip() in _CATEGORICAL_PREFIXES


# ---------------------------------------------------------------------------
# Schema fetcher
# ---------------------------------------------------------------------------


class SchemaFetcher:
    """Introspects a SQLAlchemy engine and returns a rich, LLM-ready schema string.

    The richer the schema, the better the SQL the LLM produces.  Three optional
    enrichments go beyond basic column names and types:

    * **Indexes** (``include_indexes=True``, default) — tells the LLM which
      columns are indexed so it can write efficient WHERE conditions.
    * **Row counts** (``include_row_counts=True``) — tells the LLM the scale of
      each table (e.g. *~1,200 rows* vs *~50,000,000 rows*) so it knows when
      to be careful with joins and can make smarter filtering choices.
    * **Sample values** (``include_sample_values=True``) — for low-cardinality
      string columns the fetcher appends a few representative distinct values
      so the LLM knows the exact strings to use in WHERE clauses (e.g.
      ``status IN ('active', 'inactive')`` rather than guessing).

    Example output (all enrichments on)::

        Table: orders  (~12,540 rows)
          id INTEGER NOT NULL
          status VARCHAR  -- e.g. 'pending', 'shipped', 'delivered', 'cancelled'
          customer_id INTEGER NOT NULL
          total NUMERIC NOT NULL
          created_at TIMESTAMP NOT NULL
          PRIMARY KEY (id)
          FOREIGN KEY (customer_id) REFERENCES customers(id)
          INDEX (status)
          INDEX (created_at)

        Table: customers  (~3,200 rows)
          id INTEGER NOT NULL
          name VARCHAR NOT NULL
          region VARCHAR  -- e.g. 'North', 'South', 'East', 'West'
          PRIMARY KEY (id)
          INDEX UNIQUE (email)
    """

    @staticmethod
    def fetch(
        engine: Any,
        *,
        include_indexes: bool = True,
        include_row_counts: bool = False,
        include_sample_values: bool = False,
        sample_value_limit: int = 8,
        sample_cardinality_threshold: int = 25,
    ) -> str:
        """Return an enriched schema string by inspecting *engine*.

        Parameters
        ----------
        engine:
            A SQLAlchemy ``Engine`` instance.
        include_indexes:
            Append ``INDEX`` lines for each non-PK index.  Zero extra queries —
            data comes from the SQLAlchemy inspector.  Default: ``True``.
        include_row_counts:
            Append approximate row counts to table headers via
            ``SELECT COUNT(*)``.  One extra query per table.  Default: ``False``.
        include_sample_values:
            For string/categorical columns with ≤ *sample_cardinality_threshold*
            distinct values, append sample values as an inline comment.  One
            extra query per eligible column.  Default: ``False``.
        sample_value_limit:
            Maximum number of sample values to show per column.  Default: ``8``.
        sample_cardinality_threshold:
            Only show sample values when the column has at most this many
            distinct values (avoids showing 10k-row free-text columns).
            Default: ``25``.
        """
        sa = _require_sqlalchemy()
        inspector = sa.inspect(engine)
        parts: list[str] = []
        need_conn = include_row_counts or include_sample_values

        # Open a single connection for all optional extra queries
        extra_conn: Any = None
        try:
            if need_conn:
                extra_conn = engine.connect()

            for table_name in inspector.get_table_names():
                col_lines: list[str] = []

                # ── Row count header hint ──────────────────────────────
                row_count_hint = ""
                if include_row_counts and extra_conn is not None:
                    try:
                        rc = extra_conn.execute(
                            sa.text(f"SELECT COUNT(*) FROM {table_name}")
                        ).scalar()
                        row_count_hint = f"  (~{rc:,} rows)"
                    except Exception:  # noqa: BLE001
                        pass

                # ── Columns ────────────────────────────────────────────
                for col in inspector.get_columns(table_name):
                    col_name: str = col["name"]
                    col_type: str = str(col["type"])
                    nullable: str = "" if col.get("nullable", True) else " NOT NULL"
                    line = f"  {col_name} {col_type}{nullable}"

                    # Optionally append sample distinct values
                    if (
                        include_sample_values
                        and extra_conn is not None
                        and _is_categorical(col_type)
                    ):
                        try:
                            # Only fetch samples for low-cardinality columns
                            n_distinct = extra_conn.execute(
                                sa.text(
                                    f"SELECT COUNT(DISTINCT {col_name}) "
                                    f"FROM {table_name}"
                                )
                            ).scalar()
                            if n_distinct is not None and n_distinct <= sample_cardinality_threshold:
                                rows = extra_conn.execute(
                                    sa.text(
                                        f"SELECT DISTINCT {col_name} "
                                        f"FROM {table_name} "
                                        f"WHERE {col_name} IS NOT NULL "
                                        f"LIMIT {sample_value_limit}"
                                    )
                                ).fetchall()
                                samples = [str(r[0]) for r in rows if r[0] is not None]
                                if samples:
                                    quoted = ", ".join(f"'{s}'" for s in samples)
                                    line += f"  -- e.g. {quoted}"
                        except Exception:  # noqa: BLE001
                            pass

                    col_lines.append(line)

                # ── Primary key ────────────────────────────────────────
                pk_info = inspector.get_pk_constraint(table_name)
                pk_cols: list[str] = pk_info.get("constrained_columns", [])
                if pk_cols:
                    col_lines.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")

                # ── Foreign keys ───────────────────────────────────────
                for fk in inspector.get_foreign_keys(table_name):
                    src = ", ".join(fk["constrained_columns"])
                    ref_table = fk["referred_table"]
                    ref_cols = ", ".join(fk["referred_columns"])
                    col_lines.append(
                        f"  FOREIGN KEY ({src}) REFERENCES {ref_table}({ref_cols})"
                    )

                # ── Indexes ────────────────────────────────────────────
                if include_indexes:
                    try:
                        for idx in inspector.get_indexes(table_name):
                            idx_cols_list: list[str] = idx.get("column_names") or []
                            if not idx_cols_list:
                                continue
                            idx_cols_str = ", ".join(idx_cols_list)
                            unique_tag = " UNIQUE" if idx.get("unique") else ""
                            col_lines.append(f"  INDEX{unique_tag} ({idx_cols_str})")
                    except Exception:  # noqa: BLE001
                        pass  # Some dialects don't support get_indexes

                header = f"Table: {table_name}{row_count_hint}"
                parts.append(header + "\n" + "\n".join(col_lines))

        finally:
            if extra_conn is not None:
                extra_conn.close()

        return "\n\n".join(parts) if parts else "(no tables found)"
