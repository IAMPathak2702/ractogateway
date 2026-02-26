# SQL Analyst Pipeline

`SQLAnalystPipeline` turns a plain-English question into a DB-backed answer:

1. Generate read-only SQL from schema + question
2. Execute SQL and build a dataframe
3. Optionally generate analysis code (pandas or polars)
4. Optionally generate a markdown answer
5. Optionally build a Plotly chart

Use `AsyncSQLAnalystPipeline` when your app stack is async.

## Best Use Cases

- Natural language analytics over PostgreSQL/MySQL-compatible schemas
- Internal data assistant for support, finance, growth, and operations teams
- Fast prototyping of BI-style Q&A endpoints
- Safe query generation with table/column restrictions and masking

## Minimal Example

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import SQLAnalystPipeline

pipeline = SQLAnalystPipeline(
    kit=gpt.Chat(model="gpt-4o"),
    safe_mode=True,
)

result = pipeline.run(
    user_query="Top 5 products by quantity sold in the last 30 days",
    connection_string="postgresql://user:pass@localhost:5432/shop",
)

if result.error:
    print("Pipeline error:", result.error)
else:
    print(result.sql_query)
    print(result.answer)
```

## Async Example

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import AsyncSQLAnalystPipeline

pipeline = AsyncSQLAnalystPipeline(
    kit=gpt.Chat(model="gpt-4o"),
    pandas_kit=gpt.Chat(model="gpt-4o-mini"),
)

result = await pipeline.run(
    user_query="Monthly revenue trend for 2025",
    connection_string="postgresql://user:pass@localhost:5432/analytics",
)
```

## Per-Step Model Control

Use different models per step for quality/cost balance:

```python
pipeline = SQLAnalystPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),      # fallback
    sql_kit=gpt.Chat(model="gpt-4o"),       # highest quality SQL
    pandas_kit=gpt.Chat(model="gpt-4o-mini"),
    answer_kit=gpt.Chat(model="gpt-4o"),
)
```

## Security and Data Governance

`SQLAnalystPipeline` includes controls designed for production access patterns:

- `force_read_only=True` blocks non-SELECT SQL
- `allowed_tables=[...]` hides all other tables from the LLM
- `blocked_columns=[...]` removes sensitive columns from schema context
- `mask_columns=[...]` masks values in returned rows and answer context
- `max_rows=...` auto-injects `LIMIT` when needed

```python
pipeline = SQLAnalystPipeline(
    kit=gpt.Chat(model="gpt-4o"),
    allowed_tables=["orders", "customers", "products"],
    blocked_columns=["ssn", "credit_card_number"],
    mask_columns=["email", "phone"],
    max_rows=5000,
)
```

## Charting

Use `chart="auto"` (default) or pass an explicit `ChartSpec`:

```python
from ractogateway.pipelines import ChartSpec

result = pipeline.run(
    user_query="Revenue by category",
    connection_string="postgresql://user:pass@localhost:5432/shop",
    chart=ChartSpec(
        chart_type="bar",
        x="category",
        y="revenue",
        title="Revenue by Category",
    ),
)

if result.plotly_figure is not None:
    result.plotly_figure.show()
```

## Analysis Engine: Pandas or Polars

```python
pipeline = SQLAnalystPipeline(
    kit=gpt.Chat(model="gpt-4o"),
    analysis_engine="polars",  # "pandas" (default) or "polars"
)
```

Install extras:

```bash
pip install "ractogateway[pipelines-sql-polars]"
```

## Result Object

`run()`/`arun()` return `SQLAnalystResult` with:

- `sql_query`, `columns`, `raw_rows`, `row_count`
- `pandas_code`, `pandas_result`
- `answer`
- `chart_spec`, `plotly_figure`
- `usage` token counters
- `error` (populated in `safe_mode=True`)

Export helpers:

```python
result.to_csv("query_output.csv")
result.to_json("query_output.json")
result.to_excel("query_output.xlsx")
```
