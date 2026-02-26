# Prebuilt Pipelines

RactoGateway prebuilt pipelines package complete, production-ready workflows on top
of developer kits. A pipeline usually combines multiple LLM calls, validation,
optional retries, and operational controls into one class with `run()` and `arun()`.

## Installation

```bash
# SQL Analyst + List Classifier
pip install "ractogateway[pipelines]"

# SQL Analyst only (requires sqlalchemy + pandas)
pip install "ractogateway[pipelines-sql]"

# SQL Analyst + Plotly chart support
pip install "ractogateway[pipelines-sql-viz]"

# SQL Analyst with Polars analysis engine
pip install "ractogateway[pipelines-sql-polars]"

# List Classifier only (no extra deps beyond core package)
pip install "ractogateway[pipelines-classifier]"
```

## Pipeline Catalog

| Pipeline | Classes | Main job | Best for |
| --- | --- | --- | --- |
| SQL Analyst | `SQLAnalystPipeline`, `AsyncSQLAnalystPipeline` | Natural language -> SQL -> analysis -> markdown answer -> optional chart | Analytics copilots, BI assistants, ops reporting |
| List Classifier | `ListClassifierPipeline`, `AsyncListClassifierPipeline` | Natural language query -> best matching option(s) from `list[str]` | Ticket routing, intent detection, queue triage |

## Common Import Pattern

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import SQLAnalystPipeline, ListClassifierPipeline

sql_pipeline = SQLAnalystPipeline(kit=gpt.Chat(model="gpt-4o"))
classifier = ListClassifierPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    options=["Billing", "Technical Support", "Sales"],
)
```

## Detailed Guides

```{toctree}
:maxdepth: 1

pipelines/sql_analyst
pipelines/list_classifier
```
