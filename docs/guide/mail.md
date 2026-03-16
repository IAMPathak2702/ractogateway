# RactoMail

`RactoMailKit` adds email intelligence to RactoGateway through one typed API.
This foundation consolidates the V1-V8 RactoMail feature families into a single
surface: multi-connector sync, grounded email Q&A, saved searches, conversation
memory, V7 search helpers, and a lightweight V8 DAG runner.

## Installation

```bash
pip install "ractogateway[mail]"
```

## Quick Start

```python
from datetime import UTC, datetime

from ractogateway.mail import (
    MailAddress,
    MailMessage,
    MockMailConnector,
    RactoMailKit,
)

messages = [
    MailMessage(
        message_id="m-1",
        thread_id="t-1",
        connector_id="mock",
        subject="Invoice dispute INV-2024-447",
        sender=MailAddress(email="mehta@example.com", name="Mehta"),
        sent_at=datetime(2024, 2, 20, 10, 0, tzinfo=UTC),
        body_text="GST dispute raised on invoice INV-2024-447 for Rs. 118,200.",
    )
]

mail = RactoMailKit(connectors=[MockMailConnector(messages=messages)])
mail.sync()

response = mail.ask("What invoice disputes do we have?")
print(response.answer)
print(response.references[0].subject)
```

## Connectors

The built-in connector classes normalize messages into `MailMessage` records:

- `GmailConnector`
- `OutlookConnector`
- `ExchangeConnector`
- `IMAPConnector`
- `GoDaddyConnector`
- `ZohoConnector`
- `YahooConnector`
- `MockMailConnector`

The current implementation is fixture-backed and read-only, which makes it safe
for local testing and pipeline integration without real mailbox credentials.

## Search and Ask

`search()` returns ranked `EmailSearchResult` objects. `ask()` wraps the same
engine and adds grounded references, evidence chains, and V4-style action hints.

```python
from ractogateway.mail import AmountFilter, FuzzySearch

results = mail.search(
    "invoice from Mehtta",
    fuzzy=FuzzySearch(enabled=True, max_distance=2),
    regex_patterns={"invoice": r"INV-[0-9]{4}-[0-9]{3}"},
    amount_filter=AmountFilter(min_amount=100_000),
)

answer = mail.ask("invoice from Mehtta", fuzzy=FuzzySearch(enabled=True))
print(answer.fuzzy_matches)
print(answer.references)
```

Built-in V7 helpers include:

- Fuzzy matching
- Regex search
- Boolean filtering
- Indian calendar date resolution
- Amount-range filtering

## Conversation Memory

```python
session = mail.start_session(session_id="vendor-review", user="ceo@example.com")

first = session.ask("Show me all complaints")
second = session.ask("Only the complaints from Mehta")

print(first.total_matches)
print(second.references[0].sender)
```

You can also persist and replay searches:

```python
mail.save_search("mehta_complaints", "Mehta complaint")
replay = mail.run_saved_search("mehta_complaints")
```

## RactoFlow DAG

`ractogateway.mail.flow` provides a lightweight DAG runner for V8-style task
parallelism.

```python
from ractogateway.mail import PythonOperator, RactoDAG

dag = RactoDAG(dag_id="mail_query", max_parallel_tasks=2)
fetch = dag.add_task(PythonOperator(task_id="fetch", python_callable=lambda _ctx: ["m-1"]))
classify = dag.add_task(PythonOperator(task_id="classify", python_callable=lambda _ctx: "finance"))
combine = dag.add_task(
    PythonOperator(
        task_id="combine",
        python_callable=lambda ctx: (
            ctx["task_outputs"]["fetch"],
            ctx["task_outputs"]["classify"],
        ),
    )
)

fetch >> combine
classify >> combine

run = dag.run()
print(run.status)
print(run.task_runs["combine"].output)
```

## Version Matrix

`mail.supported_versions()` returns all eight version families. The feature
matrix is also available through `mail.feature_matrix()` and
`SUPPORTED_VERSION_FEATURES`.

