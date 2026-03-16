# API Reference - RactoMail

Module: `ractogateway.mail`

```bash
pip install "ractogateway[mail]"
```

## RactoMailKit

```python
class RactoMailKit
```

Typed email intelligence surface that consolidates the V1-V8 RactoMail feature
families into one API.

### Constructor

```python
RactoMailKit(
    *,
    kit=None,
    connectors=None,
    privacy=None,
    session_store=None,
    run_store=None,
)
```

| Parameter | Type | Description |
| --- | --- | --- |
| `kit` | `Any | None` | Optional developer kit reserved for future answer synthesis |
| `connectors` | `Sequence[BaseMailConnector] | None` | Connector instances to sync from |
| `privacy` | `MailPrivacyConfig | None` | Output redaction and hash controls |
| `session_store` | `SessionStore | None` | Multi-turn session persistence |
| `run_store` | `DAGRunStore | None` | Retained DAG execution history |

### Methods

| Method | Returns | Purpose |
| --- | --- | --- |
| `sync(mode="passive", since=None, use_flow=True)` | `SyncStatus` | Sync and index connector messages |
| `async_sync(...)` | `SyncStatus` | Async sync variant |
| `search(query, **kwargs)` | `list[EmailSearchResult]` | Ranked search without narrative answer |
| `ask(query, **kwargs)` | `MailResponse` | Grounded answer with references and evidence |
| `aask(query, **kwargs)` | `MailResponse` | Async answer variant |
| `faceted_search(query, **kwargs)` | `FacetedResult` | Search plus connector/sender/label counts |
| `entity_search(entity, top_k=10)` | `EntityResult` | Entity-focused search |
| `start_session(session_id, user, context=None)` | `ConversationSession` | Start a multi-turn session |
| `save_search(name, query, **kwargs)` | `SavedSearch` | Persist a query config |
| `run_saved_search(name)` | `MailResponse` | Replay a saved query |
| `supported_versions()` | `list[MailVersion]` | Return the consolidated version families |
| `feature_matrix()` | `dict[MailVersion, tuple[str, ...]]` | Return the V1-V8 feature map |
| `stats()` | `MailStats` | Return basic counters |
| `history()` | `list[MailQueryConfig]` | Return recorded query configs |

## Core Models

### MailMessage

Normalized email record with sender, subject, body, labels, and attachments.

### MailAttachment

Normalized attachment content with extracted text payloads.

### MailQueryConfig

Canonical query model used by `search()` and `ask()`. Includes:

- `search_mode`
- `top_k`
- `max_references`
- `verbatim`
- `fuzzy`
- `regex_patterns`
- `boolean_query`
- `amount_filter`
- `facets`

### MailResponse

Grounded answer object returned by `ask()` and `aask()`.

| Field | Type | Description |
| --- | --- | --- |
| `query` | `str` | Original user query |
| `answer` | `str` | Synthesized answer text |
| `references` | `list[EmailRef]` | Surgical citations into matching emails |
| `evidence_chain` | `list[EvidenceItem]` | Short trace of why each reference mattered |
| `search_results` | `list[EmailSearchResult]` | Ranked hits used to answer |
| `fuzzy_matches` | `list[FuzzyMatch]` | V7 fuzzy-match trace |
| `actions` | `list[MailActionSuggestion]` | V4 action suggestions |
| `total_matches` | `int` | Total ranked matches returned |
| `resolved_date_range` | `DateRange | None` | Date range inferred from Indian calendar phrases |

### EmailRef

Citation object with:

- `message_id`
- `thread_id`
- `connector_id`
- `mailbox`
- `subject`
- `sender`
- `sent_at`
- `snippet`
- `attachment`
- `sha256`

### SyncStatus

Result of `sync()` / `async_sync()`.

| Field | Type | Description |
| --- | --- | --- |
| `mode` | `SyncMode` | Selected sync mode |
| `indexed_count` | `int` | Number of indexed messages |
| `thread_count` | `int` | Number of threads represented |
| `connector_counts` | `dict[str, int]` | Per-connector counts |
| `deduplicated_count` | `int` | Messages skipped as duplicates |
| `used_flow` | `bool` | Whether the DAG runner was used |
| `dag_run_id` | `str | None` | Run ID when `use_flow=True` |

## Connectors

Built-in read-only connector classes:

- `GmailConnector`
- `OutlookConnector`
- `ExchangeConnector`
- `IMAPConnector`
- `GoDaddyConnector`
- `ZohoConnector`
- `YahooConnector`
- `MockMailConnector`

## Search Helpers

- `FuzzySearch`
- `AmountFilter`
- `MailFacets`
- `IndianCalendar`

## Memory

- `SessionStore`
- `ConversationSession`
- `ConversationTurn`
- `SavedSearch`

## Flow

- `RactoDAG`
- `PythonOperator`
- `TaskRun`
- `DAGRun`
- `DAGRunStore`

## Constants

`SUPPORTED_VERSION_FEATURES` exposes the consolidated V1-V8 feature map.

## See Also

- [Guide - RactoMail](../guide/mail.md)
- [Guide - Pipelines](../guide/pipelines.md)

