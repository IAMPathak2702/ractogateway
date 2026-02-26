# List Classifier Pipeline

`ListClassifierPipeline` maps a natural-language query to one or more options
from a `list[str]`. It validates outputs against a dynamic enum and retries when
the model returns invalid JSON or unknown options.

Use `AsyncListClassifierPipeline` in async app stacks.

## Best Use Cases

- Support ticket routing
- Intent classification for chatbots
- Lead triage (sales vs support vs onboarding)
- Queue assignment based on user request text

## Minimal Example

```python
from ractogateway import openai_developer_kit as gpt
from ractogateway.pipelines import ListClassifierPipeline

pipeline = ListClassifierPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    options=["Billing", "Technical Support", "Sales", "Account Management"],
    selection_mode="single",
    include_confidence=True,
    include_reasoning=True,
)

result = pipeline.run("I was charged twice for my subscription")
print(result.first)           # "Billing"
print(result.top_confidence)  # e.g. 0.94
print(result.reasoning)
```

## Provider Factory (`from_provider`)

Use provider/model directly without manually constructing a kit:

```python
from ractogateway.pipelines import ListClassifierPipeline

pipeline = ListClassifierPipeline.from_provider(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    options=["Billing", "Technical Support", "Sales"],
    selection_mode="single",
)
```

Supported providers: `openai`, `anthropic`, `google`, `ollama`, `huggingface`.

## Single vs Multiple Selection

```python
# exactly one option
single = pipeline.run("I cannot log in", selection_mode="single")

# one or more options
multi = pipeline.run(
    "I cannot log in and my invoice is wrong",
    selection_mode="multiple",
)
print(multi.selected)
```

## Output Formats

- `output_format="pydantic"` (default): returns `ClassifierResult`
- `output_format="dict"`: returns plain dictionary
- `output_format="string"`: returns comma-joined selected options

```python
raw_dict = pipeline.run("Need to update payment method", output_format="dict")
as_text = pipeline.run("Need invoice copy", output_format="string")
```

## Better Quality with Option Descriptions

```python
pipeline = ListClassifierPipeline(
    kit=gpt.Chat(model="gpt-4o-mini"),
    options=["Billing", "Technical Support", "Sales"],
    option_descriptions={
        "Billing": "Invoices, payments, refunds, subscription charges",
        "Technical Support": "Bugs, login issues, API errors, outages",
        "Sales": "Pricing plans, demos, contracts, upgrades",
    },
    score_all=True,
    include_confidence=True,
)
```

## Batch and Async

```python
# sync batch
results = pipeline.batch_run(
    [
        "My account is locked",
        "Can I get a volume discount?",
        "Please refund last month charge",
    ]
)

# async batch with concurrency cap
async_results = await pipeline.abatch_run(
    ["Issue 1", "Issue 2", "Issue 3"],
    max_concurrency=2,
)
```

## Production Controls

- `safe_mode=True`: return error inside result instead of raising
- `max_retries`: retry invalid model outputs
- `exact_cache`, `semantic_cache`: avoid repeated calls
- `rate_limiter` + `user_id`: enforce quotas
- `memory` + `session_id`: conversation-aware classification
- `audit_logger`: emit audit entries per call

```python
result = pipeline.run(
    "No category seems to match this request",
    uncertain_label="Other",
    confidence_threshold=0.35,
    session_id="session-42",
    user_id="user-42",
)
```

## Result Helpers

`ClassifierResult` includes convenience helpers:

- `first`, `top_confidence`, `is_empty`
- `as_string()`, `as_dict()`, `as_enum()`
- `top_n(n)`, `score_for(option)`
- `usage` with input/output/retry counters
