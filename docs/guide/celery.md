# Celery

`RactoCeleryWorker` integrates RactoGateway with Celery for durable background LLM tasks, automatic retries, and parallel batch inference.

## Installation

```bash
pip install ractogateway[celery]
```

## Setup

Define your worker in a shared `tasks.py` module imported by both client and worker processes:

```python
# tasks.py
from celery import Celery
from ractogateway import openai_developer_kit as gpt
from ractogateway.celery import RactoCeleryWorker, RetryConfig
from ractogateway.prompts.engine import RactoPrompt

celery_app = Celery(
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0",
)

my_prompt = RactoPrompt(
    system="You are a helpful assistant.",
    output_format="Return a concise answer.",
)

kit = gpt.OpenAIDeveloperKit(model="gpt-4o", default_prompt=my_prompt)

worker = RactoCeleryWorker(
    celery_app,
    kit=kit,
    retry_config=RetryConfig(max_retries=3, initial_delay_s=2.0),
)
```

## Never-Fail Generation

Enqueue an LLM call that retries automatically with exponential backoff:

```python
handle = worker.generate("Summarise the quarterly report: …")
result = worker.wait(handle.id, timeout_s=60.0)
print(result.result["content"])
```

## Background Document Ingestion

Offload the full RAG `chunk → embed → store` pipeline to a worker node:

```python
handle = worker.ingest_document("/path/to/report.pdf")
worker.wait(handle.id, timeout_s=120.0)
```

## Parallel Batch Inference

Fan a list of items across the worker pool using Celery `group()`:

```python
items = ["Summarise doc 1", "Summarise doc 2", "Summarise doc 3"]
results = worker.parallel_batch(items)
for r in results:
    print(r.result["content"])
```

## Starting Workers

```bash
celery -A tasks.celery_app worker --loglevel=info
```
