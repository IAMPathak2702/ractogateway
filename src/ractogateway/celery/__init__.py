"""Celery task-queue integration for RactoGateway.

Three production patterns exposed through a single :class:`RactoCeleryWorker`:

* **Never-Fail generation** — :meth:`~RactoCeleryWorker.generate` enqueues an
  LLM call that retries automatically with exponential backoff on transient
  failures (timeouts, 5xx API errors).

* **Background document ingestion** — :meth:`~RactoCeleryWorker.ingest_document`
  offloads the full ``chunk → embed → store`` RAG pipeline to a worker node so
  your web server returns a task ID immediately.

* **Parallel batch inference** — :meth:`~RactoCeleryWorker.parallel_batch` fans
  a list of items across the worker pool using Celery ``group()``.

Quick start::

    pip install ractogateway[celery]

    # tasks.py  ← imported by BOTH client and worker
    from celery import Celery
    from ractogateway import openai_developer_kit as gpt
    from ractogateway.celery import RactoCeleryWorker, RetryConfig

    celery_app = Celery(
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/0",
    )
    kit = gpt.Chat(model="gpt-4o", default_prompt=my_prompt)

    worker = RactoCeleryWorker(
        celery_app,
        kit=kit,
        retry_config=RetryConfig(max_retries=3, initial_delay_s=2.0),
    )

    # Fire-and-forget with automatic retry:
    handle = worker.generate("Summarise the quarterly report: …")
    result  = worker.wait(handle.id, timeout_s=60.0)
    print(result.result["content"])

    # Start your Celery workers:
    #   celery -A tasks.celery_app worker --loglevel=info
"""

from ractogateway.celery._models import RetryConfig, TaskResult, TaskStatus
from ractogateway.celery.worker import RactoCeleryWorker

__all__ = [
    "RactoCeleryWorker",
    "RetryConfig",
    "TaskResult",
    "TaskStatus",
]
