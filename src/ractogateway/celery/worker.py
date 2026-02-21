"""RactoCeleryWorker — Celery-backed async task queue for RactoGateway.

Three production patterns in one class:

1. **Never-Fail generation** — :meth:`generate` enqueues an LLM call that
   automatically retries with exponential backoff on transient failures
   (timeouts, 5xx API errors).  Auth errors and bad inputs are *not* retried.

2. **Background document ingestion** — :meth:`ingest_document` offloads the
   full ``chunk → embed → store`` pipeline to a worker node so your web server
   returns immediately with a task ID.

3. **Parallel batch inference** — :meth:`parallel_batch` fans a list of items
   out to the worker pool using Celery ``group()``, allowing N workers to
   process them concurrently.

How Celery serialization works here
------------------------------------
Celery workers run in **separate processes** — live Python objects (kit, rag)
cannot be sent over the message broker.  Instead:

* Task arguments are JSON-primitive: ``str``, ``float``, ``int``, ``dict``,
  ``list``.  The task reconstructs :class:`ChatConfig` / :class:`Message`
  objects *inside* the worker.
* The ``kit`` and ``rag`` objects are captured by closure.  For this to work
  in a distributed worker fleet, **the same module that instantiates**
  ``RactoCeleryWorker`` **must be imported by the worker process** — identical
  to the standard Flask-Celery / Django-Celery pattern.

Example
-------
::

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
        retry_config=RetryConfig(max_retries=3),
    )

    # Start workers:
    #   celery -A tasks.celery_app worker --loglevel=info

    # In your request handler:
    handle = worker.generate("Summarise this report: …")
    result  = worker.wait(handle.id, timeout_s=60.0)
    print(result.result["content"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ractogateway.celery._models import RetryConfig, TaskResult, TaskStatus

if TYPE_CHECKING:
    from ractogateway.batch._models import BatchItem

DEFAULT_MAX_TOKENS = 4096


def _require_celery() -> Any:
    try:
        import celery as celery_lib
    except ImportError as exc:
        raise ImportError(
            "The 'celery' package is required for RactoCeleryWorker. "
            "Install it with:  pip install ractogateway[celery]"
        ) from exc
    return celery_lib


def _backoff_delay(cfg: RetryConfig, attempt: int) -> float:
    """Return the countdown (seconds) for retry attempt *attempt* (0-based)."""
    return min(cfg.initial_delay_s * (cfg.backoff_factor**attempt), cfg.max_delay_s)


def _map_task_result(async_result: Any, task_id: str) -> TaskResult:
    """Convert a ``celery.result.AsyncResult`` to a :class:`TaskResult`."""
    state = async_result.state  # Celery native string
    state_map = {
        "PENDING": TaskStatus.PENDING,
        "STARTED": TaskStatus.STARTED,
        "SUCCESS": TaskStatus.SUCCESS,
        "FAILURE": TaskStatus.FAILURE,
        "RETRY": TaskStatus.RETRY,
        "REVOKED": TaskStatus.REVOKED,
    }
    status = state_map.get(state, TaskStatus.PENDING)

    result: Any = None
    error: str | None = None

    if status == TaskStatus.SUCCESS:
        result = async_result.result
    elif status == TaskStatus.FAILURE:
        exc = async_result.result  # Celery stores the exception object here
        error = str(exc) if exc is not None else "Unknown task failure"

    return TaskResult(task_id=task_id, status=status, result=result, error=error)


class RactoCeleryWorker:
    """Celery-backed task queue wrapper for RactoGateway developer kits.

    Parameters
    ----------
    app:
        A pre-configured ``celery.Celery`` instance with broker and backend
        already set.  You create and configure it yourself so you retain full
        control over serializers, routing, concurrency, etc.
    kit:
        Any RactoGateway developer kit — ``OpenAIDeveloperKit``,
        ``GoogleDeveloperKit``, or ``AnthropicDeveloperKit``.  The kit's
        ``default_prompt`` is used by generation tasks (prompts are not
        serialisable over the broker).
    rag:
        Optional :class:`~ractogateway.rag.RactoRAG` instance.  Required only
        when calling :meth:`ingest_document`.
    retry_config:
        Exponential-backoff configuration.  Defaults are applied when ``None``.
    """

    def __init__(
        self,
        app: Any,
        *,
        kit: Any,
        rag: Any | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        _require_celery()  # validate install early
        self._app = app
        self._kit = kit
        self._rag = rag
        self._cfg = retry_config or RetryConfig()
        self._register_tasks()

    # ------------------------------------------------------------------
    # Task registration
    # ------------------------------------------------------------------

    def _register_tasks(self) -> None:
        """Dynamically register Celery tasks on ``self._app``.

        Tasks are closures that capture ``kit`` and ``rag`` by reference.
        They are registered once at worker startup when this module is imported.
        """
        cfg = self._cfg
        kit = self._kit
        rag = self._rag

        # ── Task 1: LLM generation with exponential-backoff retry ─────────────

        @self._app.task(  # type: ignore[untyped-decorator]
            bind=True,
            name="ractogateway.generate",
            max_retries=cfg.max_retries,
        )
        def _generate(
            task_self: Any,
            user_message: str,
            temperature: float,
            max_tokens: int,
            history: list[dict[str, str]],
            extra: dict[str, Any],
        ) -> dict[str, Any]:
            from ractogateway._models.chat import ChatConfig, Message
            from ractogateway.exceptions import (
                RactoGatewayAPIError,
                RactoGatewayTimeoutError,
            )

            config = ChatConfig(
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
                history=[Message.model_validate(m) for m in history],
                extra=extra,
            )
            try:
                response = kit.chat(config)
                return cast("dict[str, Any]", response.model_dump())
            except (RactoGatewayTimeoutError, RactoGatewayAPIError) as exc:
                delay = _backoff_delay(cfg, task_self.request.retries)
                raise task_self.retry(
                    exc=exc,
                    countdown=delay,
                    max_retries=cfg.max_retries,
                ) from exc
            # Auth errors, validation errors, etc. → let them propagate; no retry.

        self._generate_task = _generate

        # ── Task 2: Background RAG document ingestion ──────────────────────────

        @self._app.task(  # type: ignore[untyped-decorator]
            bind=True,
            name="ractogateway.ingest",
            max_retries=cfg.max_retries,
        )
        def _ingest(
            task_self: Any,
            path: str,
            metadata: dict[str, Any],
        ) -> list[dict[str, Any]]:
            from ractogateway.exceptions import (
                RactoGatewayAPIError,
                RactoGatewayTimeoutError,
            )

            if rag is None:
                raise RuntimeError(
                    "RactoRAG is required for ingest_document. "
                    "Pass rag=<RactoRAG instance> to RactoCeleryWorker."
                )
            try:
                chunks = rag.ingest(path, **metadata)
                return [c.model_dump() for c in chunks]
            except (RactoGatewayTimeoutError, RactoGatewayAPIError) as exc:
                delay = _backoff_delay(cfg, task_self.request.retries)
                raise task_self.retry(
                    exc=exc,
                    countdown=delay,
                    max_retries=cfg.max_retries,
                ) from exc

        self._ingest_task = _ingest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        user_message: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        history: list[dict[str, str]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        """Enqueue an LLM generation task and return immediately.

        The task automatically retries with exponential backoff on transient
        errors (:class:`~ractogateway.exceptions.RactoGatewayTimeoutError` and
        :class:`~ractogateway.exceptions.RactoGatewayAPIError`).

        Parameters
        ----------
        user_message:
            The prompt text for this generation.
        temperature:
            Sampling temperature passed to the LLM.
        max_tokens:
            Maximum completion tokens.
        history:
            Previous conversation turns as ``[{"role": …, "content": …}, …]``.
            Use this to continue a multi-turn dialogue asynchronously.
        extra:
            Provider-specific pass-through parameters (``top_p``, ``seed``, …).

        Returns
        -------
        celery.result.AsyncResult
            Call ``.id`` for the task UUID.  Use :meth:`wait` or
            :meth:`get_result` to retrieve the outcome.

        Note
        ----
        The prompt is **not** an argument because Pydantic models containing
        ``type[BaseModel]`` fields cannot be serialised over the message broker.
        Set your prompt on the kit's ``default_prompt`` at construction time.
        """
        return self._generate_task.delay(
            user_message,
            temperature,
            max_tokens,
            history or [],
            extra or {},
        )

    def ingest_document(self, path: str, **metadata: Any) -> Any:
        """Enqueue a background RAG document-ingestion task.

        The full ``read → chunk → process → embed → store`` pipeline runs in
        a Celery worker.  Your web request returns immediately with an
        ``AsyncResult`` whose ``.id`` you can poll later.

        Parameters
        ----------
        path:
            Absolute or relative path to the file to ingest.  The string is
            passed as-is to :meth:`~ractogateway.rag.RactoRAG.ingest`.
        **metadata:
            Extra key-value metadata merged into every chunk's
            ``ChunkMetadata.extra``.

        Returns
        -------
        celery.result.AsyncResult

        Raises
        ------
        RuntimeError
            If ``rag`` was not provided to :class:`RactoCeleryWorker`.
        """
        return self._ingest_task.delay(path, metadata)

    def parallel_batch(
        self,
        items: list[BatchItem],
        *,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Any:
        """Fan a list of items out to the worker pool in parallel.

        Uses Celery ``group()`` so all tasks are submitted at once and run
        concurrently across available workers.

        Parameters
        ----------
        items:
            A list of :class:`~ractogateway.batch._models.BatchItem` objects.
            Each item's ``user_message`` becomes one generation task.
        temperature:
            Shared sampling temperature for all items.
        max_tokens:
            Shared max-tokens limit for all items.

        Returns
        -------
        celery.result.GroupResult
            Call :meth:`wait_parallel` to block until all tasks finish, or
            iterate ``.results`` for individual ``AsyncResult`` objects.
        """
        from celery import group

        tasks = group(
            self._generate_task.s(
                item.user_message,
                item.temperature if item.temperature != 0.0 else temperature,
                item.max_tokens if item.max_tokens != DEFAULT_MAX_TOKENS else max_tokens,
                [],          # no history for batch items
                item.extra,
            )
            for item in items
        )
        return tasks.apply_async()

    def get_result(self, task_id: str) -> TaskResult:
        """Return the current state of a task without blocking.

        Parameters
        ----------
        task_id:
            The UUID returned by :meth:`generate`, :meth:`ingest_document`, or
            the ``.id`` attribute of an ``AsyncResult``.

        Returns
        -------
        TaskResult
            ``status`` will be ``PENDING`` if the task has not started yet.
        """
        async_result = self._app.AsyncResult(task_id)
        return _map_task_result(async_result, task_id)

    def wait(
        self,
        task_id: str,
        *,
        timeout_s: float | None = None,
    ) -> TaskResult:
        """Block until a task completes (or times out) and return its result.

        Parameters
        ----------
        task_id:
            The task UUID from :meth:`generate` or :meth:`ingest_document`.
        timeout_s:
            Maximum seconds to wait.  ``None`` = wait indefinitely.

        Returns
        -------
        TaskResult
            ``result.ok`` is ``True`` on success; ``result.error`` is set on
            failure or timeout.
        """
        async_result = self._app.AsyncResult(task_id)
        try:
            value = async_result.get(timeout=timeout_s)
            return TaskResult(task_id=task_id, status=TaskStatus.SUCCESS, result=value)
        except Exception as exc:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                error=str(exc),
            )

    def wait_parallel(
        self,
        group_result: Any,
        *,
        timeout_s: float | None = None,
    ) -> list[TaskResult]:
        """Block until all tasks from :meth:`parallel_batch` complete.

        Parameters
        ----------
        group_result:
            The ``celery.result.GroupResult`` returned by :meth:`parallel_batch`.
        timeout_s:
            Maximum seconds to wait for the *entire group*.  ``None`` = wait
            indefinitely.

        Returns
        -------
        list[TaskResult]
            One :class:`TaskResult` per item, in submission order.  Inspect
            each ``.ok`` / ``.error`` individually.
        """
        results: list[TaskResult] = []
        try:
            values: list[Any] = group_result.get(timeout=timeout_s)
            for async_r, value in zip(group_result.results, values, strict=True):
                results.append(
                    TaskResult(
                        task_id=async_r.id,
                        status=TaskStatus.SUCCESS,
                        result=value,
                    )
                )
        except Exception as exc:
            # At least one task failed — collect per-task states
            for async_r in group_result.results:
                results.append(_map_task_result(async_r, async_r.id))
            # If the list is empty (group itself failed), surface the error
            if not results:
                results.append(
                    TaskResult(
                        task_id="group",
                        status=TaskStatus.FAILURE,
                        error=str(exc),
                    )
                )
        return results

    def __repr__(self) -> str:  # pragma: no cover
        cfg = self._cfg
        return (
            f"RactoCeleryWorker("
            f"kit={type(self._kit).__name__}, "
            f"rag={type(self._rag).__name__ if self._rag else None}, "
            f"max_retries={cfg.max_retries}, "
            f"backoff={cfg.initial_delay_s}sx{cfg.backoff_factor})"
        )
