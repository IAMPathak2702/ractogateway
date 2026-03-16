"""Lightweight DAG execution inspired by the RactoFlow V8 spec."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TaskCallable = Callable[[dict[str, Any]], Any] | Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass
class TaskRun:
    """Execution record for one task."""

    task_id: str
    status: str
    started_at: datetime
    finished_at: datetime
    output: Any = None
    error: str | None = None


@dataclass
class DAGRun:
    """Execution record for one DAG run."""

    dag_id: str
    run_id: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    task_runs: dict[str, TaskRun] = field(default_factory=dict)


@dataclass
class DAGRunStore:
    """In-memory run tracker."""

    retain_runs: int = 100
    _runs: dict[str, list[DAGRun]] = field(default_factory=dict)

    def add(self, run: DAGRun) -> None:
        """Store a completed run and trim history."""
        bucket = self._runs.setdefault(run.dag_id, [])
        bucket.append(run)
        if len(bucket) > self.retain_runs:
            del bucket[0 : len(bucket) - self.retain_runs]

    def list_runs(self, dag_id: str) -> list[DAGRun]:
        """Return retained runs for one DAG."""
        return list(self._runs.get(dag_id, []))


class BaseOperator:
    """Base DAG operator."""

    def __init__(self, *, task_id: str, func: TaskCallable) -> None:
        self.task_id = task_id
        self._func = func
        self.upstream: set[str] = set()
        self.downstream: set[str] = set()

    def __rshift__(self, other: BaseOperator) -> BaseOperator:
        self.downstream.add(other.task_id)
        other.upstream.add(self.task_id)
        return other

    def run(self, context: dict[str, Any]) -> Any:
        """Execute synchronously."""
        if inspect.iscoroutinefunction(self._func):
            return asyncio.run(self._func(context))
        return self._func(context)

    async def arun(self, context: dict[str, Any]) -> Any:
        """Execute asynchronously."""
        if inspect.iscoroutinefunction(self._func):
            async_func = self._func
            return await async_func(context)
        sync_func = self._func
        return await asyncio.to_thread(sync_func, context)


class PythonOperator(BaseOperator):
    """Simple function-backed operator."""

    def __init__(self, *, task_id: str, python_callable: TaskCallable) -> None:
        super().__init__(task_id=task_id, func=python_callable)


class RactoDAG:
    """Python-defined DAG with limited parallel execution."""

    def __init__(
        self,
        *,
        dag_id: str,
        max_parallel_tasks: int = 4,
        run_store: DAGRunStore | None = None,
    ) -> None:
        self.dag_id = dag_id
        self.max_parallel_tasks = max_parallel_tasks
        self.run_store = run_store
        self._tasks: dict[str, BaseOperator] = {}

    def __enter__(self) -> RactoDAG:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> Literal[False]:
        return False

    def add_task(self, operator: BaseOperator) -> BaseOperator:
        """Register an operator with the DAG."""
        self._tasks[operator.task_id] = operator
        return operator

    def run(self, *, initial_context: dict[str, Any] | None = None) -> DAGRun:
        """Run the DAG synchronously."""
        return asyncio.run(self.arun(initial_context=initial_context))

    async def arun(self, *, initial_context: dict[str, Any] | None = None) -> DAGRun:
        """Run the DAG asynchronously."""
        run = DAGRun(
            dag_id=self.dag_id,
            run_id=f"{self.dag_id}-{uuid.uuid4().hex[:8]}",
            status="RUNNING",
            started_at=datetime.now(tz=timezone.utc),
        )
        shared_context: dict[str, Any] = {"task_outputs": dict(initial_context or {})}

        if not self._tasks:
            run.status = "SUCCESS"
            run.finished_at = datetime.now(tz=timezone.utc)
            if self.run_store is not None:
                self.run_store.add(run)
            return run

        pending = set(self._tasks)
        completed: set[str] = set()
        running: dict[str, asyncio.Task[TaskRun]] = {}
        semaphore = asyncio.Semaphore(self.max_parallel_tasks)

        async def _execute(operator: BaseOperator) -> TaskRun:
            started_at = datetime.now(tz=timezone.utc)
            try:
                async with semaphore:
                    context = {"task_outputs": dict(shared_context["task_outputs"])}
                    output = await operator.arun(context)
                return TaskRun(
                    task_id=operator.task_id,
                    status="SUCCESS",
                    started_at=started_at,
                    finished_at=datetime.now(tz=timezone.utc),
                    output=output,
                )
            except Exception as exc:
                return TaskRun(
                    task_id=operator.task_id,
                    status="FAILED",
                    started_at=started_at,
                    finished_at=datetime.now(tz=timezone.utc),
                    error=str(exc),
                )

        while pending or running:
            ready_ids = [
                task_id
                for task_id in pending
                if self._tasks[task_id].upstream.issubset(completed) and task_id not in running
            ]
            for task_id in ready_ids:
                running[task_id] = asyncio.create_task(_execute(self._tasks[task_id]))
                pending.remove(task_id)

            if not running:
                raise ValueError(f"DAG '{self.dag_id}' has a cycle or blocked dependency.")

            done, _ = await asyncio.wait(
                set(running.values()),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished_task in done:
                task_run = await finished_task
                run.task_runs[task_run.task_id] = task_run
                running.pop(task_run.task_id, None)
                if task_run.status == "FAILED":
                    for remaining in running.values():
                        remaining.cancel()
                    run.status = "FAILED"
                    run.finished_at = datetime.now(tz=timezone.utc)
                    if self.run_store is not None:
                        self.run_store.add(run)
                    return run

                shared_context["task_outputs"][task_run.task_id] = task_run.output
                completed.add(task_run.task_id)

        run.status = "SUCCESS"
        run.finished_at = datetime.now(tz=timezone.utc)
        if self.run_store is not None:
            self.run_store.add(run)
        return run
