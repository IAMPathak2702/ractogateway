"""Anthropic Message Batches API processor.

Submits large sets of Claude messages using Anthropic's asynchronous Batch
API, which processes them at **~50 % of standard API cost**.

Workflow::

    create batch job  →  poll until ended_at is set
    →  stream results  →  parse into BatchResult list

Both synchronous and async variants are provided for every operation.

Usage::

    from ractogateway import anthropic_developer_kit as claude
    from ractogateway.prompts.engine import RactoPrompt

    prompt = RactoPrompt(role="assistant", aim="answer briefly",
                         constraints="", tone="", output_format="text")

    processor = claude.AnthropicBatchProcessor(
        model="claude-haiku-4-5-20251001",
        default_prompt=prompt,
    )

    results = processor.submit_and_wait([
        claude.BatchItem(custom_id="q1", user_message="What is 2+2?"),
        claude.BatchItem(custom_id="q2", user_message="Capital of France?"),
    ])

    for r in results:
        print(r.custom_id, r.response.content if r.ok else r.error)
"""

from __future__ import annotations

import os
import time
from typing import Any

from ractogateway.adapters.base import (
    FinishReason,
    LLMResponse,
    strip_markdown_fences,
    try_parse_json,
)
from ractogateway.batch._models import BatchItem, BatchJobInfo, BatchResult, BatchStatus
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt


def _require_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for AnthropicBatchProcessor. "
            "Install it with:  pip install ractogateway[anthropic]"
        ) from exc
    return anthropic


def _ts(dt: Any) -> float:
    """Extract a Unix float timestamp from a datetime or return now()."""
    return dt.timestamp() if hasattr(dt, "timestamp") else float(time.time())


_ANTHROPIC_STATUS_MAP: dict[str, BatchStatus] = {
    "in_progress": BatchStatus.IN_PROGRESS,
    "canceling": BatchStatus.CANCELLING,
    "ended": BatchStatus.COMPLETED,
}


def _map_batch_status(status: str) -> BatchStatus:
    return _ANTHROPIC_STATUS_MAP.get(status, BatchStatus.IN_PROGRESS)


def _build_requests(
    items: list[BatchItem],
    prompt: RactoPrompt,
    model: str,
) -> list[dict[str, Any]]:
    """Build Anthropic MessageBatchRequestParam dicts from batch items."""
    system_prompt = prompt.compile()
    requests: list[dict[str, Any]] = []
    for item in items:
        params: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": item.user_message}],
            "max_tokens": item.max_tokens,
            "temperature": item.temperature,
        }
        params.update(item.extra)
        requests.append(
            {
                "custom_id": item.custom_id,
                "params": params,
            }
        )
    return requests


def _parse_anthropic_result(result: Any) -> BatchResult:
    """Convert one Anthropic MessageBatchResult into a :class:`BatchResult`."""
    custom_id: str = result.custom_id

    result_type = result.result.type if hasattr(result, "result") else "error"

    if result_type == "errored":
        err = result.result.error if hasattr(result.result, "error") else result.result
        return BatchResult(custom_id=custom_id, error=str(err), raw=result)

    if result_type != "succeeded":
        return BatchResult(
            custom_id=custom_id,
            error=f"Unexpected result type: {result_type!r}",
            raw=result,
        )

    message = result.result.message

    # Extract text content
    text_parts: list[str] = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)

    content = "\n".join(text_parts) if text_parts else None
    cleaned = strip_markdown_fences(content) if content else None
    parsed = try_parse_json(cleaned) if cleaned else None

    finish_map = {
        "end_turn": FinishReason.STOP,
        "tool_use": FinishReason.TOOL_CALL,
        "max_tokens": FinishReason.LENGTH,
    }
    finish = finish_map.get(getattr(message, "stop_reason", "end_turn"), FinishReason.STOP)

    usage: dict[str, int] = {}
    if hasattr(message, "usage") and message.usage:
        inp = getattr(message.usage, "input_tokens", 0) or 0
        out = getattr(message.usage, "output_tokens", 0) or 0
        usage = {
            "prompt_tokens": inp,
            "completion_tokens": out,
            "total_tokens": inp + out,
        }

    llm_response = LLMResponse(
        content=cleaned,
        parsed=parsed,
        finish_reason=finish,
        usage=usage,
        raw=result,
    )
    return BatchResult(custom_id=custom_id, response=llm_response, raw=result)


class AnthropicBatchProcessor:
    """Submit thousands of Claude requests to Anthropic's Message Batches API
    at ~50 % of standard API cost.

    Parameters
    ----------
    model:
        Claude model for all items (e.g. ``"claude-haiku-4-5-20251001"``).
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    default_prompt:
        RACTO prompt used as the ``system`` message for every batch item.

    Methods
    -------
    submit_batch / asubmit_batch:
        Create a batch job → returns :class:`BatchJobInfo`.
    poll_status / apoll_status:
        Fetch current job state → returns updated :class:`BatchJobInfo`.
    get_results / aget_results:
        Stream and parse completed job results → ``list[BatchResult]``.
    submit_and_wait / asubmit_and_wait:
        Convenience: submit + poll until done + return results.
    """

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        *,
        api_key: str | None = None,
        default_prompt: RactoPrompt | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._default_prompt = default_prompt

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    def _sync_client(self) -> Any:
        anthropic = _require_anthropic()
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        kw: dict[str, Any] = {"api_key": key} if key else {}
        return anthropic.Anthropic(**kw)

    def _async_client(self) -> Any:
        anthropic = _require_anthropic()
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        kw: dict[str, Any] = {"api_key": key} if key else {}
        return anthropic.AsyncAnthropic(**kw)

    def _resolve_prompt(self, prompt: RactoPrompt | None) -> RactoPrompt:
        p = prompt or self._default_prompt
        if p is None:
            raise ValueError(
                "No prompt provided and no default_prompt on the processor."
            )
        return p

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
    ) -> BatchJobInfo:
        """Create an Anthropic Message Batch job.

        Returns immediately with a :class:`BatchJobInfo`
        (status = IN_PROGRESS).
        """
        resolved_prompt = self._resolve_prompt(prompt)
        requests = _build_requests(items, resolved_prompt, self._model)
        client = self._sync_client()

        try:
            batch = client.beta.messages.batches.create(requests=requests)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

        created_ts = _ts(batch.created_at)
        return BatchJobInfo(
            job_id=batch.id,
            provider="anthropic",
            status=_map_batch_status(batch.processing_status),
            created_at=created_ts,
            request_count=len(items),
            raw=batch,
        )

    def poll_status(self, job_id: str) -> BatchJobInfo:
        """Fetch the current status of batch job *job_id*."""
        client = self._sync_client()
        try:
            batch = client.beta.messages.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

        created_ts = _ts(batch.created_at)
        rc = batch.request_counts
        total = (rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired) if rc else 0

        return BatchJobInfo(
            job_id=batch.id,
            provider="anthropic",
            status=_map_batch_status(batch.processing_status),
            created_at=created_ts,
            request_count=total,
            raw=batch,
        )

    def get_results(self, job_id: str) -> list[BatchResult]:
        """Stream and parse results for a completed batch job.

        Raises
        ------
        RuntimeError
            If the job is not yet completed.
        """
        client = self._sync_client()
        try:
            batch = client.beta.messages.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

        if batch.processing_status != "ended":
            raise RuntimeError(
                f"Batch {job_id!r} is not completed yet "
                f"(processing_status={batch.processing_status!r})."
            )

        results: list[BatchResult] = []
        try:
            for result in client.beta.messages.batches.results(job_id):
                results.append(_parse_anthropic_result(result))
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc
        return results

    def submit_and_wait(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
        poll_interval_s: float = 60.0,
        max_wait_s: float = 86_400.0,
    ) -> list[BatchResult]:
        """Submit a batch and block until it completes, then return results.

        Parameters
        ----------
        poll_interval_s:
            Seconds between status-poll API calls.  Default ``60.0``.
        max_wait_s:
            Maximum total wait.  Default ``86400`` (24 h).
        """
        info = self.submit_batch(items, prompt=prompt)
        deadline = time.monotonic() + max_wait_s

        while True:
            info = self.poll_status(info.job_id)
            if info.status == BatchStatus.COMPLETED:
                return self.get_results(info.job_id)
            if info.status in (BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED):
                raise RuntimeError(
                    f"Batch {info.job_id!r} ended with status {info.status.value!r}."
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {info.job_id!r} did not complete within {max_wait_s:.0f} s."
                )
            time.sleep(poll_interval_s)

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def asubmit_batch(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
    ) -> BatchJobInfo:
        """Async variant of :meth:`submit_batch`."""
        resolved_prompt = self._resolve_prompt(prompt)
        requests = _build_requests(items, resolved_prompt, self._model)
        client = self._async_client()

        try:
            batch = await client.beta.messages.batches.create(requests=requests)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

        created_ts = _ts(batch.created_at)
        return BatchJobInfo(
            job_id=batch.id,
            provider="anthropic",
            status=_map_batch_status(batch.processing_status),
            created_at=created_ts,
            request_count=len(items),
            raw=batch,
        )

    async def apoll_status(self, job_id: str) -> BatchJobInfo:
        """Async variant of :meth:`poll_status`."""
        client = self._async_client()
        try:
            batch = await client.beta.messages.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

        created_ts = _ts(batch.created_at)
        rc = batch.request_counts
        total = (rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired) if rc else 0
        return BatchJobInfo(
            job_id=batch.id,
            provider="anthropic",
            status=_map_batch_status(batch.processing_status),
            created_at=created_ts,
            request_count=total,
            raw=batch,
        )

    async def aget_results(self, job_id: str) -> list[BatchResult]:
        """Async variant of :meth:`get_results`."""
        client = self._async_client()
        try:
            batch = await client.beta.messages.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

        if batch.processing_status != "ended":
            raise RuntimeError(
                f"Batch {job_id!r} is not completed yet "
                f"(processing_status={batch.processing_status!r})."
            )

        results: list[BatchResult] = []
        try:
            async for result in await client.beta.messages.batches.results(job_id):
                results.append(_parse_anthropic_result(result))
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc
        return results

    async def asubmit_and_wait(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
        poll_interval_s: float = 60.0,
        max_wait_s: float = 86_400.0,
    ) -> list[BatchResult]:
        """Async variant of :meth:`submit_and_wait`."""
        import asyncio

        info = await self.asubmit_batch(items, prompt=prompt)
        deadline = time.monotonic() + max_wait_s

        while True:
            info = await self.apoll_status(info.job_id)
            if info.status == BatchStatus.COMPLETED:
                return await self.aget_results(info.job_id)
            if info.status in (BatchStatus.FAILED, BatchStatus.CANCELLED, BatchStatus.EXPIRED):
                raise RuntimeError(
                    f"Batch {info.job_id!r} ended with status {info.status.value!r}."
                )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Batch {info.job_id!r} did not complete within {max_wait_s:.0f} s."
                )
            await asyncio.sleep(poll_interval_s)
