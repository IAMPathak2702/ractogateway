"""OpenAI Batch API processor.

Submits large sets of chat-completion requests to OpenAI's asynchronous
Batch API, which processes them within 24 hours at **~50 % cost** compared
to the synchronous Chat Completions API.

Workflow::

    upload JSONL file  →  create batch job  →  poll status
    →  download results  →  parse into BatchResult list

Both synchronous and async variants are provided for every operation.

Usage::

    from ractogateway import openai_developer_kit as gpt
    from ractogateway.prompts.engine import RactoPrompt

    prompt = RactoPrompt(role="assistant", aim="answer briefly",
                         constraints="", tone="", output_format="text")

    processor = gpt.OpenAIBatchProcessor(model="gpt-4o-mini", default_prompt=prompt)

    results = processor.submit_and_wait([
        gpt.BatchItem(custom_id="q1", user_message="What is 2+2?"),
        gpt.BatchItem(custom_id="q2", user_message="Capital of France?"),
    ])

    for r in results:
        print(r.custom_id, r.response.content if r.ok else r.error)
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Any

from ractogateway.adapters.base import LLMResponse
from ractogateway.batch._models import BatchItem, BatchJobInfo, BatchResult, BatchStatus
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt


def _require_openai() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for OpenAIBatchProcessor. "
            "Install it with:  pip install ractogateway[openai]"
        ) from exc
    return openai


_OPENAI_STATUS_MAP: dict[str, BatchStatus] = {
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.FINALIZING,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "expired": BatchStatus.EXPIRED,
    "cancelling": BatchStatus.CANCELLING,
    "cancelled": BatchStatus.CANCELLED,
}


_HTTP_ERROR_THRESHOLD: int = 400


def _map_batch_status(status: str) -> BatchStatus:
    return _OPENAI_STATUS_MAP.get(status, BatchStatus.IN_PROGRESS)


def _build_jsonl(
    items: list[BatchItem],
    prompt: RactoPrompt,
    model: str,
) -> bytes:
    """Serialise *items* into OpenAI batch JSONL format."""
    lines: list[str] = []
    system_prompt = prompt.compile()

    for item in items:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item.user_message},
            ],
            "temperature": item.temperature,
            "max_tokens": item.max_tokens,
        }
        body.update(item.extra)

        record: dict[str, Any] = {
            "custom_id": item.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        lines.append(json.dumps(record))

    return "\n".join(lines).encode()


def _parse_result(line_data: dict[str, Any]) -> BatchResult:
    """Convert one output JSONL record into a :class:`BatchResult`."""
    custom_id: str = line_data.get("custom_id", "")
    error_obj = line_data.get("error")

    if error_obj:
        err_msg = (
            str(error_obj.get("message", error_obj))
            if isinstance(error_obj, dict)
            else str(error_obj)
        )
        return BatchResult(custom_id=custom_id, error=err_msg, raw=line_data)

    response_data = line_data.get("response", {})
    body = response_data.get("body", {}) if isinstance(response_data, dict) else {}

    if not body or response_data.get("status_code", 200) >= _HTTP_ERROR_THRESHOLD:
        status_code = response_data.get("status_code") if isinstance(response_data, dict) else None
        return BatchResult(
            custom_id=custom_id,
            error=f"Request failed with status {status_code}",
            raw=line_data,
        )

    # Parse the chat completion response body
    choices = body.get("choices", [])
    if not choices:
        return BatchResult(custom_id=custom_id, error="Empty choices in response", raw=line_data)

    choice = choices[0]
    msg = choice.get("message", {})
    content = msg.get("content")
    usage_raw = body.get("usage", {})
    usage: dict[str, int] = {
        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
        "completion_tokens": usage_raw.get("completion_tokens", 0),
        "total_tokens": usage_raw.get("total_tokens", 0),
    }

    from ractogateway.adapters.base import FinishReason, strip_markdown_fences, try_parse_json

    finish_map = {
        "stop": FinishReason.STOP,
        "tool_calls": FinishReason.TOOL_CALL,
        "length": FinishReason.LENGTH,
    }
    finish = finish_map.get(choice.get("finish_reason", "stop"), FinishReason.STOP)

    cleaned = strip_markdown_fences(content) if content else None
    parsed = try_parse_json(cleaned) if cleaned else None

    llm_response = LLMResponse(
        content=cleaned,
        parsed=parsed,
        finish_reason=finish,
        usage=usage,
        raw=line_data,
    )
    return BatchResult(custom_id=custom_id, response=llm_response, raw=line_data)


class OpenAIBatchProcessor:
    """Submit thousands of chat-completion requests to OpenAI's Batch API at
    ~50 % of standard API cost.

    Parameters
    ----------
    model:
        Chat model to use for all items in a batch (e.g. ``"gpt-4o-mini"``).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Custom base URL (Azure OpenAI / proxy).
    default_prompt:
        RACTO prompt used as the system message for every batch item.

    Methods
    -------
    submit_batch / asubmit_batch:
        Upload JSONL and create batch job → returns :class:`BatchJobInfo`.
    poll_status / apoll_status:
        Fetch current job state → returns updated :class:`BatchJobInfo`.
    get_results / aget_results:
        Download and parse completed job results → ``list[BatchResult]``.
    submit_and_wait / asubmit_and_wait:
        Convenience: submit + poll until done + return results.
    """

    provider: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_prompt: RactoPrompt | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._default_prompt = default_prompt

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    def _sync_client(self) -> Any:
        openai = _require_openai()
        kw: dict[str, Any] = {}
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            kw["api_key"] = key
        if self._base_url:
            kw["base_url"] = self._base_url
        return openai.OpenAI(**kw)

    def _async_client(self) -> Any:
        openai = _require_openai()
        kw: dict[str, Any] = {}
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            kw["api_key"] = key
        if self._base_url:
            kw["base_url"] = self._base_url
        return openai.AsyncOpenAI(**kw)

    def _resolve_prompt(self, prompt: RactoPrompt | None) -> RactoPrompt:
        p = prompt or self._default_prompt
        if p is None:
            raise ValueError("No prompt provided and no default_prompt on the processor.")
        return p

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def submit_batch(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
        completion_window: str = "24h",
    ) -> BatchJobInfo:
        """Upload *items* as a JSONL file and create an OpenAI batch job.

        Returns immediately with a :class:`BatchJobInfo` (status = IN_PROGRESS).
        """
        resolved_prompt = self._resolve_prompt(prompt)
        jsonl_bytes = _build_jsonl(items, resolved_prompt, self._model)
        client = self._sync_client()

        try:
            file_obj = client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            batch = client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
            )
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        return BatchJobInfo(
            job_id=batch.id,
            provider="openai",
            status=_map_batch_status(batch.status),
            created_at=float(batch.created_at),
            request_count=len(items),
            raw=batch,
        )

    def poll_status(self, job_id: str) -> BatchJobInfo:
        """Fetch the current status of batch job *job_id*."""
        client = self._sync_client()
        try:
            batch = client.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        return BatchJobInfo(
            job_id=batch.id,
            provider="openai",
            status=_map_batch_status(batch.status),
            created_at=float(batch.created_at),
            request_count=batch.request_counts.total if batch.request_counts else 0,
            raw=batch,
        )

    def get_results(self, job_id: str) -> list[BatchResult]:
        """Download and parse results for a completed batch job.

        Raises
        ------
        RuntimeError
            If the job is not yet completed.
        """
        client = self._sync_client()
        try:
            batch = client.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {job_id!r} is not completed yet (status={batch.status!r}). "
                "Call poll_status() first."
            )

        if not batch.output_file_id:
            raise RuntimeError(f"Batch {job_id!r} has no output file.")

        try:
            content = client.files.content(batch.output_file_id)
            raw_text = content.text
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        results: list[BatchResult] = []
        for line in raw_text.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                results.append(_parse_result(data))
            except json.JSONDecodeError as exc:
                results.append(
                    BatchResult(custom_id="<parse_error>", error=f"JSON decode error: {exc}")
                )
        return results

    def submit_and_wait(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
        completion_window: str = "24h",
        poll_interval_s: float = 60.0,
        max_wait_s: float = 86_400.0,
    ) -> list[BatchResult]:
        """Submit a batch and block until it completes, then return results.

        Parameters
        ----------
        poll_interval_s:
            Seconds between status-poll API calls.  Default ``60.0``.
        max_wait_s:
            Maximum total seconds to wait.  Default ``86400`` (24 h).

        Raises
        ------
        TimeoutError
            If the batch does not complete within *max_wait_s*.
        RuntimeError
            If the batch job fails or is cancelled.
        """
        info = self.submit_batch(items, prompt=prompt, completion_window=completion_window)
        deadline = time.monotonic() + max_wait_s

        while True:
            info = self.poll_status(info.job_id)
            if info.status in (BatchStatus.COMPLETED,):
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
        completion_window: str = "24h",
    ) -> BatchJobInfo:
        """Async variant of :meth:`submit_batch`."""

        resolved_prompt = self._resolve_prompt(prompt)
        jsonl_bytes = _build_jsonl(items, resolved_prompt, self._model)
        client = self._async_client()

        try:
            file_obj = await client.files.create(
                file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
                purpose="batch",
            )
            batch = await client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window=completion_window,
            )
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        return BatchJobInfo(
            job_id=batch.id,
            provider="openai",
            status=_map_batch_status(batch.status),
            created_at=float(batch.created_at),
            request_count=len(items),
            raw=batch,
        )

    async def apoll_status(self, job_id: str) -> BatchJobInfo:
        """Async variant of :meth:`poll_status`."""
        client = self._async_client()
        try:
            batch = await client.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        return BatchJobInfo(
            job_id=batch.id,
            provider="openai",
            status=_map_batch_status(batch.status),
            created_at=float(batch.created_at),
            request_count=batch.request_counts.total if batch.request_counts else 0,
            raw=batch,
        )

    async def aget_results(self, job_id: str) -> list[BatchResult]:
        """Async variant of :meth:`get_results`."""
        client = self._async_client()
        try:
            batch = await client.batches.retrieve(job_id)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        if batch.status != "completed":
            raise RuntimeError(f"Batch {job_id!r} is not completed yet (status={batch.status!r}).")
        if not batch.output_file_id:
            raise RuntimeError(f"Batch {job_id!r} has no output file.")

        try:
            content = await client.files.content(batch.output_file_id)
            raw_text = content.text
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

        results: list[BatchResult] = []
        for line in raw_text.strip().splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                results.append(_parse_result(data))
            except json.JSONDecodeError as exc:
                results.append(
                    BatchResult(custom_id="<parse_error>", error=f"JSON decode error: {exc}")
                )
        return results

    async def asubmit_and_wait(
        self,
        items: list[BatchItem],
        *,
        prompt: RactoPrompt | None = None,
        completion_window: str = "24h",
        poll_interval_s: float = 60.0,
        max_wait_s: float = 86_400.0,
    ) -> list[BatchResult]:
        """Async variant of :meth:`submit_and_wait`."""
        import asyncio

        info = await self.asubmit_batch(items, prompt=prompt, completion_window=completion_window)
        deadline = time.monotonic() + max_wait_s

        while True:
            info = await self.apoll_status(info.job_id)
            if info.status in (BatchStatus.COMPLETED,):
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
