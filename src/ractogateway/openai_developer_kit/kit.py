"""OpenAI Developer Kit — production-grade OpenAI interface.

Usage::

    from ractogateway import openai_developer_kit as opd

    kit = opd.OpenAIDeveloperKit(model="gpt-4o", default_prompt=my_prompt)
    response = kit.chat(opd.ChatConfig(user_message="Hello"))

    for chunk in kit.stream(opd.ChatConfig(user_message="Hello")):
        print(chunk.delta.text, end="", flush=True)
"""

from __future__ import annotations

import json as _json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ractogateway._models.chat import ChatConfig
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult, try_parse_json
from ractogateway.adapters.openai_kit import OpenAILLMKit
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt


def _require_openai() -> Any:
    try:
        import openai
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for OpenAIDeveloperKit. "
            "Install it with:  pip install ractogateway[openai]"
        ) from exc
    return openai


class OpenAIDeveloperKit:
    """Complete OpenAI developer kit — chat, stream, and embeddings.

    Parameters
    ----------
    model:
        Chat model (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Custom base URL (Azure OpenAI or proxy).
    embedding_model:
        Default embedding model.  Defaults to ``"text-embedding-3-small"``.
    default_prompt:
        RACTO prompt used when ``ChatConfig.prompt`` is ``None``.
    """

    provider: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        default_prompt: RactoPrompt | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._embedding_model = embedding_model
        self._default_prompt = default_prompt
        self._adapter = OpenAILLMKit(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

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

    # ------------------------------------------------------------------
    # Prompt resolution
    # ------------------------------------------------------------------

    def _resolve_prompt(self, config: ChatConfig) -> RactoPrompt:
        prompt = config.prompt or self._default_prompt
        if prompt is None:
            raise ValueError(
                "No prompt in ChatConfig and no default_prompt on the kit. Set one of them."
            )
        return prompt

    # ------------------------------------------------------------------
    # Chat  (sync / async)
    # ------------------------------------------------------------------

    def chat(self, config: ChatConfig) -> LLMResponse:
        """Synchronous chat completion."""
        prompt = self._resolve_prompt(config)
        response = self._adapter.run(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        return _maybe_validate(response, config)

    async def achat(self, config: ChatConfig) -> LLMResponse:
        """Async chat completion."""
        prompt = self._resolve_prompt(config)
        response = await self._adapter.arun(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        return _maybe_validate(response, config)

    # ------------------------------------------------------------------
    # Stream  (sync / async)
    # ------------------------------------------------------------------

    def stream(self, config: ChatConfig) -> Iterator[StreamChunk]:
        """Synchronous streaming — yields ``StreamChunk`` objects.

        Example::

            for chunk in kit.stream(config):
                print(chunk.delta.text, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nTokens: {chunk.usage}")
        """
        prompt = self._resolve_prompt(config)
        client = self._sync_client()
        request = self._adapter._build_request(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}

        accumulated = ""
        tc_acc: dict[int, dict[str, Any]] = {}

        try:
            with client.chat.completions.create(**request) as stream_resp:
                for event in stream_resp:
                    chunk = self._process_openai_event(
                        event,
                        accumulated,
                        tc_acc,
                    )
                    if chunk is not None:
                        accumulated = chunk.accumulated_text
                        if chunk.is_final and config.response_model is not None:
                            _apply_stream_response_model(chunk, config)
                        yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming — yields ``StreamChunk`` objects."""
        prompt = self._resolve_prompt(config)
        client = self._async_client()
        request = self._adapter._build_request(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}

        accumulated = ""
        tc_acc: dict[int, dict[str, Any]] = {}

        try:
            async with await client.chat.completions.create(**request) as stream_resp:
                async for event in stream_resp:
                    chunk = self._process_openai_event(
                        event,
                        accumulated,
                        tc_acc,
                    )
                    if chunk is not None:
                        accumulated = chunk.accumulated_text
                        if chunk.is_final and config.response_model is not None:
                            _apply_stream_response_model(chunk, config)
                        yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

    # ------------------------------------------------------------------
    # Embeddings  (sync / async)
    # ------------------------------------------------------------------

    def embed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Synchronous embedding."""
        client = self._sync_client()
        return self._do_embed(client, config)

    async def aembed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Async embedding."""
        client = self._async_client()
        return await self._do_aembed(client, config)

    # ------------------------------------------------------------------
    # Internal — OpenAI stream event processing
    # ------------------------------------------------------------------

    def _process_openai_event(
        self,
        event: Any,
        accumulated: str,
        tc_acc: dict[int, dict[str, Any]],
    ) -> StreamChunk | None:
        """Process one OpenAI streaming event into a ``StreamChunk``."""
        # Usage-only final event (no choices)
        if not event.choices:
            usage: dict[str, int] = {}
            if event.usage:
                usage = {
                    "prompt_tokens": event.usage.prompt_tokens,
                    "completion_tokens": event.usage.completion_tokens,
                    "total_tokens": event.usage.total_tokens,
                }
            return StreamChunk(
                accumulated_text=accumulated,
                finish_reason=FinishReason.STOP,
                tool_calls=_flush_tool_calls(tc_acc),
                usage=usage,
                is_final=True,
                raw=event,
            )

        choice = event.choices[0]
        delta = choice.delta
        text = delta.content or ""
        accumulated += text

        sd = StreamDelta(text=text)

        # Accumulate tool-call fragments
        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tc_acc:
                    tc_acc[idx] = {"id": tc.id or "", "name": "", "args": ""}
                if tc.function:
                    if tc.function.name:
                        tc_acc[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tc_acc[idx]["args"] += tc.function.arguments
                sd = StreamDelta(
                    text=text,
                    tool_call_id=tc_acc[idx]["id"],
                    tool_call_name=tc_acc[idx]["name"],
                    tool_call_args_fragment=(tc.function.arguments if tc.function else None),
                )

        if choice.finish_reason is not None:
            finish = OpenAILLMKit._map_finish_reason(choice.finish_reason)
            return StreamChunk(
                delta=sd,
                accumulated_text=accumulated,
                finish_reason=finish,
                tool_calls=_flush_tool_calls(tc_acc),
                is_final=True,
                raw=event,
            )

        return StreamChunk(
            delta=sd,
            accumulated_text=accumulated,
            raw=event,
        )

    # ------------------------------------------------------------------
    # Internal — embeddings
    # ------------------------------------------------------------------

    def _do_embed(self, client: Any, config: EmbeddingConfig) -> EmbeddingResponse:
        model = config.model or self._embedding_model
        kw: dict[str, Any] = {}
        if config.dimensions is not None:
            kw["dimensions"] = config.dimensions
        kw.update(config.extra)
        raw = client.embeddings.create(input=config.texts, model=model, **kw)
        return _normalise_openai_embedding(raw, config.texts, model)

    async def _do_aembed(
        self,
        client: Any,
        config: EmbeddingConfig,
    ) -> EmbeddingResponse:
        model = config.model or self._embedding_model
        kw: dict[str, Any] = {}
        if config.dimensions is not None:
            kw["dimensions"] = config.dimensions
        kw.update(config.extra)
        raw = await client.embeddings.create(input=config.texts, model=model, **kw)
        return _normalise_openai_embedding(raw, config.texts, model)


# ======================================================================
# Module-level helpers (shared, no state)
# ======================================================================


def _flush_tool_calls(acc: dict[int, dict[str, Any]]) -> list[ToolCallResult]:
    results: list[ToolCallResult] = []
    for entry in acc.values():
        try:
            args = _json.loads(entry["args"]) if entry["args"] else {}
        except _json.JSONDecodeError:
            args = {"_raw": entry["args"]}
        results.append(
            ToolCallResult(id=entry["id"], name=entry["name"], arguments=args),
        )
    return results


def _normalise_openai_embedding(
    raw: Any,
    texts: list[str],
    model: str,
) -> EmbeddingResponse:
    vectors = [
        EmbeddingVector(
            index=item.index,
            text=texts[item.index],
            embedding=item.embedding,
        )
        for item in raw.data
    ]
    usage: dict[str, int] = {}
    if raw.usage:
        usage = {
            "prompt_tokens": raw.usage.prompt_tokens,
            "total_tokens": raw.usage.total_tokens,
        }
    return EmbeddingResponse(vectors=vectors, model=model, usage=usage, raw=raw)


def _maybe_validate(response: LLMResponse, config: ChatConfig) -> LLMResponse:
    if config.response_model is not None and isinstance(response.parsed, dict):
        try:
            validated = config.response_model.model_validate(response.parsed)
            response.parsed = validated.model_dump()
        except Exception as exc:
            warning = f"[RactoGateway] response_model validation failed: {exc}"
            response.content = f"{response.content}\n\n{warning}" if response.content else warning
    return response


def _apply_stream_response_model(chunk: StreamChunk, config: ChatConfig) -> None:
    """Parse and validate the final chunk's ``accumulated_text`` against ``config.response_model``.

    Mutates *chunk* in-place: sets ``chunk.parsed`` to the validated model
    dump on success, or to the raw parsed dict when validation fails (so the
    caller can still inspect the data).  No exception is raised on failure.
    """
    if config.response_model is None:
        return
    parsed = try_parse_json(chunk.accumulated_text)
    if isinstance(parsed, dict):
        try:
            validated = config.response_model.model_validate(parsed)
            chunk.parsed = validated.model_dump()
        except Exception:
            chunk.parsed = parsed  # best-effort: store raw parsed dict
    elif parsed is not None:
        chunk.parsed = parsed
