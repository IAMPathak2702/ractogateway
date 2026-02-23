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
from typing import TYPE_CHECKING, Any

from ractogateway._models.chat import ChatConfig
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway._validation import (
    async_validate_and_retry,
    validate_and_retry,
    validate_stream_final,
)
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.adapters.openai_kit import OpenAILLMKit
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt

if TYPE_CHECKING:
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.routing.router import CostAwareRouter
    from ractogateway.truncation.truncator import TokenTruncator


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
    """Complete OpenAI developer kit — chat, stream, embeddings, and
    optional performance/cost optimisation middleware.

    Parameters
    ----------
    model:
        Chat model (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
        Use ``"auto"`` when a :class:`~ractogateway.routing.CostAwareRouter`
        is provided — the router will select the model per-request.
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Custom base URL (Azure OpenAI or proxy).
    embedding_model:
        Default embedding model.  Defaults to ``"text-embedding-3-small"``.
    default_prompt:
        RACTO prompt used when ``ChatConfig.prompt`` is ``None``.
    exact_cache:
        Optional :class:`~ractogateway.cache.ExactMatchCache`.  Serves
        byte-identical requests from memory at zero cost.
    semantic_cache:
        Optional :class:`~ractogateway.cache.SemanticCache`.  Returns cached
        answers for semantically similar queries (similarity ≥ threshold).
    router:
        Optional :class:`~ractogateway.routing.CostAwareRouter`.  Selects
        the cheapest model that can handle each request's complexity.
        **Required** when ``model="auto"``.
    truncator:
        Optional :class:`~ractogateway.truncation.TokenTruncator`.
        Automatically trims conversation history to fit the model's context
        window before each API call.
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
        exact_cache: ExactMatchCache | None = None,
        semantic_cache: SemanticCache | None = None,
        router: CostAwareRouter | None = None,
        truncator: TokenTruncator | None = None,
    ) -> None:
        if model == "auto" and router is None:
            raise ValueError(
                "model='auto' requires a CostAwareRouter.  "
                "Pass router=CostAwareRouter([...]) to the kit."
            )
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._embedding_model = embedding_model
        self._default_prompt = default_prompt
        self._exact_cache = exact_cache
        self._semantic_cache = semantic_cache
        self._router = router
        self._truncator = truncator
        # Adapter pool: reuse per-model adapters (O(1) lookup after first use)
        self._adapters: dict[str, OpenAILLMKit] = {}
        # Warm-up the default adapter (skip for "auto" — router decides model)
        if model != "auto":
            self._adapter = self._get_adapter(model)
        else:
            self._adapter = self._get_adapter("gpt-4o")  # placeholder, never used directly

    # ------------------------------------------------------------------
    # Adapter pool
    # ------------------------------------------------------------------

    def _get_adapter(self, model: str) -> OpenAILLMKit:
        """Return (or lazily create) an adapter for *model*."""
        if model not in self._adapters:
            self._adapters[model] = OpenAILLMKit(
                model=model,
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._adapters[model]

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
        if not isinstance(config, ChatConfig):
            raise TypeError(
                f"chat() expects a ChatConfig object, got {type(config).__name__!r}. "
                "Example: kit.chat(ChatConfig(user_message='Hello'))"
            )
        prompt = config.prompt or self._default_prompt
        if prompt is None:
            raise ValueError(
                "No prompt in ChatConfig and no default_prompt on the kit. Set one of them."
            )
        return prompt

    # ------------------------------------------------------------------
    # Middleware helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, user_message: str) -> str:
        """Return the effective model for *user_message* (router or default)."""
        if self._router is not None:
            return self._router.route(user_message)
        return self._model

    def _apply_truncation(self, config: ChatConfig, model: str) -> ChatConfig:
        """Return a (possibly trimmed) copy of *config* if truncator is set."""
        if self._truncator is None:
            return config
        return self._truncator.truncate(config, model)

    # ------------------------------------------------------------------
    # Chat  (sync / async)
    # ------------------------------------------------------------------

    def chat(self, config: ChatConfig) -> LLMResponse:
        """Synchronous chat completion with optional middleware pipeline.

        Middleware order: truncate → exact cache → semantic cache →
        route model → API call → write caches.
        """
        prompt = self._resolve_prompt(config)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        system_prompt = prompt.compile()

        # Exact-match cache lookup
        if self._exact_cache is not None:
            cached = self._exact_cache.get(
                config.user_message,
                system_prompt,
                model,
                config.temperature,
                config.max_tokens,
            )
            if cached is not None:
                return cached

        # Semantic cache lookup
        if self._semantic_cache is not None:
            sem_cached = self._semantic_cache.get(config.user_message)
            if sem_cached is not None:
                return sem_cached

        # API call
        adapter = self._get_adapter(model)
        response = adapter.run(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        response = validate_and_retry(
            response,
            config,
            adapter_run=lambda msg: adapter.run(
                prompt,
                msg,
                tools=config.tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.extra,
            ),
        )

        # Write to caches
        if self._exact_cache is not None:
            self._exact_cache.put(
                config.user_message,
                system_prompt,
                model,
                config.temperature,
                config.max_tokens,
                response,
            )
        if self._semantic_cache is not None:
            self._semantic_cache.put(config.user_message, response)

        return response

    async def achat(self, config: ChatConfig) -> LLMResponse:
        """Async chat completion with optional middleware pipeline."""
        prompt = self._resolve_prompt(config)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        system_prompt = prompt.compile()

        # Exact-match cache lookup
        if self._exact_cache is not None:
            cached = self._exact_cache.get(
                config.user_message,
                system_prompt,
                model,
                config.temperature,
                config.max_tokens,
            )
            if cached is not None:
                return cached

        # Semantic cache lookup
        if self._semantic_cache is not None:
            sem_cached = self._semantic_cache.get(config.user_message)
            if sem_cached is not None:
                return sem_cached

        # API call
        adapter = self._get_adapter(model)
        response = await adapter.arun(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        response = await async_validate_and_retry(
            response,
            config,
            adapter_arun=lambda msg: adapter.arun(
                prompt,
                msg,
                tools=config.tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                **config.extra,
            ),
        )

        # Write to caches
        if self._exact_cache is not None:
            self._exact_cache.put(
                config.user_message,
                system_prompt,
                model,
                config.temperature,
                config.max_tokens,
                response,
            )
        if self._semantic_cache is not None:
            self._semantic_cache.put(config.user_message, response)

        return response

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
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        adapter = self._get_adapter(model)
        client = self._sync_client()
        request = adapter._build_request(
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
                            chunk.parsed = validate_stream_final(
                                chunk.accumulated_text, config
                            )
                        yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming — yields ``StreamChunk`` objects."""
        prompt = self._resolve_prompt(config)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        adapter = self._get_adapter(model)
        client = self._async_client()
        request = adapter._build_request(
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
                            chunk.parsed = validate_stream_final(
                                chunk.accumulated_text, config
                            )
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


