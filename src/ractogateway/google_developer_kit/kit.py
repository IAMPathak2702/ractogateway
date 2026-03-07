"""Google Gemini Developer Kit — production-grade Gemini interface.

Usage::

    from ractogateway import google_developer_kit as god

    kit = god.GoogleDeveloperKit(model="gemini-2.0-flash", default_prompt=my_prompt)
    response = kit.chat(god.ChatConfig(user_message="Hello"))

    for chunk in kit.stream(god.ChatConfig(user_message="Hello")):
        print(chunk.delta.text, end="", flush=True)
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ractogateway._models.chat import ChatConfig
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway._tool_runtime import (
    build_tool_followup_user_message,
    execute_tool_calls_async,
    execute_tool_calls_sync,
)
from ractogateway._validation import (
    async_validate_and_retry,
    validate_and_retry,
    validate_stream_final,
    with_inferred_response_model,
)
from ractogateway.adapters.base import ChatTurn, FinishReason, LLMResponse, ToolCallResult
from ractogateway.adapters.google_kit import GoogleLLMKit, build_google_contents
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt

if TYPE_CHECKING:
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.routing.router import CostAwareRouter
    from ractogateway.telemetry.metrics import GatewayMetricsMiddleware
    from ractogateway.telemetry.tracer import RactoTracer
    from ractogateway.truncation.truncator import TokenTruncator


def _require_genai() -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError(
            "The 'google-genai' package is required for GoogleDeveloperKit. "
            "Install it with:  pip install ractogateway[google]"
        ) from exc
    return genai


class GoogleDeveloperKit:
    """Complete Google Gemini developer kit — chat, stream, embeddings, and
    optional performance/cost optimisation middleware.

    Parameters
    ----------
    model:
        Gemini model (e.g. ``"gemini-2.0-flash"``, ``"gemini-2.5-pro"``).
        Use ``"auto"`` when a :class:`~ractogateway.routing.CostAwareRouter`
        is provided — the router will select the model per-request.
    api_key:
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
    embedding_model:
        Default embedding model.  Defaults to ``"text-embedding-004"``.
    default_prompt:
        RACTO prompt used when ``ChatConfig.prompt`` is ``None``.
    exact_cache:
        Optional :class:`~ractogateway.cache.ExactMatchCache`.
    semantic_cache:
        Optional :class:`~ractogateway.cache.SemanticCache`.
    router:
        Optional :class:`~ractogateway.routing.CostAwareRouter`.
        **Required** when ``model="auto"``.
    truncator:
        Optional :class:`~ractogateway.truncation.TokenTruncator`.
    tracer:
        Optional :class:`~ractogateway.telemetry.RactoTracer`.
        Emits OpenTelemetry spans for every chat, stream, and embed call.
        Requires ``pip install ractogateway[telemetry]``.
    metrics:
        Optional :class:`~ractogateway.telemetry.GatewayMetricsMiddleware`.
        Records Prometheus metrics (latency, tokens, cost, cache hit/miss).
        Requires ``pip install ractogateway[prometheus]``.
    """

    provider: str = "google"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        embedding_model: str = "text-embedding-004",
        default_prompt: RactoPrompt | None = None,
        exact_cache: ExactMatchCache | None = None,
        semantic_cache: SemanticCache | None = None,
        router: CostAwareRouter | None = None,
        truncator: TokenTruncator | None = None,
        tracer: RactoTracer | None = None,
        metrics: GatewayMetricsMiddleware | None = None,
    ) -> None:
        if model == "auto" and router is None:
            raise ValueError(
                "model='auto' requires a CostAwareRouter.  "
                "Pass router=CostAwareRouter([...]) to the kit."
            )
        self._model = model
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._default_prompt = default_prompt
        self._exact_cache = exact_cache
        self._semantic_cache = semantic_cache
        self._router = router
        self._truncator = truncator
        self._tracer = tracer
        self._metrics = metrics
        # Adapter pool for cost-aware routing
        self._adapters: dict[str, GoogleLLMKit] = {}
        self._adapter = self._get_adapter(model if model != "auto" else "gemini-2.0-flash")

    # ------------------------------------------------------------------
    # Adapter pool
    # ------------------------------------------------------------------

    def _get_adapter(self, model: str) -> GoogleLLMKit:
        """Return (or lazily create) an adapter for *model*."""
        if model not in self._adapters:
            self._adapters[model] = GoogleLLMKit(model=model, api_key=self._api_key)
        return self._adapters[model]

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        genai = _require_genai()
        key = self._api_key or os.environ.get("GEMINI_API_KEY")
        return genai.Client(api_key=key)

    def _resolve_prompt(self, config: ChatConfig) -> RactoPrompt:
        if not isinstance(config, ChatConfig):
            raise TypeError(
                f"chat() expects a ChatConfig object, got {type(config).__name__!r}. "
                "Example: kit.chat(ChatConfig(user_message='Hello'))"
            )
        prompt = config.prompt or self._default_prompt
        if prompt is None:
            return RactoPrompt(
                role="You are a helpful AI assistant.",
                aim="Answer the user's question accurately and helpfully.",
                constraints=["Be accurate, clear, and concise."],
                tone="Helpful and professional.",
                output_format="text",
            )
        return prompt

    # ------------------------------------------------------------------
    # Middleware helpers
    # ------------------------------------------------------------------

    def _resolve_model(self, user_message: str) -> str:
        if self._router is not None:
            return self._router.route(user_message)
        return self._model

    def _apply_truncation(self, config: ChatConfig, model: str) -> ChatConfig:
        if self._truncator is None:
            return config
        return self._truncator.truncate(config, model)

    # ------------------------------------------------------------------
    # Chat  (sync / async)
    # ------------------------------------------------------------------

    def chat(self, config: ChatConfig) -> LLMResponse:
        """Synchronous chat completion with optional middleware pipeline.

        Middleware order: truncate → exact cache → semantic cache →
        route model → API call → write caches → record telemetry.
        """
        t0 = time.perf_counter()
        prompt = self._resolve_prompt(config)
        if config.chain_of_thought:
            from ractogateway._cot import apply_chain_of_thought
            prompt = apply_chain_of_thought(prompt)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        validation_config = with_inferred_response_model(config, prompt)
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
                _lat = (time.perf_counter() - t0) * 1000
                if self._metrics is not None:
                    self._metrics.record_cache_hit("exact")
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider, model=model, latency_ms=_lat, cache_hit="exact"
                    )
                return cached

        # Semantic cache lookup
        if self._semantic_cache is not None:
            sem_cached = self._semantic_cache.get(config.user_message)
            if sem_cached is not None:
                _lat = (time.perf_counter() - t0) * 1000
                if self._metrics is not None:
                    self._metrics.record_cache_hit("semantic")
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider,
                        model=model,
                        latency_ms=_lat,
                        cache_hit="semantic",
                    )
                return sem_cached

        # Record cache misses for each checked cache
        if self._metrics is not None:
            if self._exact_cache is not None:
                self._metrics.record_cache_miss("exact")
            if self._semantic_cache is not None:
                self._metrics.record_cache_miss("semantic")

        # API call
        adapter = self._get_adapter(model)
        original_user_message = config.user_message
        history_turns: list[ChatTurn] | None = (
            [ChatTurn(role=m.role.value, content=m.content) for m in config.history]
            if config.history
            else None
        )

        def _run_validated(user_message: str) -> LLMResponse:
            raw = adapter.run(
                prompt,
                user_message,
                history=history_turns,
                tools=config.tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                attachments=config.attachments,
                native_thinking=config.native_thinking,
                thinking_budget=config.thinking_budget,
                **config.extra,
            )
            return validate_and_retry(
                raw,
                validation_config,
                adapter_run=lambda msg: adapter.run(
                    prompt,
                    msg,
                    history=history_turns,
                    tools=config.tools,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    **config.extra,
                ),
            )

        try:
            response = _run_validated(config.user_message)

            if config.auto_execute_tools and config.tools is not None:
                for _ in range(config.max_tool_turns):
                    if (
                        response.finish_reason is not FinishReason.TOOL_CALL
                        or not response.tool_calls
                    ):
                        break
                    results = execute_tool_calls_sync(response.tool_calls, config.tools)
                    follow_up_message = build_tool_followup_user_message(
                        original_user_message=original_user_message,
                        tool_calls=response.tool_calls,
                        results=results,
                    )
                    response = _run_validated(follow_up_message)

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

            # Record telemetry
            _lat = (time.perf_counter() - t0) * 1000
            _in = response.usage.get("prompt_tokens", 0) if response.usage else 0
            _out = response.usage.get("completion_tokens", 0) if response.usage else 0
            _tcs = response.tool_calls or []
            if self._tracer is not None:
                self._tracer.record_chat_span(
                    provider=self.provider,
                    model=model,
                    latency_ms=_lat,
                    input_tokens=_in,
                    output_tokens=_out,
                    tool_calls=len(_tcs),
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="chat",
                    status="ok",
                    latency_s=_lat / 1000,
                    input_tokens=_in,
                    output_tokens=_out,
                    tool_calls=_tcs,
                )
            return response

        except Exception as _exc:
            _lat = (time.perf_counter() - t0) * 1000
            _etype = type(_exc).__name__
            if self._tracer is not None:
                self._tracer.record_chat_span(
                    provider=self.provider,
                    model=model,
                    latency_ms=_lat,
                    status="error",
                    error_type=_etype,
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="chat",
                    status="error",
                    latency_s=_lat / 1000,
                )
            raise

    async def achat(self, config: ChatConfig) -> LLMResponse:
        """Async chat completion with optional middleware pipeline."""
        t0 = time.perf_counter()
        prompt = self._resolve_prompt(config)
        if config.chain_of_thought:
            from ractogateway._cot import apply_chain_of_thought
            prompt = apply_chain_of_thought(prompt)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        validation_config = with_inferred_response_model(config, prompt)
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
                _lat = (time.perf_counter() - t0) * 1000
                if self._metrics is not None:
                    self._metrics.record_cache_hit("exact")
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider, model=model, latency_ms=_lat, cache_hit="exact"
                    )
                return cached

        # Semantic cache lookup
        if self._semantic_cache is not None:
            sem_cached = self._semantic_cache.get(config.user_message)
            if sem_cached is not None:
                _lat = (time.perf_counter() - t0) * 1000
                if self._metrics is not None:
                    self._metrics.record_cache_hit("semantic")
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider,
                        model=model,
                        latency_ms=_lat,
                        cache_hit="semantic",
                    )
                return sem_cached

        if self._metrics is not None:
            if self._exact_cache is not None:
                self._metrics.record_cache_miss("exact")
            if self._semantic_cache is not None:
                self._metrics.record_cache_miss("semantic")

        # API call
        adapter = self._get_adapter(model)
        original_user_message = config.user_message
        history_turns: list[ChatTurn] | None = (
            [ChatTurn(role=m.role.value, content=m.content) for m in config.history]
            if config.history
            else None
        )

        async def _arun_validated(user_message: str) -> LLMResponse:
            raw = await adapter.arun(
                prompt,
                user_message,
                history=history_turns,
                tools=config.tools,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                attachments=config.attachments,
                native_thinking=config.native_thinking,
                thinking_budget=config.thinking_budget,
                **config.extra,
            )
            return await async_validate_and_retry(
                raw,
                validation_config,
                adapter_arun=lambda msg: adapter.arun(
                    prompt,
                    msg,
                    history=history_turns,
                    tools=config.tools,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    **config.extra,
                ),
            )

        try:
            response = await _arun_validated(config.user_message)

            if config.auto_execute_tools and config.tools is not None:
                for _ in range(config.max_tool_turns):
                    if (
                        response.finish_reason is not FinishReason.TOOL_CALL
                        or not response.tool_calls
                    ):
                        break
                    results = await execute_tool_calls_async(
                        response.tool_calls,
                        config.tools,
                    )
                    follow_up_message = build_tool_followup_user_message(
                        original_user_message=original_user_message,
                        tool_calls=response.tool_calls,
                        results=results,
                    )
                    response = await _arun_validated(follow_up_message)

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

            _lat = (time.perf_counter() - t0) * 1000
            _in = response.usage.get("prompt_tokens", 0) if response.usage else 0
            _out = response.usage.get("completion_tokens", 0) if response.usage else 0
            _tcs = response.tool_calls or []
            if self._tracer is not None:
                self._tracer.record_chat_span(
                    provider=self.provider,
                    model=model,
                    latency_ms=_lat,
                    input_tokens=_in,
                    output_tokens=_out,
                    tool_calls=len(_tcs),
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="chat",
                    status="ok",
                    latency_s=_lat / 1000,
                    input_tokens=_in,
                    output_tokens=_out,
                    tool_calls=_tcs,
                )
            return response

        except Exception as _exc:
            _lat = (time.perf_counter() - t0) * 1000
            _etype = type(_exc).__name__
            if self._tracer is not None:
                self._tracer.record_chat_span(
                    provider=self.provider,
                    model=model,
                    latency_ms=_lat,
                    status="error",
                    error_type=_etype,
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="chat",
                    status="error",
                    latency_s=_lat / 1000,
                )
            raise

    # ------------------------------------------------------------------
    # Stream  (sync / async)
    # ------------------------------------------------------------------

    def stream(self, config: ChatConfig) -> Iterator[StreamChunk]:
        """Synchronous streaming via ``generate_content_stream``.

        Example::

            for chunk in kit.stream(config):
                print(chunk.delta.text, end="", flush=True)
        """
        from google.genai import types

        t0 = time.perf_counter()
        prompt = self._resolve_prompt(config)
        if config.chain_of_thought:
            from ractogateway._cot import apply_chain_of_thought
            prompt = apply_chain_of_thought(prompt)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        validation_config = with_inferred_response_model(config, prompt)
        adapter = self._get_adapter(model)
        client = self._client()
        system_prompt = prompt.compile()
        gen_config = adapter._build_config(
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            native_thinking=config.native_thinking,
            thinking_budget=config.thinking_budget,
            **config.extra,
        )
        history_turns: list[ChatTurn] | None = (
            [ChatTurn(role=m.role.value, content=m.content) for m in config.history]
            if config.history
            else None
        )
        stream_contents = build_google_contents(
            history_turns, config.user_message, attachments=config.attachments
        )

        accumulated = ""
        accumulated_thinking = ""
        tool_calls: list[ToolCallResult] = []
        _span_recorded = False

        try:
            for event in client.models.generate_content_stream(
                model=model,
                contents=stream_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    **gen_config,
                ),
            ):
                chunk = self._process_gemini_event(
                    event,
                    accumulated,
                    accumulated_thinking,
                    tool_calls,
                )
                accumulated = chunk.accumulated_text
                accumulated_thinking = chunk.accumulated_thinking
                if chunk.is_final and validation_config.response_model is not None:
                    chunk.parsed = validate_stream_final(
                        chunk.accumulated_text,
                        validation_config,
                    )
                if chunk.is_final and not _span_recorded:
                    _span_recorded = True
                    _lat = (time.perf_counter() - t0) * 1000
                    _in = chunk.usage.get("prompt_tokens", 0) if chunk.usage else 0
                    _out = chunk.usage.get("completion_tokens", 0) if chunk.usage else 0
                    _tcs = chunk.tool_calls or []
                    if self._tracer is not None:
                        self._tracer.record_chat_span(
                            provider=self.provider,
                            model=model,
                            latency_ms=_lat,
                            input_tokens=_in,
                            output_tokens=_out,
                            tool_calls=len(_tcs),
                        )
                    if self._metrics is not None:
                        self._metrics.record_request(
                            provider=self.provider,
                            model=model,
                            operation="stream",
                            status="ok",
                            latency_s=_lat / 1000,
                            input_tokens=_in,
                            output_tokens=_out,
                            tool_calls=_tcs,
                        )
                yield chunk
        except RactoGatewayError:
            if not _span_recorded:
                _lat = (time.perf_counter() - t0) * 1000
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider,
                        model=model,
                        latency_ms=_lat,
                        status="error",
                        error_type="RactoGatewayError",
                    )
                if self._metrics is not None:
                    self._metrics.record_request(
                        provider=self.provider,
                        model=model,
                        operation="stream",
                        status="error",
                        latency_s=(time.perf_counter() - t0),
                    )
            raise
        except Exception as exc:
            if not _span_recorded:
                _lat = (time.perf_counter() - t0) * 1000
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider,
                        model=model,
                        latency_ms=_lat,
                        status="error",
                        error_type=type(exc).__name__,
                    )
                if self._metrics is not None:
                    self._metrics.record_request(
                        provider=self.provider,
                        model=model,
                        operation="stream",
                        status="error",
                        latency_s=(time.perf_counter() - t0),
                    )
            raise _wrap_provider_error(exc, "google") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming via ``aio.models.generate_content_stream``."""
        from google.genai import types

        t0 = time.perf_counter()
        prompt = self._resolve_prompt(config)
        if config.chain_of_thought:
            from ractogateway._cot import apply_chain_of_thought
            prompt = apply_chain_of_thought(prompt)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        validation_config = with_inferred_response_model(config, prompt)
        adapter = self._get_adapter(model)
        client = self._client()
        system_prompt = prompt.compile()
        gen_config = adapter._build_config(
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            native_thinking=config.native_thinking,
            thinking_budget=config.thinking_budget,
            **config.extra,
        )
        history_turns: list[ChatTurn] | None = (
            [ChatTurn(role=m.role.value, content=m.content) for m in config.history]
            if config.history
            else None
        )
        stream_contents = build_google_contents(
            history_turns, config.user_message, attachments=config.attachments
        )

        accumulated = ""
        accumulated_thinking = ""
        tool_calls: list[ToolCallResult] = []
        _span_recorded = False

        try:
            async for event in await client.aio.models.generate_content_stream(
                model=model,
                contents=stream_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    **gen_config,
                ),
            ):
                chunk = self._process_gemini_event(
                    event,
                    accumulated,
                    accumulated_thinking,
                    tool_calls,
                )
                accumulated = chunk.accumulated_text
                accumulated_thinking = chunk.accumulated_thinking
                if chunk.is_final and validation_config.response_model is not None:
                    chunk.parsed = validate_stream_final(
                        chunk.accumulated_text,
                        validation_config,
                    )
                if chunk.is_final and not _span_recorded:
                    _span_recorded = True
                    _lat = (time.perf_counter() - t0) * 1000
                    _in = chunk.usage.get("prompt_tokens", 0) if chunk.usage else 0
                    _out = chunk.usage.get("completion_tokens", 0) if chunk.usage else 0
                    _tcs = chunk.tool_calls or []
                    if self._tracer is not None:
                        self._tracer.record_chat_span(
                            provider=self.provider,
                            model=model,
                            latency_ms=_lat,
                            input_tokens=_in,
                            output_tokens=_out,
                            tool_calls=len(_tcs),
                        )
                    if self._metrics is not None:
                        self._metrics.record_request(
                            provider=self.provider,
                            model=model,
                            operation="stream",
                            status="ok",
                            latency_s=_lat / 1000,
                            input_tokens=_in,
                            output_tokens=_out,
                            tool_calls=_tcs,
                        )
                yield chunk
        except RactoGatewayError:
            if not _span_recorded:
                _lat = (time.perf_counter() - t0) * 1000
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider,
                        model=model,
                        latency_ms=_lat,
                        status="error",
                        error_type="RactoGatewayError",
                    )
                if self._metrics is not None:
                    self._metrics.record_request(
                        provider=self.provider,
                        model=model,
                        operation="stream",
                        status="error",
                        latency_s=(time.perf_counter() - t0),
                    )
            raise
        except Exception as exc:
            if not _span_recorded:
                _lat = (time.perf_counter() - t0) * 1000
                if self._tracer is not None:
                    self._tracer.record_chat_span(
                        provider=self.provider,
                        model=model,
                        latency_ms=_lat,
                        status="error",
                        error_type=type(exc).__name__,
                    )
                if self._metrics is not None:
                    self._metrics.record_request(
                        provider=self.provider,
                        model=model,
                        operation="stream",
                        status="error",
                        latency_s=(time.perf_counter() - t0),
                    )
            raise _wrap_provider_error(exc, "google") from exc

    # ------------------------------------------------------------------
    # Embeddings  (sync / async)
    # ------------------------------------------------------------------

    def embed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Synchronous embedding via ``embed_content``."""
        t0 = time.perf_counter()
        client = self._client()
        model = config.model or self._embedding_model
        try:
            vectors: list[EmbeddingVector] = []
            for i, text in enumerate(config.texts):
                raw = client.models.embed_content(model=model, contents=text)
                vectors.append(
                    EmbeddingVector(
                        index=i,
                        text=text,
                        embedding=raw.embeddings[0].values,
                    ),
                )
            result = EmbeddingResponse(vectors=vectors, model=model)
            _lat = (time.perf_counter() - t0) * 1000
            _in = result.usage.get("prompt_tokens", 0) if result.usage else 0
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider, model=model, latency_ms=_lat, input_tokens=_in
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="embed",
                    status="ok",
                    latency_s=_lat / 1000,
                    input_tokens=_in,
                )
            return result
        except Exception as _exc:
            _lat = (time.perf_counter() - t0) * 1000
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider,
                    model=model,
                    latency_ms=_lat,
                    status="error",
                    error_type=type(_exc).__name__,
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="embed",
                    status="error",
                    latency_s=_lat / 1000,
                )
            raise

    async def aembed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Async embedding via ``aio.models.embed_content``."""
        t0 = time.perf_counter()
        client = self._client()
        model = config.model or self._embedding_model
        try:
            vectors: list[EmbeddingVector] = []
            for i, text in enumerate(config.texts):
                raw = await client.aio.models.embed_content(
                    model=model,
                    contents=text,
                )
                vectors.append(
                    EmbeddingVector(
                        index=i,
                        text=text,
                        embedding=raw.embeddings[0].values,
                    ),
                )
            result = EmbeddingResponse(vectors=vectors, model=model)
            _lat = (time.perf_counter() - t0) * 1000
            _in = result.usage.get("prompt_tokens", 0) if result.usage else 0
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider, model=model, latency_ms=_lat, input_tokens=_in
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="embed",
                    status="ok",
                    latency_s=_lat / 1000,
                    input_tokens=_in,
                )
            return result
        except Exception as _exc:
            _lat = (time.perf_counter() - t0) * 1000
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider,
                    model=model,
                    latency_ms=_lat,
                    status="error",
                    error_type=type(_exc).__name__,
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=model,
                    operation="embed",
                    status="error",
                    latency_s=_lat / 1000,
                )
            raise

    # ------------------------------------------------------------------
    # Internal — Gemini stream event processing
    # ------------------------------------------------------------------

    @staticmethod
    def _process_gemini_event(
        event: Any,
        accumulated: str,
        accumulated_thinking: str,
        tool_calls: list[ToolCallResult],
    ) -> StreamChunk:
        text_delta = ""
        thinking_delta = ""

        if event.candidates:
            candidate = event.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "thought", False):
                        if part.text:
                            thinking_delta += part.text
                    elif part.text:
                        text_delta += part.text
                    if part.function_call:
                        fc = part.function_call
                        tool_calls.append(
                            ToolCallResult(
                                id=getattr(fc, "id", "") or "",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            ),
                        )

        accumulated += text_delta
        accumulated_thinking += thinking_delta
        is_last = bool(event.candidates and event.candidates[0].finish_reason is not None)

        usage: dict[str, int] = {}
        if is_last and hasattr(event, "usage_metadata") and event.usage_metadata:
            um = event.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        finish = (FinishReason.TOOL_CALL if tool_calls else FinishReason.STOP) if is_last else None
        is_thinking_chunk = bool(thinking_delta) and not text_delta

        return StreamChunk(
            delta=StreamDelta(text=text_delta, thinking=thinking_delta),
            accumulated_text=accumulated,
            accumulated_thinking=accumulated_thinking,
            is_thinking=is_thinking_chunk,
            finish_reason=finish,
            tool_calls=tool_calls if is_last else [],
            usage=usage,
            is_final=is_last,
            raw=event,
        )
