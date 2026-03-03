"""Anthropic Claude Developer Kit — production-grade Claude interface.

Usage::

    from ractogateway import anthropic_developer_kit as anth

    kit = anth.AnthropicDeveloperKit(model="claude-sonnet-4-5-20250929", default_prompt=my_prompt)
    response = kit.chat(anth.ChatConfig(user_message="Hello"))

    for chunk in kit.stream(anth.ChatConfig(user_message="Hello")):
        print(chunk.delta.text, end="", flush=True)

Note: Anthropic does NOT have a native embedding API.  Use OpenAI or
Google kits for embeddings.
"""

from __future__ import annotations

import json as _json
import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ractogateway._models.chat import ChatConfig
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
from ractogateway.adapters.anthropic_kit import AnthropicLLMKit
from ractogateway.adapters.base import ChatTurn, FinishReason, LLMResponse, ToolCallResult
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt

if TYPE_CHECKING:
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.routing.router import CostAwareRouter
    from ractogateway.telemetry.metrics import GatewayMetricsMiddleware
    from ractogateway.telemetry.tracer import RactoTracer
    from ractogateway.truncation.truncator import TokenTruncator


def _require_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for AnthropicDeveloperKit. "
            "Install it with:  pip install ractogateway[anthropic]"
        ) from exc
    return anthropic


class AnthropicDeveloperKit:
    """Complete Anthropic Claude developer kit — chat, streaming, and
    optional performance/cost optimisation middleware.

    Parameters
    ----------
    model:
        Claude model (e.g. ``"claude-sonnet-4-5-20250929"``, ``"claude-opus-4-6"``).
        Use ``"auto"`` when a :class:`~ractogateway.routing.CostAwareRouter`
        is provided — the router will select the model per-request.
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
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
        Emits OpenTelemetry spans for every chat and stream call.
        Requires ``pip install ractogateway[telemetry]``.
    metrics:
        Optional :class:`~ractogateway.telemetry.GatewayMetricsMiddleware`.
        Records Prometheus metrics (latency, tokens, cost, cache hit/miss).
        Requires ``pip install ractogateway[prometheus]``.
    """

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        *,
        api_key: str | None = None,
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
        self._default_prompt = default_prompt
        self._exact_cache = exact_cache
        self._semantic_cache = semantic_cache
        self._router = router
        self._truncator = truncator
        self._tracer = tracer
        self._metrics = metrics
        # Adapter pool for cost-aware routing
        self._adapters: dict[str, AnthropicLLMKit] = {}
        fallback = "claude-haiku-4-5-20251001"
        self._adapter = self._get_adapter(model if model != "auto" else fallback)

    # ------------------------------------------------------------------
    # Adapter pool
    # ------------------------------------------------------------------

    def _get_adapter(self, model: str) -> AnthropicLLMKit:
        """Return (or lazily create) an adapter for *model*."""
        if model not in self._adapters:
            self._adapters[model] = AnthropicLLMKit(model=model, api_key=self._api_key)
        return self._adapters[model]

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
        """Synchronous streaming via Anthropic's ``messages.stream()``.

        Example::

            for chunk in kit.stream(config):
                print(chunk.delta.text, end="", flush=True)
                if chunk.is_final:
                    print(f"\\nTokens: {chunk.usage}")
        """
        t0 = time.perf_counter()
        prompt = self._resolve_prompt(config)
        if config.chain_of_thought:
            from ractogateway._cot import apply_chain_of_thought
            prompt = apply_chain_of_thought(prompt)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        validation_config = with_inferred_response_model(config, prompt)
        adapter = self._get_adapter(model)
        client = self._sync_client()
        history_turns: list[ChatTurn] | None = (
            [ChatTurn(role=m.role.value, content=m.content) for m in config.history]
            if config.history
            else None
        )
        request = adapter._build_request(
            prompt,
            config.user_message,
            history=history_turns,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            attachments=config.attachments,
            native_thinking=config.native_thinking,
            thinking_budget=config.thinking_budget,
            **config.extra,
        )

        accumulated = ""
        accumulated_thinking = ""
        tc_acc: dict[int, dict[str, Any]] = {}
        thinking_indices: set[int] = set()
        finish_reason = FinishReason.STOP
        usage: dict[str, int] = {}
        _span_recorded = False

        try:
            with client.messages.stream(**request) as stream_resp:
                for event in stream_resp:
                    chunk = self._process_anthropic_event(
                        event,
                        accumulated,
                        accumulated_thinking,
                        tc_acc,
                        thinking_indices,
                        finish_reason,
                        usage,
                    )
                    if chunk is not None:
                        accumulated = chunk.accumulated_text
                        accumulated_thinking = chunk.accumulated_thinking
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.usage:
                            usage = chunk.usage
                        if chunk.is_final and validation_config.response_model is not None:
                            chunk.parsed = validate_stream_final(
                                chunk.accumulated_text, validation_config
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
            raise _wrap_provider_error(exc, "anthropic") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming via Anthropic's async ``messages.stream()``."""
        t0 = time.perf_counter()
        prompt = self._resolve_prompt(config)
        if config.chain_of_thought:
            from ractogateway._cot import apply_chain_of_thought
            prompt = apply_chain_of_thought(prompt)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        validation_config = with_inferred_response_model(config, prompt)
        adapter = self._get_adapter(model)
        client = self._async_client()
        history_turns: list[ChatTurn] | None = (
            [ChatTurn(role=m.role.value, content=m.content) for m in config.history]
            if config.history
            else None
        )
        request = adapter._build_request(
            prompt,
            config.user_message,
            history=history_turns,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            attachments=config.attachments,
            native_thinking=config.native_thinking,
            thinking_budget=config.thinking_budget,
            **config.extra,
        )

        accumulated = ""
        accumulated_thinking = ""
        tc_acc: dict[int, dict[str, Any]] = {}
        thinking_indices: set[int] = set()
        finish_reason = FinishReason.STOP
        usage: dict[str, int] = {}
        _span_recorded = False

        try:
            async with client.messages.stream(**request) as stream_resp:
                async for event in stream_resp:
                    chunk = self._process_anthropic_event(
                        event,
                        accumulated,
                        accumulated_thinking,
                        tc_acc,
                        thinking_indices,
                        finish_reason,
                        usage,
                    )
                    if chunk is not None:
                        accumulated = chunk.accumulated_text
                        accumulated_thinking = chunk.accumulated_thinking
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.usage:
                            usage = chunk.usage
                        if chunk.is_final and validation_config.response_model is not None:
                            chunk.parsed = validate_stream_final(
                                chunk.accumulated_text, validation_config
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
            raise _wrap_provider_error(exc, "anthropic") from exc

    # ------------------------------------------------------------------
    # Internal — Anthropic stream event processing
    # ------------------------------------------------------------------

    @staticmethod
    def _process_anthropic_event(
        event: Any,
        accumulated: str,
        accumulated_thinking: str,
        tc_acc: dict[int, dict[str, Any]],
        thinking_indices: set[int],
        finish_reason: FinishReason,
        usage: dict[str, int],
    ) -> StreamChunk | None:
        etype = event.type

        if etype == "content_block_start":
            block = event.content_block
            if block.type == "thinking":
                thinking_indices.add(event.index)
            elif block.type == "tool_use":
                tc_acc[event.index] = {
                    "id": block.id,
                    "name": block.name,
                    "args": "",
                }
            return StreamChunk(
                accumulated_text=accumulated,
                accumulated_thinking=accumulated_thinking,
                raw=event,
            )

        if etype == "content_block_delta":
            delta = event.delta
            if delta.type == "thinking_delta":
                thinking_text = delta.thinking
                accumulated_thinking += thinking_text
                return StreamChunk(
                    delta=StreamDelta(thinking=thinking_text),
                    accumulated_text=accumulated,
                    accumulated_thinking=accumulated_thinking,
                    is_thinking=True,
                    raw=event,
                )
            if delta.type == "text_delta":
                text = delta.text
                accumulated += text
                return StreamChunk(
                    delta=StreamDelta(text=text),
                    accumulated_text=accumulated,
                    accumulated_thinking=accumulated_thinking,
                    raw=event,
                )
            if delta.type == "input_json_delta":
                idx = event.index
                if idx in tc_acc:
                    tc_acc[idx]["args"] += delta.partial_json
                return StreamChunk(
                    delta=StreamDelta(
                        tool_call_args_fragment=delta.partial_json,
                    ),
                    accumulated_text=accumulated,
                    accumulated_thinking=accumulated_thinking,
                    raw=event,
                )
            return None

        if etype == "message_delta":
            stop_reason = getattr(event.delta, "stop_reason", None)
            fr = AnthropicLLMKit._map_finish_reason(stop_reason)
            u: dict[str, int] = {}
            if hasattr(event, "usage") and event.usage:
                inp = getattr(event.usage, "input_tokens", 0) or 0
                out = getattr(event.usage, "output_tokens", 0) or 0
                u = {
                    "prompt_tokens": inp,
                    "completion_tokens": out,
                    "total_tokens": inp + out,
                }
            return StreamChunk(
                accumulated_text=accumulated,
                accumulated_thinking=accumulated_thinking,
                finish_reason=fr,
                usage=u,
                raw=event,
            )

        if etype == "message_stop":
            return StreamChunk(
                accumulated_text=accumulated,
                accumulated_thinking=accumulated_thinking,
                finish_reason=finish_reason,
                tool_calls=_flush_tool_calls(tc_acc),
                usage=usage,
                is_final=True,
                raw=event,
            )

        return None


# ======================================================================
# Module-level helpers
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
