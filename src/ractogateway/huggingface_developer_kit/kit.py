"""HuggingFace Developer Kit — production-grade HuggingFace interface.

Usage::

    from ractogateway import huggingface_developer_kit as hf

    kit = hf.HuggingFaceDeveloperKit(
        model="meta-llama/Llama-3.2-3B-Instruct",
        default_prompt=my_prompt,
    )
    response = kit.chat(hf.ChatConfig(user_message="Hello"))

    for chunk in kit.stream(hf.ChatConfig(user_message="Hello")):
        print(chunk.delta.text, end="", flush=True)

Set ``HF_TOKEN`` in the environment for cloud inference, or pass
``base_url`` to point at a self-hosted TGI / vLLM / Llama.cpp server.

Local TGI example::

    # docker run --rm -p 8080:80 ghcr.io/huggingface/text-generation-inference \\
    #     --model-id meta-llama/Llama-3.2-3B-Instruct
    kit = hf.HuggingFaceDeveloperKit(
        model="tgi",
        base_url="http://localhost:8080",
    )
"""

from __future__ import annotations

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
from ractogateway.adapters.huggingface_kit import HuggingFaceLLMKit
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt

if TYPE_CHECKING:
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.routing.router import CostAwareRouter
    from ractogateway.telemetry.metrics import GatewayMetricsMiddleware
    from ractogateway.telemetry.tracer import RactoTracer
    from ractogateway.truncation.truncator import TokenTruncator


def _require_huggingface_hub() -> Any:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required for HuggingFaceDeveloperKit. "
            "Install it with:  pip install ractogateway[huggingface]"
        ) from exc
    return huggingface_hub


class HuggingFaceDeveloperKit:
    """Complete HuggingFace developer kit — chat, stream, embeddings, and
    optional performance/cost optimisation middleware.

    Works with both the HuggingFace Inference API (cloud) and local
    deployments (TGI / vLLM / Llama.cpp) via ``base_url``.

    Parameters
    ----------
    model:
        HuggingFace model repo ID
        (e.g. ``"meta-llama/Llama-3.2-3B-Instruct"``).
        For local servers use any identifier the server expects (e.g. ``"tgi"``).
        Use ``"auto"`` when a :class:`~ractogateway.routing.CostAwareRouter`
        is provided — the router will select the model per-request.
    api_key:
        HuggingFace token.  Falls back to ``HF_TOKEN`` then
        ``HUGGINGFACE_TOKEN`` environment variables.
    base_url:
        Custom endpoint URL.  When set, requests go to the local/private server
        instead of the HuggingFace Inference API.
    embedding_model:
        Default model for embedding calls.
        Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"``.
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
    metrics:
        Optional :class:`~ractogateway.telemetry.GatewayMetricsMiddleware`.
    """

    provider: str = "huggingface"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
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
        self._base_url = base_url
        self._embedding_model = embedding_model
        self._default_prompt = default_prompt
        self._exact_cache = exact_cache
        self._semantic_cache = semantic_cache
        self._router = router
        self._truncator = truncator
        self._tracer = tracer
        self._metrics = metrics
        self._adapters: dict[str, HuggingFaceLLMKit] = {}
        if model != "auto":
            self._adapter = self._get_adapter(model)
        else:
            self._adapter = self._get_adapter("meta-llama/Llama-3.2-3B-Instruct")  # placeholder

    # ------------------------------------------------------------------
    # Adapter pool
    # ------------------------------------------------------------------

    def _get_adapter(self, model: str) -> HuggingFaceLLMKit:
        """Return (or lazily create) an adapter for *model*."""
        if model not in self._adapters:
            self._adapters[model] = HuggingFaceLLMKit(
                model=model,
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._adapters[model]

    # ------------------------------------------------------------------
    # Client factories
    # ------------------------------------------------------------------

    def _sync_client(self) -> Any:
        import os

        hf = _require_huggingface_hub()
        params: dict[str, Any] = {}
        token = self._api_key or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_TOKEN"
        )
        if token:
            params["token"] = token
        if self._base_url:
            params["base_url"] = self._base_url
        return hf.InferenceClient(**params)

    def _async_client(self) -> Any:
        import os

        hf = _require_huggingface_hub()
        params: dict[str, Any] = {}
        token = self._api_key or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGINGFACE_TOKEN"
        )
        if token:
            params["token"] = token
        if self._base_url:
            params["base_url"] = self._base_url
        return hf.AsyncInferenceClient(**params)

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

        if self._exact_cache is not None:
            cached = self._exact_cache.get(
                config.user_message, system_prompt, model, config.temperature, config.max_tokens
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
                    follow_up = build_tool_followup_user_message(
                        original_user_message=original_user_message,
                        tool_calls=response.tool_calls,
                        results=results,
                    )
                    response = _run_validated(follow_up)

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

        if self._exact_cache is not None:
            cached = self._exact_cache.get(
                config.user_message, system_prompt, model, config.temperature, config.max_tokens
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
                    results = await execute_tool_calls_async(response.tool_calls, config.tools)
                    follow_up = build_tool_followup_user_message(
                        original_user_message=original_user_message,
                        tool_calls=response.tool_calls,
                        results=results,
                    )
                    response = await _arun_validated(follow_up)

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
        """Synchronous streaming — yields ``StreamChunk`` objects.

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
            **config.extra,
        )
        request["stream"] = True

        accumulated = ""
        tc_acc: dict[int, dict[str, Any]] = {}
        _span_recorded = False

        try:
            for event in client.chat_completion(**request):
                chunk = self._process_hf_event(event, accumulated, tc_acc)
                if chunk is not None:
                    accumulated = chunk.accumulated_text
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
            raise _wrap_provider_error(exc, "huggingface") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming — yields ``StreamChunk`` objects."""
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
            **config.extra,
        )
        request["stream"] = True

        accumulated = ""
        tc_acc: dict[int, dict[str, Any]] = {}
        _span_recorded = False

        try:
            async for event in await client.chat_completion(**request):
                chunk = self._process_hf_event(event, accumulated, tc_acc)
                if chunk is not None:
                    accumulated = chunk.accumulated_text
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
            raise _wrap_provider_error(exc, "huggingface") from exc

    # ------------------------------------------------------------------
    # Embeddings  (sync / async)
    # ------------------------------------------------------------------

    def embed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Synchronous embedding via HuggingFace ``feature_extraction``.

        Example::

            resp = kit.embed(EmbeddingConfig(texts=["hello", "world"]))
            print(resp.vectors[0].embedding[:5])
        """
        t0 = time.perf_counter()
        client = self._sync_client()
        try:
            result = self._do_embed(client, config)
            _lat = (time.perf_counter() - t0) * 1000
            _model = config.model or self._embedding_model
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider, model=_model, latency_ms=_lat
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=_model,
                    operation="embed",
                    status="ok",
                    latency_s=_lat / 1000,
                )
            return result
        except Exception as _exc:
            _lat = (time.perf_counter() - t0) * 1000
            _model = config.model or self._embedding_model
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider,
                    model=_model,
                    latency_ms=_lat,
                    status="error",
                    error_type=type(_exc).__name__,
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=_model,
                    operation="embed",
                    status="error",
                    latency_s=_lat / 1000,
                )
            raise

    async def aembed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Async embedding via HuggingFace ``feature_extraction``."""
        t0 = time.perf_counter()
        client = self._async_client()
        try:
            result = await self._do_aembed(client, config)
            _lat = (time.perf_counter() - t0) * 1000
            _model = config.model or self._embedding_model
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider, model=_model, latency_ms=_lat
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=_model,
                    operation="embed",
                    status="ok",
                    latency_s=_lat / 1000,
                )
            return result
        except Exception as _exc:
            _lat = (time.perf_counter() - t0) * 1000
            _model = config.model or self._embedding_model
            if self._tracer is not None:
                self._tracer.record_embed_span(
                    provider=self.provider,
                    model=_model,
                    latency_ms=_lat,
                    status="error",
                    error_type=type(_exc).__name__,
                )
            if self._metrics is not None:
                self._metrics.record_request(
                    provider=self.provider,
                    model=_model,
                    operation="embed",
                    status="error",
                    latency_s=_lat / 1000,
                )
            raise

    # ------------------------------------------------------------------
    # Internal — HuggingFace streaming event processing (OpenAI-compatible)
    # ------------------------------------------------------------------

    def _process_hf_event(
        self,
        event: Any,
        accumulated: str,
        tc_acc: dict[int, dict[str, Any]],
    ) -> StreamChunk | None:
        """Process one HuggingFace streaming event into a ``StreamChunk``."""
        choices = getattr(event, "choices", None)
        if not choices:
            return None

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return None

        text: str = getattr(delta, "content", "") or ""
        accumulated += text
        sd = StreamDelta(text=text)

        # Accumulate tool-call fragments (OpenAI-compatible)
        raw_tcs = getattr(delta, "tool_calls", None)
        if raw_tcs:
            for tc in raw_tcs:
                idx = getattr(tc, "index", 0)
                if idx not in tc_acc:
                    tc_acc[idx] = {"id": str(getattr(tc, "id", "") or ""), "name": "", "args": ""}
                func = getattr(tc, "function", None)
                if func is not None:
                    name = getattr(func, "name", None)
                    args_frag = getattr(func, "arguments", None)
                    if name:
                        tc_acc[idx]["name"] = name
                    if args_frag:
                        tc_acc[idx]["args"] += args_frag
                sd = StreamDelta(
                    text=text,
                    tool_call_id=tc_acc[idx]["id"],
                    tool_call_name=tc_acc[idx]["name"],
                    tool_call_args_fragment=(
                        getattr(getattr(tc, "function", None), "arguments", None)
                    ),
                )

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None:
            finish = HuggingFaceLLMKit._map_finish_reason(finish_reason)
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
        """Run a synchronous embedding call via HuggingFace ``feature_extraction``."""
        model = config.model or self._embedding_model
        kw: dict[str, Any] = {}
        kw.update(config.extra)
        raw = client.feature_extraction(config.texts, model=model, **kw)
        return _normalise_hf_embedding(raw, config.texts, model)

    async def _do_aembed(self, client: Any, config: EmbeddingConfig) -> EmbeddingResponse:
        """Run an async embedding call via HuggingFace ``feature_extraction``."""
        model = config.model or self._embedding_model
        kw: dict[str, Any] = {}
        kw.update(config.extra)
        raw = await client.feature_extraction(config.texts, model=model, **kw)
        return _normalise_hf_embedding(raw, config.texts, model)


# ======================================================================
# Module-level helpers
# ======================================================================


def _flush_tool_calls(acc: dict[int, dict[str, Any]]) -> list[ToolCallResult]:
    """Materialise accumulated streaming tool-call fragments."""
    import json as _json

    results: list[ToolCallResult] = []
    for entry in acc.values():
        try:
            args: dict[str, Any] = _json.loads(entry["args"]) if entry["args"] else {}
        except _json.JSONDecodeError:
            args = {"_raw": entry["args"]}
        results.append(
            ToolCallResult(id=entry["id"], name=entry["name"], arguments=args),
        )
    return results


def _normalise_hf_embedding(
    raw: Any,
    texts: list[str],
    model: str,
) -> EmbeddingResponse:
    """Normalise HuggingFace feature_extraction output into EmbeddingResponse.

    ``feature_extraction`` returns either:

    * ``list[float]`` — single text was passed  → wrap in a list
    * ``list[list[float]]`` — multiple texts → use directly
    * numpy array or nested array — convert to Python lists
    """
    # Normalise numpy arrays if present
    try:
        import numpy as np

        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
    except ImportError:
        pass

    # Coerce to list[list[float]]
    if raw and isinstance(raw[0], float):
        embeddings: list[list[float]] = [list(raw)]
    else:
        embeddings = [list(row) for row in raw]

    vectors = [
        EmbeddingVector(index=i, text=texts[i], embedding=emb)
        for i, emb in enumerate(embeddings)
    ]
    return EmbeddingResponse(vectors=vectors, model=model, usage={}, raw=raw)
