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
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

from ractogateway._models.chat import ChatConfig
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.anthropic_kit import AnthropicLLMKit
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult, try_parse_json
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt

if TYPE_CHECKING:
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.routing.router import CostAwareRouter
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
        """Synchronous chat completion with optional middleware pipeline."""
        prompt = self._resolve_prompt(config)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        system_prompt = prompt.compile()

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

        if self._semantic_cache is not None:
            sem_cached = self._semantic_cache.get(config.user_message)
            if sem_cached is not None:
                return sem_cached

        adapter = self._get_adapter(model)
        response = adapter.run(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        response = _maybe_validate(response, config)

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

        if self._semantic_cache is not None:
            sem_cached = self._semantic_cache.get(config.user_message)
            if sem_cached is not None:
                return sem_cached

        adapter = self._get_adapter(model)
        response = await adapter.arun(
            prompt,
            config.user_message,
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
        response = _maybe_validate(response, config)

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
        """Synchronous streaming via Anthropic's ``messages.stream()``.

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

        accumulated = ""
        tc_acc: dict[int, dict[str, Any]] = {}
        finish_reason = FinishReason.STOP
        usage: dict[str, int] = {}

        try:
            with client.messages.stream(**request) as stream_resp:
                for event in stream_resp:
                    chunk = self._process_anthropic_event(
                        event,
                        accumulated,
                        tc_acc,
                        finish_reason,
                        usage,
                    )
                    if chunk is not None:
                        accumulated = chunk.accumulated_text
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.usage:
                            usage = chunk.usage
                        if chunk.is_final and config.response_model is not None:
                            _apply_stream_response_model(chunk, config)
                        yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming via Anthropic's async ``messages.stream()``."""
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

        accumulated = ""
        tc_acc: dict[int, dict[str, Any]] = {}
        finish_reason = FinishReason.STOP
        usage: dict[str, int] = {}

        try:
            async with client.messages.stream(**request) as stream_resp:
                async for event in stream_resp:
                    chunk = self._process_anthropic_event(
                        event,
                        accumulated,
                        tc_acc,
                        finish_reason,
                        usage,
                    )
                    if chunk is not None:
                        accumulated = chunk.accumulated_text
                        if chunk.finish_reason is not None:
                            finish_reason = chunk.finish_reason
                        if chunk.usage:
                            usage = chunk.usage
                        if chunk.is_final and config.response_model is not None:
                            _apply_stream_response_model(chunk, config)
                        yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc

    # ------------------------------------------------------------------
    # Internal — Anthropic stream event processing
    # ------------------------------------------------------------------

    @staticmethod
    def _process_anthropic_event(
        event: Any,
        accumulated: str,
        tc_acc: dict[int, dict[str, Any]],
        finish_reason: FinishReason,
        usage: dict[str, int],
    ) -> StreamChunk | None:
        etype = event.type

        if etype == "content_block_start":
            block = event.content_block
            if block.type == "tool_use":
                tc_acc[event.index] = {
                    "id": block.id,
                    "name": block.name,
                    "args": "",
                }
            return StreamChunk(
                accumulated_text=accumulated,
                raw=event,
            )

        if etype == "content_block_delta":
            delta = event.delta
            if delta.type == "text_delta":
                text = delta.text
                accumulated += text
                return StreamChunk(
                    delta=StreamDelta(text=text),
                    accumulated_text=accumulated,
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
                finish_reason=fr,
                usage=u,
                raw=event,
            )

        if etype == "message_stop":
            return StreamChunk(
                accumulated_text=accumulated,
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
