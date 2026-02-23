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
from ractogateway.adapters.google_kit import GoogleLLMKit
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt

if TYPE_CHECKING:
    from ractogateway.cache.exact_cache import ExactMatchCache
    from ractogateway.cache.semantic_cache import SemanticCache
    from ractogateway.routing.router import CostAwareRouter
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
        """Synchronous streaming via ``generate_content_stream``.

        Example::

            for chunk in kit.stream(config):
                print(chunk.delta.text, end="", flush=True)
        """
        from google.genai import types

        prompt = self._resolve_prompt(config)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        adapter = self._get_adapter(model)
        client = self._client()
        system_prompt = prompt.compile()
        gen_config = adapter._build_config(
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )

        accumulated = ""
        tool_calls: list[ToolCallResult] = []

        try:
            for event in client.models.generate_content_stream(
                model=model,
                contents=config.user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    **gen_config,
                ),
            ):
                chunk = self._process_gemini_event(
                    event,
                    accumulated,
                    tool_calls,
                )
                accumulated = chunk.accumulated_text
                if chunk.is_final and config.response_model is not None:
                    chunk.parsed = validate_stream_final(chunk.accumulated_text, config)
                yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "google") from exc

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming via ``aio.models.generate_content_stream``."""
        from google.genai import types

        prompt = self._resolve_prompt(config)
        model = self._resolve_model(config.user_message)
        config = self._apply_truncation(config, model)
        adapter = self._get_adapter(model)
        client = self._client()
        system_prompt = prompt.compile()
        gen_config = adapter._build_config(
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )

        accumulated = ""
        tool_calls: list[ToolCallResult] = []

        try:
            async for event in await client.aio.models.generate_content_stream(
                model=model,
                contents=config.user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    **gen_config,
                ),
            ):
                chunk = self._process_gemini_event(
                    event,
                    accumulated,
                    tool_calls,
                )
                accumulated = chunk.accumulated_text
                if chunk.is_final and config.response_model is not None:
                    chunk.parsed = validate_stream_final(chunk.accumulated_text, config)
                yield chunk
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "google") from exc

    # ------------------------------------------------------------------
    # Embeddings  (sync / async)
    # ------------------------------------------------------------------

    def embed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Synchronous embedding via ``embed_content``."""
        client = self._client()
        model = config.model or self._embedding_model
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
        return EmbeddingResponse(vectors=vectors, model=model)

    async def aembed(self, config: EmbeddingConfig) -> EmbeddingResponse:
        """Async embedding via ``aio.models.embed_content``."""
        client = self._client()
        model = config.model or self._embedding_model
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
        return EmbeddingResponse(vectors=vectors, model=model)

    # ------------------------------------------------------------------
    # Internal — Gemini stream event processing
    # ------------------------------------------------------------------

    @staticmethod
    def _process_gemini_event(
        event: Any,
        accumulated: str,
        tool_calls: list[ToolCallResult],
    ) -> StreamChunk:
        text_delta = ""

        if event.candidates:
            candidate = event.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
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

        return StreamChunk(
            delta=StreamDelta(text=text_delta),
            accumulated_text=accumulated,
            finish_reason=finish,
            tool_calls=tool_calls if is_last else [],
            usage=usage,
            is_final=is_last,
            raw=event,
        )


