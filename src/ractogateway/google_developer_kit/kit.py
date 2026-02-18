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
from typing import Any

from ractogateway._models.chat import ChatConfig
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.adapters.google_kit import GoogleLLMKit
from ractogateway.prompts.engine import RactoPrompt


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
    """Complete Google Gemini developer kit — chat, stream, and embeddings.

    Parameters
    ----------
    model:
        Gemini model (e.g. ``"gemini-2.0-flash"``, ``"gemini-2.5-pro"``).
    api_key:
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
    embedding_model:
        Default embedding model.  Defaults to ``"text-embedding-004"``.
    default_prompt:
        RACTO prompt used when ``ChatConfig.prompt`` is ``None``.
    """

    provider: str = "google"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        embedding_model: str = "text-embedding-004",
        default_prompt: RactoPrompt | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._embedding_model = embedding_model
        self._default_prompt = default_prompt
        self._adapter = GoogleLLMKit(model=model, api_key=api_key)

    # ------------------------------------------------------------------
    # Client factory
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        genai = _require_genai()
        key = self._api_key or os.environ.get("GEMINI_API_KEY")
        return genai.Client(api_key=key)

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
        """Synchronous streaming via ``generate_content_stream``.

        Example::

            for chunk in kit.stream(config):
                print(chunk.delta.text, end="", flush=True)
        """
        from google.genai import types

        prompt = self._resolve_prompt(config)
        client = self._client()
        system_prompt = prompt.compile()
        gen_config = self._adapter._build_config(
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )

        accumulated = ""
        tool_calls: list[ToolCallResult] = []

        for event in client.models.generate_content_stream(
            model=self._model,
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
            yield chunk

    async def astream(self, config: ChatConfig) -> AsyncIterator[StreamChunk]:
        """Async streaming via ``aio.models.generate_content_stream``."""
        from google.genai import types

        prompt = self._resolve_prompt(config)
        client = self._client()
        system_prompt = prompt.compile()
        gen_config = self._adapter._build_config(
            tools=config.tools,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )

        accumulated = ""
        tool_calls: list[ToolCallResult] = []

        async for event in await client.aio.models.generate_content_stream(
            model=self._model,
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
            yield chunk

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


# ======================================================================
# Module-level helpers
# ======================================================================


def _maybe_validate(response: LLMResponse, config: ChatConfig) -> LLMResponse:
    if config.response_model is not None and isinstance(response.parsed, dict):
        try:
            validated = config.response_model.model_validate(response.parsed)
            response.parsed = validated.model_dump()
        except Exception as exc:
            warning = f"[RactoGateway] response_model validation failed: {exc}"
            response.content = f"{response.content}\n\n{warning}" if response.content else warning
    return response
