"""OpenAI / Azure OpenAI adapter."""

from __future__ import annotations

import os
from typing import Any

from ractogateway.adapters.base import (
    BaseLLMAdapter,
    FinishReason,
    LLMResponse,
    ToolCallResult,
)
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry


def _require_openai() -> Any:
    try:
        import openai  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for OpenAILLMKit. "
            "Install it with:  pip install ractogateway[openai]"
        ) from exc
    return openai


class OpenAILLMKit(BaseLLMAdapter):
    """Adapter for the OpenAI Chat Completions API.

    Parameters
    ----------
    model:
        Model name (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    base_url:
        Optional custom base URL (for Azure OpenAI or proxies).
    """

    provider: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _make_client(self, *, async_: bool = False) -> Any:
        openai = _require_openai()
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        params: dict[str, Any] = {}
        if key:
            params["api_key"] = key
        if self.base_url:
            params["base_url"] = self.base_url
        if async_:
            return openai.AsyncOpenAI(**params)
        return openai.OpenAI(**params)

    # ------------------------------------------------------------------
    # Tool translation
    # ------------------------------------------------------------------

    def translate_tools(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Convert registry schemas to OpenAI function-calling format."""
        tools: list[dict[str, Any]] = []
        for schema in registry.schemas:
            tools.append({
                "type": "function",
                "function": {
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.to_json_schema(),
                },
            })
        return tools

    # ------------------------------------------------------------------
    # Finish reason mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_finish_reason(reason: str | None) -> FinishReason:
        mapping: dict[str | None, FinishReason] = {
            "stop": FinishReason.STOP,
            "tool_calls": FinishReason.TOOL_CALL,
            "length": FinishReason.LENGTH,
            "content_filter": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(reason, FinishReason.STOP)

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalise(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message

        # Text content
        content = msg.content

        # Tool calls
        tool_calls: list[ToolCallResult] = []
        if msg.tool_calls:
            import json

            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCallResult(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        # Usage
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return self._build_response(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._map_finish_reason(choice.finish_reason),
            usage=usage,
            raw=response,
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._make_client()
        request = self._build_request(
            prompt, user_message,
            tools=tools, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        response = client.chat.completions.create(**request)
        return self._normalise(response)

    async def arun(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._make_client(async_=True)
        request = self._build_request(
            prompt, user_message,
            tools=tools, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        response = await client.chat.completions.create(**request)
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = prompt.to_messages(user_message, provider="openai")
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and len(tools) > 0:
            request["tools"] = self.translate_tools(tools)
        request.update(kwargs)
        return request
