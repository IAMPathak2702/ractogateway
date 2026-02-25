"""OpenAI / Azure OpenAI adapter."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from ractogateway.adapters._openai_schema import build_response_format
from ractogateway.adapters.base import (
    BaseLLMAdapter,
    ChatTurn,
    FinishReason,
    LLMResponse,
    ToolCallResult,
)
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry


def _require_openai() -> Any:
    try:
        import openai
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
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": schema.name,
                        "description": schema.description,
                        "parameters": schema.to_json_schema(),
                    },
                }
            )
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
        history: list[ChatTurn] | None = None,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._make_client()
        request = self._build_request(
            prompt,
            user_message,
            history=history,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        try:
            response = client.chat.completions.create(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc
        return self._normalise(response)

    async def arun(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        history: list[ChatTurn] | None = None,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._make_client(async_=True)
        request = self._build_request(
            prompt,
            user_message,
            history=history,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        try:
            response = await client.chat.completions.create(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "openai") from exc
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        prompt: RactoPrompt,
        user_message: str,
        *,
        history: list[ChatTurn] | None = None,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        _atts = list(kwargs.pop("attachments", None) or [])
        messages = prompt.to_messages(user_message, attachments=_atts, provider="openai")
        if history:
            # Splice history turns between the system message and the current user message.
            messages = [
                messages[0],
                *[{"role": t["role"], "content": t["content"]} for t in history],
                messages[-1],
            ]
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and len(tools) > 0:
            request["tools"] = self.translate_tools(tools)

        # Use OpenAI Structured Outputs when the output_format is a Pydantic model.
        # build_response_format validates and sanitises the schema *before* the API
        # call so that incompatible Pydantic keywords (default, minimum, anyOf issues
        # etc.) raise a clear ValueError here rather than an opaque API rejection.
        # Users can override by passing ``response_format=...`` in kwargs.
        if (
            "response_format" not in kwargs
            and isinstance(prompt.output_format, type)
            and issubclass(prompt.output_format, BaseModel)
        ):
            request["response_format"] = build_response_format(prompt.output_format)

        request.update(kwargs)
        return request
