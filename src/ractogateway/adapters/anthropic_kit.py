"""Anthropic Claude adapter."""

from __future__ import annotations

import os
from typing import Any

from ractogateway.adapters.base import (
    BaseLLMAdapter,
    FinishReason,
    LLMResponse,
    ToolCallResult,
)
from ractogateway.exceptions import RactoGatewayError, _wrap_provider_error
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry


def _require_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for AnthropicLLMKit. "
            "Install it with:  pip install ractogateway[anthropic]"
        ) from exc
    return anthropic


class AnthropicLLMKit(BaseLLMAdapter):
    """Adapter for the Anthropic Messages API.

    Parameters
    ----------
    model:
        Model name (e.g. ``"claude-sonnet-4-5-20250929"``).
    api_key:
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    """

    provider: str = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _make_client(self, *, async_: bool = False) -> Any:
        anthropic = _require_anthropic()
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        params: dict[str, Any] = {}
        if key:
            params["api_key"] = key
        if async_:
            return anthropic.AsyncAnthropic(**params)
        return anthropic.Anthropic(**params)

    # ------------------------------------------------------------------
    # Tool translation
    # ------------------------------------------------------------------

    def translate_tools(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Convert registry schemas to Anthropic tool format."""
        tools: list[dict[str, Any]] = []
        for schema in registry.schemas:
            tools.append(
                {
                    "name": schema.name,
                    "description": schema.description,
                    "input_schema": schema.to_json_schema(),
                }
            )
        return tools

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _map_finish_reason(stop_reason: str | None) -> FinishReason:
        mapping: dict[str | None, FinishReason] = {
            "end_turn": FinishReason.STOP,
            "tool_use": FinishReason.TOOL_CALL,
            "max_tokens": FinishReason.LENGTH,
        }
        return mapping.get(stop_reason, FinishReason.STOP)

    def _normalise(self, response: Any) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCallResult] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCallResult(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                    )
                )

        content = "\n".join(text_parts) if text_parts else None

        # Usage
        usage: dict[str, int] = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

        return self._build_response(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._map_finish_reason(response.stop_reason),
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
            prompt,
            user_message,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        try:
            response = client.messages.create(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc
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
            prompt,
            user_message,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        try:
            response = await client.messages.create(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "anthropic") from exc
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
        system_prompt = prompt.compile()
        request: dict[str, Any] = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools and len(tools) > 0:
            request["tools"] = self.translate_tools(tools)
        request.update(kwargs)
        return request
