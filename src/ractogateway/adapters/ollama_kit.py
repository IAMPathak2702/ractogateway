"""Ollama adapter — local model inference via the Ollama REST API.

Requires the ``ollama`` Python package::

    pip install ractogateway[ollama]

The Ollama server must be running locally (default: ``http://localhost:11434``).
Pull a model first::

    ollama pull llama3.2
"""

from __future__ import annotations

import json as _json
from typing import Any

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


def _require_ollama() -> Any:
    """Lazily import ollama — raises a friendly error if not installed."""
    try:
        import ollama
    except ImportError as exc:
        raise ImportError(
            "The 'ollama' package is required for OllamaLLMKit. "
            "Install it with:  pip install ractogateway[ollama]"
        ) from exc
    return ollama


class OllamaLLMKit(BaseLLMAdapter):
    """Low-level adapter for the Ollama local-inference REST API.

    Parameters
    ----------
    model:
        Model name as reported by ``ollama list``
        (e.g. ``"llama3.2"``, ``"mistral"``, ``"gemma3"``).
    base_url:
        Ollama server base URL.  Defaults to ``http://localhost:11434``.
    """

    provider: str = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        base_url: str = "http://localhost:11434",
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key=None, **kwargs)
        self.base_url = base_url

    # ------------------------------------------------------------------
    # Client helpers
    # ------------------------------------------------------------------

    def _make_client(self, *, async_: bool = False) -> Any:
        """Return a sync or async Ollama client bound to *base_url*."""
        ollama = _require_ollama()
        params: dict[str, Any] = {"host": self.base_url}
        if async_:
            return ollama.AsyncClient(**params)
        return ollama.Client(**params)

    # ------------------------------------------------------------------
    # Tool translation  (OpenAI-compatible function-calling format)
    # ------------------------------------------------------------------

    def translate_tools(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Convert registry schemas to Ollama function-calling format."""
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
        }
        return mapping.get(reason, FinishReason.STOP)

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalise(self, response: Any) -> LLMResponse:
        """Map an Ollama ChatResponse to a unified LLMResponse."""
        msg = response.message
        content: str | None = getattr(msg, "content", None) or None

        # Tool calls
        tool_calls: list[ToolCallResult] = []
        raw_tcs = getattr(msg, "tool_calls", None)
        if raw_tcs:
            for tc in raw_tcs:
                func = tc.function
                raw_args = getattr(func, "arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args: dict[str, Any] = _json.loads(raw_args)
                    except _json.JSONDecodeError:
                        args = {"_raw": raw_args}
                else:
                    args = dict(raw_args) if raw_args else {}
                tool_calls.append(
                    ToolCallResult(
                        id=str(getattr(tc, "id", "") or ""),
                        name=str(func.name),
                        arguments=args,
                    )
                )

        # Usage (only populated on the final non-streaming response)
        usage: dict[str, int] = {}
        prompt_count = getattr(response, "prompt_eval_count", None)
        eval_count = getattr(response, "eval_count", None)
        if prompt_count is not None:
            usage["prompt_tokens"] = int(prompt_count)
        if eval_count is not None:
            usage["completion_tokens"] = int(eval_count)
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )

        finish_reason = FinishReason.TOOL_CALL if tool_calls else FinishReason.STOP
        done_reason = getattr(response, "done_reason", None)
        if done_reason and not tool_calls:
            finish_reason = self._map_finish_reason(done_reason)

        return self._build_response(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=response,
        )

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
        """Build the kwargs dict for ``client.chat(**request)``."""
        _atts = list(kwargs.pop("attachments", None) or [])
        messages = prompt.to_messages(user_message, attachments=_atts, provider="ollama")
        if history:
            messages = [
                messages[0],
                *[{"role": t["role"], "content": t["content"]} for t in history],
                messages[-1],
            ]

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools and len(tools) > 0:
            request["tools"] = self.translate_tools(tools)
        request.update(kwargs)
        return request

    # ------------------------------------------------------------------
    # Execution — sync / async
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
        """Execute a chat request synchronously."""
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
            response = client.chat(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "ollama") from exc
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
        """Execute a chat request asynchronously."""
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
            response = await client.chat(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "ollama") from exc
        return self._normalise(response)
