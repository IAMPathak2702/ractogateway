"""HuggingFace Inference adapter — cloud and local TGI/vLLM servers.

Requires the ``huggingface_hub`` package::

    pip install ractogateway[huggingface]

Set ``HF_TOKEN`` (or ``HUGGINGFACE_TOKEN``) in the environment for the
HuggingFace Inference API.  For local servers (TGI / vLLM) pass
``base_url`` and omit the token.
"""

from __future__ import annotations

import json as _json
import os
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


def _require_huggingface_hub() -> Any:
    """Lazily import huggingface_hub — raises a friendly error if not installed."""
    try:
        import huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required for HuggingFaceLLMKit. "
            "Install it with:  pip install ractogateway[huggingface]"
        ) from exc
    return huggingface_hub


class HuggingFaceLLMKit(BaseLLMAdapter):
    """Low-level adapter for HuggingFace Inference API and local TGI/vLLM servers.

    Uses ``InferenceClient.chat_completion()`` (OpenAI-compatible endpoint) so
    it works with any chat-capable model hosted on HF or self-hosted via TGI /
    vLLM / Llama.cpp.

    Parameters
    ----------
    model:
        HuggingFace model repo ID
        (e.g. ``"meta-llama/Llama-3.2-3B-Instruct"``).
        For local servers set ``base_url`` and this can be any identifier the
        server understands (often ``"tgi"`` for TGI's default endpoint).
    api_key:
        HuggingFace token.  Falls back to ``HF_TOKEN`` then
        ``HUGGINGFACE_TOKEN`` environment variables.
    base_url:
        Custom endpoint URL.  When supplied, requests are sent there instead
        of the HF Inference API (useful for local TGI / vLLM deployments).
    """

    provider: str = "huggingface"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
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

    def _resolve_token(self) -> str | None:
        """Return the first available HF token from kwargs or env vars."""
        return (
            self.api_key
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_TOKEN")
        )

    def _make_client(self, *, async_: bool = False) -> Any:
        """Return a sync or async HuggingFace InferenceClient."""
        hf = _require_huggingface_hub()
        params: dict[str, Any] = {}
        token = self._resolve_token()
        if token:
            params["token"] = token
        if self.base_url:
            params["base_url"] = self.base_url
        if async_:
            return hf.AsyncInferenceClient(**params)
        return hf.InferenceClient(**params)

    # ------------------------------------------------------------------
    # Tool translation  (OpenAI-compatible function-calling format)
    # ------------------------------------------------------------------

    def translate_tools(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Convert registry schemas to HuggingFace/OpenAI function-calling format."""
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
            "eos_token": FinishReason.STOP,
        }
        return mapping.get(reason, FinishReason.STOP)

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalise(self, response: Any) -> LLMResponse:
        """Map a HuggingFace ChatCompletionOutput to a unified LLMResponse."""
        choice = response.choices[0]
        msg = choice.message
        content: str | None = getattr(msg, "content", None) or None

        # Tool calls
        tool_calls: list[ToolCallResult] = []
        raw_tcs = getattr(msg, "tool_calls", None)
        if raw_tcs:
            for tc in raw_tcs:
                func = tc.function
                raw_args = getattr(func, "arguments", "{}")
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

        # Usage
        usage: dict[str, int] = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "prompt_tokens": int(getattr(raw_usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(raw_usage, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }

        return self._build_response(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._map_finish_reason(choice.finish_reason),
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
        """Build the kwargs dict for ``client.chat_completion(**request)``."""
        _atts = list(kwargs.pop("attachments", None) or [])
        messages = prompt.to_messages(user_message, attachments=_atts, provider="openai")
        if history:
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
            response = client.chat_completion(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "huggingface") from exc
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
            response = await client.chat_completion(**request)
        except RactoGatewayError:
            raise
        except Exception as exc:
            raise _wrap_provider_error(exc, "huggingface") from exc
        return self._normalise(response)
