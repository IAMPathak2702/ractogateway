"""Google Gemini adapter (using the ``google-genai`` SDK)."""

from __future__ import annotations

import json as _json
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


def _require_genai() -> Any:
    try:
        from google import genai  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "The 'google-genai' package is required for GoogleLLMKit. "
            "Install it with:  pip install ractogateway[google]"
        ) from exc
    return genai


class GoogleLLMKit(BaseLLMAdapter):
    """Adapter for the Google Gemini API via ``google-genai``.

    Parameters
    ----------
    model:
        Model name (e.g. ``"gemini-2.0-flash"``, ``"gemini-2.5-pro"``).
    api_key:
        Gemini API key.  Falls back to ``GEMINI_API_KEY`` env var.
    """

    provider: str = "google"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)

    # ------------------------------------------------------------------
    # Client helper
    # ------------------------------------------------------------------

    def _make_client(self) -> Any:
        genai = _require_genai()
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        return genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Tool translation
    # ------------------------------------------------------------------

    def translate_tools(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        """Convert registry schemas to Gemini function declarations."""
        declarations: list[dict[str, Any]] = []
        for schema in registry.schemas:
            properties: dict[str, Any] = {}
            required: list[str] = []

            for p in schema.parameters:
                prop: dict[str, Any] = {"type": p.type.upper()}
                if p.description:
                    prop["description"] = p.description
                if p.enum is not None:
                    prop["enum"] = p.enum
                properties[p.name] = prop
                if p.required:
                    required.append(p.name)

            decl: dict[str, Any] = {
                "name": schema.name,
                "description": schema.description,
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                },
            }
            if required:
                decl["parameters"]["required"] = required
            declarations.append(decl)

        return declarations

    # ------------------------------------------------------------------
    # Response normalisation
    # ------------------------------------------------------------------

    def _normalise(self, response: Any) -> LLMResponse:
        # google-genai response: response.text, response.candidates, etc.
        content: str | None = None
        tool_calls: list[ToolCallResult] = []
        finish = FinishReason.STOP

        candidate = response.candidates[0] if response.candidates else None
        if candidate:
            parts = candidate.content.parts if candidate.content else []
            text_parts: list[str] = []

            for part in parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    tool_calls.append(
                        ToolCallResult(
                            id=fc.id if hasattr(fc, "id") and fc.id else "",
                            name=fc.name,
                            arguments=args,
                        )
                    )

            if text_parts:
                content = "\n".join(text_parts)
            if tool_calls:
                finish = FinishReason.TOOL_CALL

        # Usage
        usage: dict[str, int] = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
            }

        return self._build_response(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish,
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
        from google.genai import types

        client = self._make_client()
        config = self._build_config(
            tools=tools, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        system_prompt = prompt.compile()
        response = client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                **config,
            ),
        )
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
        from google.genai import types

        client = self._make_client()
        config = self._build_config(
            tools=tools, temperature=temperature,
            max_tokens=max_tokens, **kwargs,
        )
        system_prompt = prompt.compile()
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                **config,
            ),
        )
        return self._normalise(response)

    # ------------------------------------------------------------------
    # Config building
    # ------------------------------------------------------------------

    def _build_config(
        self,
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if tools and len(tools) > 0:
            config["tools"] = self.translate_tools(tools)
        config.update(kwargs)
        return config
