"""Unified Gateway Runner — single entry point for all LLM interactions.

The ``Gateway`` class ties together prompt compilation, adapter selection,
tool injection, and response standardisation into one high-level interface.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ractogateway.adapters.base import BaseLLMAdapter, LLMResponse
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry


class Gateway:
    """Unified entry point that wraps any ``BaseLLMAdapter``.

    Parameters
    ----------
    adapter:
        A concrete adapter instance (``OpenAILLMKit``, ``GoogleLLMKit``,
        ``AnthropicLLMKit``).
    tools:
        An optional ``ToolRegistry`` containing registered tools that the
        LLM is allowed to call.
    default_prompt:
        An optional ``RactoPrompt`` to use when ``run()`` is called
        without an explicit prompt.

    Usage::

        from ractogateway import RactoPrompt, Gateway
        from ractogateway.adapters import OpenAILLMKit

        adapter = OpenAILLMKit(model="gpt-4o", api_key="sk-...")
        prompt  = RactoPrompt(...)
        gw      = Gateway(adapter=adapter)

        response = gw.run(prompt, "Analyse this code for bugs.")
        print(response.parsed)   # auto-parsed JSON dict
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        *,
        tools: ToolRegistry | None = None,
        default_prompt: RactoPrompt | None = None,
    ) -> None:
        self.adapter = adapter
        self.tools = tools
        self.default_prompt = default_prompt

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: RactoPrompt | None = None,
        user_message: str = "",
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute a request and return a standardised ``LLMResponse``.

        Parameters
        ----------
        prompt:
            The RACTO prompt.  Falls back to ``default_prompt``.
        user_message:
            The end-user's query.
        tools:
            Override the gateway-level tool registry for this call.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens in the response.
        response_model:
            Optional Pydantic model to validate ``parsed`` output against.
            If provided and the LLM returns valid JSON, it is validated
            through this model and attached to ``response.parsed``.
        **kwargs:
            Passed through to the adapter.
        """
        effective_prompt = prompt or self.default_prompt
        if effective_prompt is None:
            raise ValueError("No prompt provided and no default_prompt configured on the Gateway.")

        effective_tools = tools or self.tools

        response = self.adapter.run(
            effective_prompt,
            user_message,
            tools=effective_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if response_model is not None and response.parsed is not None:
            response = self._validate_response(response, response_model)

        return response

    async def arun(
        self,
        prompt: RactoPrompt | None = None,
        user_message: str = "",
        *,
        tools: ToolRegistry | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async variant of ``run()``."""
        effective_prompt = prompt or self.default_prompt
        if effective_prompt is None:
            raise ValueError("No prompt provided and no default_prompt configured on the Gateway.")

        effective_tools = tools or self.tools

        response = await self.adapter.arun(
            effective_prompt,
            user_message,
            tools=effective_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        if response_model is not None and response.parsed is not None:
            response = self._validate_response(response, response_model)

        return response

    # ------------------------------------------------------------------
    # Response validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_response(
        response: LLMResponse,
        model: type[BaseModel],
    ) -> LLMResponse:
        """Validate and re-parse the response through a Pydantic model.

        If validation succeeds, ``response.parsed`` is replaced with the
        validated model's ``.model_dump()``.  If it fails, the original
        ``parsed`` dict is left untouched and the validation error is
        stored in ``response.content`` as a warning.
        """
        try:
            if isinstance(response.parsed, dict):
                validated = model.model_validate(response.parsed)
                response.parsed = validated.model_dump()
        except Exception as exc:
            # Don't crash — surface the validation issue but keep raw data.
            warning = f"[RactoGateway] response_model validation failed: {exc}"
            if response.content:
                response.content = f"{response.content}\n\n{warning}"
            else:
                response.content = warning
        return response
