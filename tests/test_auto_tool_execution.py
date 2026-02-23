from __future__ import annotations

import sys
from typing import Any

import pytest

from ractogateway._models.chat import ChatConfig
from ractogateway.adapters.base import FinishReason, LLMResponse, ToolCallResult
from ractogateway.prompts.engine import RactoPrompt
from ractogateway.tools.registry import ToolRegistry


class _FakeAdapter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def run(
        self,
        prompt: RactoPrompt,
        user_message: str,
        **_: Any,
    ) -> LLMResponse:
        self.messages.append(user_message)
        if len(self.messages) == 1:
            return LLMResponse(
                content=None,
                parsed=None,
                finish_reason=FinishReason.TOOL_CALL,
                tool_calls=[
                    ToolCallResult(name="calculate", arguments={"expression": "12 * 8"})
                ],
            )
        return LLMResponse(content="12 * 8 = 96", parsed=None)

    async def arun(
        self,
        prompt: RactoPrompt,
        user_message: str,
        **_: Any,
    ) -> LLMResponse:
        return self.run(prompt, user_message)


def _prompt() -> RactoPrompt:
    return RactoPrompt(
        role="You are a helpful assistant with tools.",
        aim="Answer the user using available tools.",
        constraints=["Use tools when needed."],
        tone="Helpful.",
        output_format="text",
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def calculate(expression: str) -> float:
        return float(eval(expression))  # noqa: S307

    return registry


@pytest.fixture(autouse=True)
def _cleanup_openai_modules() -> None:
    yield
    sys.modules.pop("ractogateway.openai_developer_kit", None)
    sys.modules.pop("ractogateway.openai_developer_kit.kit", None)


def test_chat_auto_execute_tools_returns_final_content(monkeypatch: Any) -> None:
    from ractogateway.openai_developer_kit.kit import OpenAIDeveloperKit

    fake_adapter = _FakeAdapter()
    monkeypatch.setattr(OpenAIDeveloperKit, "_get_adapter", lambda self, model: fake_adapter)

    kit = OpenAIDeveloperKit(model="gpt-4o", default_prompt=_prompt())
    config = ChatConfig(
        user_message="What is 12 * 8?",
        tools=_registry(),
        auto_execute_tools=True,
        max_tool_turns=3,
    )

    response = kit.chat(config)

    assert response.content == "12 * 8 = 96"
    assert len(fake_adapter.messages) == 2
    assert fake_adapter.messages[0] == "What is 12 * 8?"
    assert "Tool execution results:" in fake_adapter.messages[1]


def test_chat_without_auto_execute_tools_returns_tool_call_turn(monkeypatch: Any) -> None:
    from ractogateway.openai_developer_kit.kit import OpenAIDeveloperKit

    fake_adapter = _FakeAdapter()
    monkeypatch.setattr(OpenAIDeveloperKit, "_get_adapter", lambda self, model: fake_adapter)

    kit = OpenAIDeveloperKit(model="gpt-4o", default_prompt=_prompt())
    config = ChatConfig(
        user_message="What is 12 * 8?",
        tools=_registry(),
        auto_execute_tools=False,
    )

    response = kit.chat(config)

    assert response.finish_reason is FinishReason.TOOL_CALL
    assert response.content is None
    assert len(response.tool_calls) == 1
    assert len(fake_adapter.messages) == 1
