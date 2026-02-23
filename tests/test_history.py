"""Regression tests for ChatConfig.history being propagated to every adapter.

All tests are pure unit tests — they mock the provider SDK and assert that the
correct message sequence (history + current user turn) reaches the API call.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway.adapters.anthropic_kit import AnthropicLLMKit
from ractogateway.adapters.base import ChatTurn
from ractogateway.adapters.google_kit import GoogleLLMKit, build_google_contents
from ractogateway.adapters.openai_kit import OpenAILLMKit
from ractogateway.prompts.engine import RactoPrompt

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

_PROMPT = RactoPrompt(
    role="You are a helpful assistant.",
    aim="Recall information from the conversation.",
    constraints=["Do not make up answers."],
    tone="Friendly.",
    output_format="Plain text.",
)

_HISTORY: list[ChatTurn] = [
    {"role": "user", "content": "My name is Alex."},
    {"role": "assistant", "content": "Nice to meet you, Alex!"},
]


# ---------------------------------------------------------------------------
# Helper — build a ChatConfig with the "My name is Alex" history
# ---------------------------------------------------------------------------


def _alex_config() -> ChatConfig:
    return ChatConfig(
        user_message="What is my name?",
        prompt=_PROMPT,
        history=[
            Message(role=MessageRole.USER, content="My name is Alex."),
            Message(role=MessageRole.ASSISTANT, content="Nice to meet you, Alex!"),
        ],
    )


# ---------------------------------------------------------------------------
# OpenAI adapter — _build_request
# ---------------------------------------------------------------------------


class TestOpenAIHistoryInMessages:
    def test_history_spliced_between_system_and_user(self) -> None:
        adapter = OpenAILLMKit(model="gpt-4o", api_key="test")
        request = adapter._build_request(
            _PROMPT,
            "What is my name?",
            history=_HISTORY,
        )
        messages = request["messages"]
        roles = [m["role"] for m in messages]
        # Expected: system, user (Alex), assistant (Nice to meet you), user (What is my name?)
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[1]["content"] == "My name is Alex."
        assert messages[2]["content"] == "Nice to meet you, Alex!"
        assert messages[3]["content"] == "What is my name?"

    def test_no_history_unchanged(self) -> None:
        adapter = OpenAILLMKit(model="gpt-4o", api_key="test")
        request = adapter._build_request(_PROMPT, "Hello", history=None)
        roles = [m["role"] for m in request["messages"]]
        assert roles == ["system", "user"]

    def test_history_forwarded_from_run(self) -> None:
        """run() must pass history through to _build_request / the API call."""
        adapter = OpenAILLMKit(model="gpt-4o", api_key="test")

        captured: list[Any] = []

        def _fake_create(**kwargs: Any) -> Any:
            captured.append(kwargs)
            raise RuntimeError("stop after capture")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _fake_create

        with patch.object(adapter, "_make_client", return_value=mock_client):
            with pytest.raises(Exception):  # _wrap_provider_error wraps RuntimeError
                adapter.run(_PROMPT, "What is my name?", history=_HISTORY)

        assert captured, "API was never called"
        messages = captured[0]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Anthropic adapter — _build_request
# ---------------------------------------------------------------------------


class TestAnthropicHistoryInMessages:
    def test_history_prepended_to_messages(self) -> None:
        adapter = AnthropicLLMKit(model="claude-haiku-4-5-20251001", api_key="test")
        request = adapter._build_request(
            _PROMPT,
            "What is my name?",
            history=_HISTORY,
        )
        messages = request["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user"]
        assert messages[0]["content"] == "My name is Alex."
        assert messages[1]["content"] == "Nice to meet you, Alex!"
        assert messages[2]["content"] == "What is my name?"
        # system must NOT appear in messages — it belongs in the top-level key
        assert "system" in request

    def test_no_history_single_user_message(self) -> None:
        adapter = AnthropicLLMKit(model="claude-haiku-4-5-20251001", api_key="test")
        request = adapter._build_request(_PROMPT, "Hello", history=None)
        assert [m["role"] for m in request["messages"]] == ["user"]

    def test_history_forwarded_from_run(self) -> None:
        adapter = AnthropicLLMKit(model="claude-haiku-4-5-20251001", api_key="test")

        captured: list[Any] = []

        def _fake_create(**kwargs: Any) -> Any:
            captured.append(kwargs)
            raise RuntimeError("stop after capture")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = _fake_create

        with patch.object(adapter, "_make_client", return_value=mock_client):
            with pytest.raises(Exception):  # _wrap_provider_error wraps RuntimeError
                adapter.run(_PROMPT, "What is my name?", history=_HISTORY)

        assert captured, "API was never called"
        messages = captured[0]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Google adapter — build_google_contents helper
# ---------------------------------------------------------------------------


class TestBuildGoogleContents:
    def test_no_history_returns_plain_string(self) -> None:
        result = build_google_contents(None, "Hello")
        assert result == "Hello"

    def test_empty_history_returns_plain_string(self) -> None:
        result = build_google_contents([], "Hello")
        assert result == "Hello"

    def test_history_builds_content_list(self) -> None:
        result = build_google_contents(_HISTORY, "What is my name?")
        assert isinstance(result, list)
        assert len(result) == 3
        # First turn: user
        assert result[0].role == "user"
        assert result[0].parts[0].text == "My name is Alex."
        # Second turn: model (mapped from "assistant")
        assert result[1].role == "model"
        assert result[1].parts[0].text == "Nice to meet you, Alex!"
        # Current turn: user
        assert result[2].role == "user"
        assert result[2].parts[0].text == "What is my name?"

    def test_assistant_role_mapped_to_model(self) -> None:
        history: list[ChatTurn] = [{"role": "assistant", "content": "hi"}]
        result = build_google_contents(history, "ok")
        assert result[0].role == "model"


# ---------------------------------------------------------------------------
# Kit-level integration: ChatConfig.history flows through OpenAI kit
# ---------------------------------------------------------------------------


class TestOpenAIKitHistory:
    def test_chat_passes_history_to_adapter(self) -> None:
        """OpenAIDeveloperKit.chat() must plumb ChatConfig.history into adapter.run()."""
        from ractogateway.openai_developer_kit.kit import OpenAIDeveloperKit

        kit = OpenAIDeveloperKit(model="gpt-4o", api_key="test", default_prompt=_PROMPT)

        captured: list[Any] = []

        def _fake_create(**kwargs: Any) -> Any:
            captured.append(kwargs)
            raise RuntimeError("stop after capture")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = _fake_create

        with patch(
            "ractogateway.adapters.openai_kit.OpenAILLMKit._make_client",
            return_value=mock_client,
        ):
            with pytest.raises(Exception):  # _wrap_provider_error wraps RuntimeError
                kit.chat(_alex_config())

        assert captured, "API was never called"
        messages = captured[0]["messages"]
        roles = [m["role"] for m in messages]
        # system + 2 history turns + current user
        assert roles == ["system", "user", "assistant", "user"]
        assert messages[-1]["content"] == "What is my name?"
