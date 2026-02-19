from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

import ractogateway._models as shared_models
from ractogateway._models.chat import ChatConfig, Message, MessageRole
from ractogateway._models.embedding import EmbeddingConfig, EmbeddingResponse, EmbeddingVector
from ractogateway._models.stream import StreamChunk, StreamDelta
from ractogateway.adapters.base import FinishReason, ToolCallResult


class StructuredAnswer(BaseModel):
    answer: str


def test_message_role_values_are_stable() -> None:
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole("user") is MessageRole.USER


def test_message_model_captures_role_and_content() -> None:
    message = Message(role=MessageRole.USER, content="Hello")

    assert message.role is MessageRole.USER
    assert message.content == "Hello"


def test_chat_config_defaults_are_isolated_per_instance() -> None:
    config_one = ChatConfig(user_message="First")
    config_two = ChatConfig(user_message="Second")

    assert config_one.temperature == 0.0
    assert config_one.max_tokens == 4096
    assert config_one.history == []
    assert config_one.extra == {}
    assert config_one.prompt is None
    assert config_one.tools is None
    assert config_one.response_model is None

    config_one.history.append(Message(role=MessageRole.USER, content="Earlier turn"))
    config_one.extra["top_p"] = 0.8

    assert config_two.history == []
    assert config_two.extra == {}


def test_chat_config_validation_constraints() -> None:
    with pytest.raises(ValidationError):
        ChatConfig(user_message="")

    with pytest.raises(ValidationError):
        ChatConfig(user_message="ok", temperature=-0.01)

    with pytest.raises(ValidationError):
        ChatConfig(user_message="ok", temperature=2.01)

    with pytest.raises(ValidationError):
        ChatConfig(user_message="ok", max_tokens=0)


def test_chat_config_accepts_pydantic_response_model_type() -> None:
    config = ChatConfig(user_message="Return structured output", response_model=StructuredAnswer)

    assert config.response_model is StructuredAnswer


def test_embedding_config_defaults_and_validation() -> None:
    config = EmbeddingConfig(texts=["alpha", "beta"])

    assert config.model is None
    assert config.dimensions is None
    assert config.extra == {}

    with pytest.raises(ValidationError):
        EmbeddingConfig(texts=[])


def test_embedding_response_contains_vectors_and_default_usage() -> None:
    vector = EmbeddingVector(index=0, text="alpha", embedding=[0.1, 0.2, 0.3])
    response = EmbeddingResponse(vectors=[vector], model="mock-model")

    assert response.model == "mock-model"
    assert len(response.vectors) == 1
    assert response.vectors[0].embedding == [0.1, 0.2, 0.3]
    assert response.usage == {}
    assert response.raw is None


def test_stream_models_defaults_and_explicit_values() -> None:
    default_chunk = StreamChunk()
    assert default_chunk.delta == StreamDelta()
    assert default_chunk.accumulated_text == ""
    assert default_chunk.finish_reason is None
    assert default_chunk.tool_calls == []
    assert default_chunk.usage == {}
    assert default_chunk.is_final is False
    assert default_chunk.raw is None

    tool_call = ToolCallResult(name="lookup_user", arguments={"user_id": "u-1"})
    chunk = StreamChunk(
        delta=StreamDelta(
            text="Hello",
            tool_call_id="tool-1",
            tool_call_name="lookup_user",
            tool_call_args_fragment='{"user_id": "u-1"}',
        ),
        accumulated_text="Hello",
        finish_reason=FinishReason.STOP,
        tool_calls=[tool_call],
        usage={"total_tokens": 5},
        is_final=True,
        raw={"provider": "mock"},
    )

    assert chunk.delta.text == "Hello"
    assert chunk.finish_reason is FinishReason.STOP
    assert chunk.tool_calls[0].name == "lookup_user"
    assert chunk.usage["total_tokens"] == 5
    assert chunk.is_final is True
    assert chunk.raw == {"provider": "mock"}


def test_shared_models_init_exports_expected_symbols() -> None:
    assert "ChatConfig" in shared_models.__all__
    assert "EmbeddingConfig" in shared_models.__all__
    assert "EmbeddingResponse" in shared_models.__all__
    assert "EmbeddingVector" in shared_models.__all__
    assert "Message" in shared_models.__all__
    assert "MessageRole" in shared_models.__all__
    assert "StreamChunk" in shared_models.__all__
    assert "StreamDelta" in shared_models.__all__
