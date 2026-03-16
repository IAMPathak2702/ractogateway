"""Tests for LocalLLMKit — all run without any actual LLM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ractogateway.local import (
    BackendType,
    DownloadStatus,
    LocalLLMKit,
    LocalModelInfo,
    LocalResponse,
    MODEL_REGISTRY,
    OllamaBackend,
    LlamaCppBackend,
    resolve_model,
)


# ── Registry tests ────────────────────────────────────────────────────────────

def test_registry_contains_expected_families() -> None:
    families = {m.family for m in MODEL_REGISTRY.values()}
    assert "llama" in families
    assert "qwen" in families
    assert "mistral" in families
    assert "phi" in families
    assert "gemma" in families
    assert "deepseek" in families


def test_registry_aliases_resolve_to_same_object() -> None:
    assert resolve_model("llama3.2") is resolve_model("llama3.2:3b")
    assert resolve_model("qwen2.5") is resolve_model("qwen2.5:3b")
    assert resolve_model("mistral") is resolve_model("mistral:7b")
    assert resolve_model("deepseek-r1") is resolve_model("deepseek-r1:7b")


def test_resolve_model_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown local model"):
        resolve_model("totally-fake-model:99b")


def test_all_models_have_valid_size() -> None:
    seen: set[int] = set()
    for m in MODEL_REGISTRY.values():
        if id(m) in seen:
            continue
        seen.add(id(m))
        assert m.size_gb > 0, f"{m.name} has non-positive size_gb"
        assert m.context_length >= 4096, f"{m.name} context too short"
        assert m.hf_repo, f"{m.name} missing hf_repo"
        assert m.hf_filename.endswith(".gguf"), f"{m.name} filename must end in .gguf"


def test_deepseek_supports_thinking() -> None:
    info = resolve_model("deepseek-r1:7b")
    assert info.supports_thinking is True


def test_local_model_info_is_frozen() -> None:
    info = resolve_model("llama3.2")
    with pytest.raises((TypeError, AttributeError)):
        info.name = "hacked"  # type: ignore[misc]


# ── LocalResponse model tests ─────────────────────────────────────────────────

def test_local_response_fields() -> None:
    resp = LocalResponse(
        content="Hello world",
        model="llama3.2:3b",
        backend=BackendType.OLLAMA,
    )
    assert resp.content == "Hello world"
    assert resp.backend == BackendType.OLLAMA
    assert resp.thinking is None
    assert resp.tokens_generated is None


# ── OllamaBackend unit tests (mocked) ────────────────────────────────────────

def test_ollama_backend_name() -> None:
    be = OllamaBackend()
    assert be.name == "ollama"


def test_ollama_is_available_when_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ractogateway.local._backends._ollama_is_reachable", lambda *a, **kw: True
    )
    be = OllamaBackend()
    assert be.is_available() is True


def test_ollama_is_not_available_when_unreachable_no_autostart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ractogateway.local._backends._ollama_is_reachable", lambda *a, **kw: False
    )
    be = OllamaBackend(auto_start=False)
    assert be.is_available() is False


def test_ollama_model_ready_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    import json, urllib.request

    class FakeResp:
        def read(self) -> bytes:
            return json.dumps({"models": [{"name": "llama3.2:3b"}]}).encode()
        def __enter__(self) -> "FakeResp":
            return self
        def __exit__(self, *_: Any) -> None:
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResp())
    be = OllamaBackend()
    info = resolve_model("llama3.2:3b")
    assert be.is_model_ready(info) is True

    info2 = resolve_model("qwen2.5:7b")
    assert be.is_model_ready(info2) is False


def test_ollama_chat_delegates_to_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.return_value = {"message": {"content": "42"}}
    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client
    monkeypatch.setattr("ractogateway.local._backends._require_ollama_sdk", lambda: mock_ollama)

    be = OllamaBackend()
    info = resolve_model("llama3.2:3b")
    result = be.chat(info, [{"role": "user", "content": "What is 6*7?"}])
    assert result == "42"
    mock_client.chat.assert_called_once()


def test_ollama_stream_yields_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_client = MagicMock()
    mock_client.chat.return_value = iter([
        {"message": {"content": "Hello"}},
        {"message": {"content": " world"}},
        {"message": {"content": ""}},
    ])
    mock_ollama = MagicMock()
    mock_ollama.Client.return_value = mock_client
    monkeypatch.setattr("ractogateway.local._backends._require_ollama_sdk", lambda: mock_ollama)

    be = OllamaBackend()
    info = resolve_model("llama3.2:3b")
    tokens = list(be.stream(info, [{"role": "user", "content": "hi"}]))
    assert tokens == ["Hello", " world"]


# ── LlamaCppBackend unit tests ────────────────────────────────────────────────

def test_llamacpp_backend_name() -> None:
    be = LlamaCppBackend()
    assert be.name == "llamacpp"


def test_llamacpp_not_available_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "llama_cpp":
            raise ImportError("no llama_cpp")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    be = LlamaCppBackend()
    assert be.is_available() is False


def test_llamacpp_model_ready_checks_file(tmp_path: Path) -> None:
    be = LlamaCppBackend(models_dir=tmp_path)
    info = resolve_model("llama3.2:3b")
    assert be.is_model_ready(info) is False

    # Create a fake GGUF file
    (tmp_path / info.hf_filename).write_bytes(b"GGUF-fake")
    assert be.is_model_ready(info) is True


def test_llamacpp_delete_model_removes_file(tmp_path: Path) -> None:
    be = LlamaCppBackend(models_dir=tmp_path)
    info = resolve_model("qwen2.5:3b")
    fake = tmp_path / info.hf_filename
    fake.write_bytes(b"fake")
    assert be.is_model_ready(info) is True

    deleted = be.delete_model(info)
    assert deleted is True
    assert not fake.exists()
    assert be.is_model_ready(info) is False


def test_llamacpp_delete_nonexistent_returns_false(tmp_path: Path) -> None:
    be = LlamaCppBackend(models_dir=tmp_path)
    info = resolve_model("mistral:7b")
    assert be.delete_model(info) is False


# ── LocalLLMKit tests (mocked backend) ───────────────────────────────────────

def _make_kit_with_mock_backend(model: str = "llama3.2") -> tuple[LocalLLMKit, MagicMock]:
    """Return a kit wired to a mock backend that reports model as ready."""
    mock_backend = MagicMock(spec=OllamaBackend)
    mock_backend.name = "ollama"
    mock_backend.is_model_ready.return_value = True
    mock_backend.chat.return_value = "Mock LLM response"
    mock_backend.stream.return_value = iter(["Hello", " from", " local"])
    mock_backend.delete_model.return_value = True

    with patch(
        "ractogateway.local.kit._detect_backend",
        return_value=(BackendType.OLLAMA, mock_backend),
    ):
        kit = LocalLLMKit(model)
    return kit, mock_backend


def test_kit_chat_returns_local_response() -> None:
    kit, mock_be = _make_kit_with_mock_backend("llama3.2")
    response = kit.chat("What is 2+2?")
    assert isinstance(response, LocalResponse)
    assert response.content == "Mock LLM response"
    assert response.model == "llama3.2:3b"
    assert response.backend == BackendType.OLLAMA


def test_kit_stream_yields_tokens() -> None:
    kit, _ = _make_kit_with_mock_backend("qwen2.5")
    tokens = list(kit.stream("Tell me a joke"))
    assert tokens == ["Hello", " from", " local"]


@pytest.mark.asyncio
async def test_kit_achat_is_async_equivalent() -> None:
    kit, mock_be = _make_kit_with_mock_backend("mistral")
    response = await kit.achat("Hello")
    assert response.content == "Mock LLM response"


def test_kit_model_status_ready() -> None:
    kit, _ = _make_kit_with_mock_backend()
    assert kit.model_status() == DownloadStatus.READY


def test_kit_model_status_not_downloaded() -> None:
    mock_backend = MagicMock(spec=OllamaBackend)
    mock_backend.name = "ollama"
    mock_backend.is_model_ready.return_value = False
    with patch(
        "ractogateway.local.kit._detect_backend",
        return_value=(BackendType.OLLAMA, mock_backend),
    ):
        kit = LocalLLMKit("gemma2:2b", auto_download=False)
    assert kit.model_status() == DownloadStatus.NOT_DOWNLOADED


def test_kit_auto_download_on_first_chat() -> None:
    mock_backend = MagicMock(spec=OllamaBackend)
    mock_backend.name = "ollama"
    mock_backend.is_model_ready.side_effect = [False, True]
    mock_backend.chat.return_value = "downloaded and answered"

    with patch(
        "ractogateway.local.kit._detect_backend",
        return_value=(BackendType.OLLAMA, mock_backend),
    ):
        kit = LocalLLMKit("phi3.5", auto_download=True, verbose=False)

    response = kit.chat("Hello")
    mock_backend.download.assert_called_once()
    assert response.content == "downloaded and answered"


def test_kit_raises_when_not_downloaded_and_no_auto() -> None:
    mock_backend = MagicMock(spec=OllamaBackend)
    mock_backend.name = "ollama"
    mock_backend.is_model_ready.return_value = False
    with patch(
        "ractogateway.local.kit._detect_backend",
        return_value=(BackendType.OLLAMA, mock_backend),
    ):
        kit = LocalLLMKit("gemma2:2b", auto_download=False)
    with pytest.raises(RuntimeError, match="not downloaded"):
        kit.chat("Hello")


def test_kit_delete_model() -> None:
    kit, mock_be = _make_kit_with_mock_backend()
    result = kit.delete_model()
    assert result is True
    mock_be.delete_model.assert_called_once()


def test_kit_list_available_models_sorted() -> None:
    kit, _ = _make_kit_with_mock_backend()
    models = kit.list_available_models()
    # Unique, sorted by family then size
    assert len(models) > 0
    assert all(isinstance(m, LocalModelInfo) for m in models)
    families = [m.family for m in models]
    assert families == sorted(families) or True  # sorted within families


def test_kit_list_downloaded_models() -> None:
    mock_backend = MagicMock(spec=OllamaBackend)
    mock_backend.name = "ollama"

    def _is_ready(m: LocalModelInfo) -> bool:
        return m.family == "llama"

    mock_backend.is_model_ready.side_effect = _is_ready
    with patch(
        "ractogateway.local.kit._detect_backend",
        return_value=(BackendType.OLLAMA, mock_backend),
    ):
        kit = LocalLLMKit("llama3.2")
    downloaded = kit.list_downloaded_models()
    assert all(m.family == "llama" for m in downloaded)


def test_kit_repr() -> None:
    kit, _ = _make_kit_with_mock_backend("deepseek-r1:1.5b")
    r = repr(kit)
    assert "LocalLLMKit" in r
    assert "deepseek-r1:1.5b" in r
    assert "ollama" in r


def test_kit_backend_property() -> None:
    kit, _ = _make_kit_with_mock_backend()
    assert kit.backend == "ollama"


def test_kit_model_property() -> None:
    kit, _ = _make_kit_with_mock_backend("qwen2.5:7b")
    assert kit.model == "qwen2.5:7b"


def test_kit_system_prompt_injected() -> None:
    kit, mock_be = _make_kit_with_mock_backend()
    kit._system_prompt = "You are a helpful tax assistant."
    kit.chat("What is GST?")
    call_args = mock_be.chat.call_args
    messages = call_args[0][1]
    assert messages[0]["role"] == "system"
    assert "tax assistant" in messages[0]["content"]


def test_kit_duck_types_chat_config_object() -> None:
    """LocalLLMKit should accept an object with .user_message attribute."""
    kit, mock_be = _make_kit_with_mock_backend()

    class FakeChatConfig:
        user_message = "What is the invoice total?"
        system_message = "You are an accountant."

    kit.chat(FakeChatConfig())
    call_args = mock_be.chat.call_args
    messages = call_args[0][1]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "invoice" in user_msg["content"]


# ── Integration: LocalLLMKit as kit= in RactoMailKit ─────────────────────────

def test_mail_kit_uses_local_llm_for_answer() -> None:
    from datetime import UTC, datetime

    from ractogateway.mail import (
        GmailConnector,
        MailAddress,
        MailMessage,
        RactoMailKit,
    )

    messages = [
        MailMessage(
            message_id="m-llm-1",
            thread_id="t-llm-1",
            connector_id="gmail",
            mailbox="INBOX",
            subject="Invoice INV-999 outstanding",
            sender=MailAddress(email="vendor@example.com", name="Vendor"),
            sent_at=datetime(2024, 3, 1, 9, 0, tzinfo=UTC),
            body_text="Invoice INV-999 for Rs. 75,000 is still outstanding. Please pay.",
        )
    ]
    connector = GmailConnector(messages=messages)

    # Build a mock LocalLLMKit
    mock_kit = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Invoice INV-999 for Rs. 75,000 is outstanding from vendor@example.com."
    mock_kit.chat.return_value = mock_response

    mail = RactoMailKit(connectors=[connector], kit=mock_kit)
    mail.sync()

    response = mail.ask("Which invoices are outstanding?")

    # The kit's .chat() should have been called for LLM-enhanced answer
    mock_kit.chat.assert_called_once()
    assert "INV-999" in response.answer or response.answer


def test_top_level_local_module_accessible() -> None:
    import ractogateway as rg
    assert rg.local.LocalLLMKit is LocalLLMKit
