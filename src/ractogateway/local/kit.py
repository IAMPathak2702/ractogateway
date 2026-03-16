"""LocalLLMKit — privacy-first local LLM inference for RactoGateway."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from ._backends import (
    DEFAULT_MODELS_DIR,
    BaseLocalBackend,
    LlamaCppBackend,
    OllamaBackend,
    _OLLAMA_DEFAULT_URL,
)
from ._registry import LocalModelInfo, MODEL_REGISTRY, resolve_model


# ── Public enums & models ─────────────────────────────────────────────────────

class BackendType(str, Enum):
    """Which local inference backend to use."""

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    AUTO = "auto"


class DownloadStatus(str, Enum):
    """Whether a model is available for inference."""

    NOT_DOWNLOADED = "not_downloaded"
    READY = "ready"


class LocalResponse(BaseModel):
    """Response returned by :meth:`LocalLLMKit.chat`."""

    content: str
    model: str
    backend: BackendType
    tokens_generated: int | None = None
    thinking: str | None = None


# ── Auto backend detection ────────────────────────────────────────────────────

def _detect_backend(
    *,
    models_dir: Path | None,
    context_length: int,
    gpu_layers: int,
    base_url: str,
) -> tuple[BackendType, BaseLocalBackend]:
    """Try Ollama first, then llama-cpp, raise if neither is available."""
    ollama_be = OllamaBackend(base_url=base_url)
    if ollama_be.is_available():
        return BackendType.OLLAMA, ollama_be

    llama_be = LlamaCppBackend(
        models_dir=models_dir,
        context_length=context_length,
        gpu_layers=gpu_layers,
    )
    if llama_be.is_available():
        return BackendType.LLAMACPP, llama_be

    raise RuntimeError(
        "No local LLM backend found on this machine.\n\n"
        "Option A — Ollama  (easiest, recommended):\n"
        "  1. Install: https://ollama.com/download\n"
        "  2. Start:   ollama serve\n"
        "  3. Then re-create LocalLLMKit — it auto-downloads the model.\n\n"
        "Option B — llama-cpp-python  (no server, runs GGUF files directly):\n"
        "  pip install ractogateway[local-llamacpp]\n"
        "  Then re-create LocalLLMKit.\n\n"
        "Option C — llama-cpp with GPU acceleration:\n"
        "  pip install ractogateway[local-llamacpp-gpu]\n"
    )


# ── Main kit ──────────────────────────────────────────────────────────────────

class LocalLLMKit:
    """Privacy-first local LLM kit — all inference stays on your laptop.

    Supports Ollama and llama-cpp-python.  Auto-detects which backend is
    available and downloads the model on first use.

    Quick start::

        from ractogateway.local import LocalLLMKit

        kit = LocalLLMKit("llama3.2")          # 3B — good default
        response = kit.chat("What is GST?")
        print(response.content)

        # Stream tokens
        for token in kit.stream("Explain RAG in 3 sentences"):
            print(token, end="", flush=True)

        # Use as the LLM brain for RactoMailKit
        from ractogateway.mail import RactoMailKit
        mail = RactoMailKit(connectors=[...], kit=kit)

    Available models (pass any name or alias):

    +--------------------+-------+------+---------------------------------------+
    | Name               |  Size | RAM  | Description                           |
    +====================+=======+======+=======================================+
    | llama3.2:1b        | 0.8GB |  1GB | Meta Llama 3.2 1B — ultra-light       |
    | llama3.2:3b (def.) | 2.0GB |  2GB | Meta Llama 3.2 3B — best small model  |
    | llama3.1:8b        | 4.7GB |  6GB | Meta Llama 3.1 8B — high quality      |
    | qwen2.5:0.5b       | 0.4GB |  1GB | Alibaba Qwen 2.5 0.5B — tiny          |
    | qwen2.5:3b         | 2.0GB |  2GB | Alibaba Qwen 2.5 3B — multilingual    |
    | qwen2.5:7b         | 4.5GB |  6GB | Alibaba Qwen 2.5 7B — reasoning       |
    | mistral:7b         | 4.1GB |  5GB | Mistral 7B — strong instructions      |
    | phi3.5             | 2.3GB |  3GB | Microsoft Phi 3.5 — fast reasoning    |
    | gemma2:2b          | 1.6GB |  2GB | Google Gemma 2 2B — compact           |
    | deepseek-r1:7b     | 4.7GB |  6GB | DeepSeek R1 7B — chain-of-thought     |
    | deepseek-r1:1.5b   | 1.0GB |  1GB | DeepSeek R1 1.5B — reasoning tiny     |
    +--------------------+-------+------+---------------------------------------+

    Parameters
    ----------
    model:
        Model shorthand, e.g. ``"llama3.2"``, ``"qwen2.5:3b"``.
        Default: ``"llama3.2"`` (alias for ``"llama3.2:3b"``).
    backend:
        ``"auto"`` (default) — tries Ollama first, then llama-cpp.
        Force with ``"ollama"`` or ``"llamacpp"``.
    auto_download:
        Download the model automatically on first chat/stream call. Default ``True``.
    system_prompt:
        Default system message prepended to every request.
    context_length:
        Context window tokens (llama-cpp only; Ollama manages this itself).
    gpu_layers:
        Layers to offload to GPU (llama-cpp: ``-1`` = all, ``0`` = CPU only).
    models_dir:
        Directory to store GGUF files. Defaults to ``~/.ractogateway/models/``.
    base_url:
        Ollama server URL. Defaults to ``http://localhost:11434``.
    verbose:
        Print download progress. Default ``True``.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        backend: BackendType | str = BackendType.AUTO,
        auto_download: bool = True,
        system_prompt: str | None = None,
        context_length: int = 4096,
        gpu_layers: int = -1,
        models_dir: Path | str | None = None,
        base_url: str = _OLLAMA_DEFAULT_URL,
        verbose: bool = True,
    ) -> None:
        self._model_info = resolve_model(model)
        self._backend_type = BackendType(backend)
        self._auto_download = auto_download
        self._system_prompt = system_prompt
        self._context_length = context_length
        self._gpu_layers = gpu_layers
        self._models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self._base_url = base_url
        self._verbose = verbose
        self._backend: BaseLocalBackend = self._resolve_backend()

    # ── Backend resolution ────────────────────────────────────────────────────

    def _resolve_backend(self) -> BaseLocalBackend:
        if self._backend_type == BackendType.AUTO:
            btype, be = _detect_backend(
                models_dir=self._models_dir,
                context_length=self._context_length,
                gpu_layers=self._gpu_layers,
                base_url=self._base_url,
            )
            self._backend_type = btype
            return be
        if self._backend_type == BackendType.OLLAMA:
            return OllamaBackend(base_url=self._base_url)
        return LlamaCppBackend(
            models_dir=self._models_dir,
            context_length=self._context_length,
            gpu_layers=self._gpu_layers,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_ready(self) -> None:
        """Download the model automatically if auto_download=True."""
        if not self._backend.is_model_ready(self._model_info):
            if self._auto_download:
                self.download()
            else:
                raise RuntimeError(
                    f"Model '{self._model_info.name}' is not downloaded.\n"
                    "Call kit.download() first, or set auto_download=True."
                )

    def _build_messages(
        self,
        message: str | Any,
        *,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the OpenAI-style message list from input.

        Accepts a plain string, a ChatConfig-compatible object (duck-typed),
        or a sequence of prior history turns for multi-turn conversations.
        """
        # Duck-type ChatConfig
        if hasattr(message, "user_message"):
            user_text = str(message.user_message)
            sys_text = (
                getattr(message, "system_message", None)
                or system
                or self._system_prompt
            )
        else:
            user_text = str(message)
            sys_text = system or self._system_prompt

        messages: list[dict[str, str]] = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        return messages

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        message: str | Any,
        *,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LocalResponse:
        """Single-turn (or multi-turn with ``history``) chat.

        Parameters
        ----------
        message:
            User message string or a ``ChatConfig``-compatible object.
        system:
            Override the default system prompt for this call only.
        history:
            Optional list of prior ``{"role": ..., "content": ...}`` turns.
        temperature:
            Sampling temperature (0 = deterministic, 1 = creative).
        max_tokens:
            Maximum tokens in the reply.

        Returns
        -------
        LocalResponse
            With ``.content`` containing the reply text.
        """
        self._ensure_ready()
        messages = self._build_messages(message, system=system, history=history)
        content = self._backend.chat(
            self._model_info,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return LocalResponse(
            content=content,
            model=self._model_info.name,
            backend=self._backend_type,
        )

    async def achat(
        self,
        message: str | Any,
        *,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> LocalResponse:
        """Async variant of :meth:`chat`."""
        return await asyncio.to_thread(
            self.chat,
            message,
            system=system,
            history=history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream(
        self,
        message: str | Any,
        *,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream reply tokens one at a time.

        Usage::

            for token in kit.stream("Tell me about Diwali"):
                print(token, end="", flush=True)
        """
        self._ensure_ready()
        messages = self._build_messages(message, system=system, history=history)
        yield from self._backend.stream(
            self._model_info,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def astream(
        self,
        message: str | Any,
        *,
        system: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Async streaming generator.

        Usage::

            async for token in kit.astream("Summarize this thread"):
                print(token, end="", flush=True)
        """
        loop = asyncio.get_running_loop()
        token_queue: queue.Queue[str | None] = queue.Queue()

        def _run() -> None:
            try:
                for token in self.stream(
                    message,
                    system=system,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    token_queue.put(token)
            finally:
                token_queue.put(None)

        threading.Thread(target=_run, daemon=True).start()
        while True:
            token = await loop.run_in_executor(None, token_queue.get)
            if token is None:
                break
            yield token

    # ── Model management ──────────────────────────────────────────────────────

    def download(
        self,
        model: str | None = None,
        *,
        verbose: bool | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> None:
        """Download a model to this machine.

        Parameters
        ----------
        model:
            Model name to download. Defaults to the kit's current model.
        verbose:
            Print progress. Defaults to the kit's ``verbose`` setting.
        on_progress:
            Optional callback called with each progress line.
        """
        info = resolve_model(model) if model else self._model_info
        self._backend.download(
            info,
            verbose=verbose if verbose is not None else self._verbose,
            on_progress=on_progress,
        )

    def model_status(self, model: str | None = None) -> DownloadStatus:
        """Return whether a model is ready for inference.

        Parameters
        ----------
        model:
            Model name to check. Defaults to the kit's current model.
        """
        info = resolve_model(model) if model else self._model_info
        if self._backend.is_model_ready(info):
            return DownloadStatus.READY
        return DownloadStatus.NOT_DOWNLOADED

    def delete_model(self, model: str | None = None) -> bool:
        """Delete a model to free disk / Ollama space.

        Parameters
        ----------
        model:
            Model name to delete. Defaults to the kit's current model.

        Returns
        -------
        bool
            ``True`` if the model was found and deleted.
        """
        info = resolve_model(model) if model else self._model_info
        return self._backend.delete_model(info)

    def list_available_models(self) -> list[LocalModelInfo]:
        """Return all models in the registry, sorted by family and size."""
        seen: dict[int, LocalModelInfo] = {}
        for m in MODEL_REGISTRY.values():
            seen[id(m)] = m
        return sorted(seen.values(), key=lambda m: (m.family, m.size_gb))

    def list_downloaded_models(self) -> list[LocalModelInfo]:
        """Return only models that are already downloaded / pulled."""
        return [m for m in self.list_available_models() if self._backend.is_model_ready(m)]

    def print_model_table(self) -> None:
        """Print a formatted table of all available models to stdout."""
        rows = self.list_available_models()
        print(
            f"\n{'Model':<22} {'Size':>6}  {'RAM':>5}  {'Family':<10}  Description"
        )
        print("-" * 80)
        for m in rows:
            status = "✓" if self._backend.is_model_ready(m) else " "
            ram = f"{m.size_gb * 1.3:.1f}GB"
            print(
                f"{status} {m.name:<20} {m.size_gb:>5.1f}GB  {ram:>5}  "
                f"{m.family:<10}  {m.description}"
            )
        print()

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def model(self) -> str:
        """Current model shorthand (e.g. ``"llama3.2:3b"``)."""
        return self._model_info.name

    @property
    def model_info(self) -> LocalModelInfo:
        """Full metadata record for the current model."""
        return self._model_info

    @property
    def backend(self) -> str:
        """Active backend name: ``"ollama"`` or ``"llamacpp"``."""
        return self._backend_type.value

    def __repr__(self) -> str:
        return (
            f"LocalLLMKit(model={self._model_info.name!r}, "
            f"backend={self._backend_type.value!r}, "
            f"status={self.model_status().value!r})"
        )
