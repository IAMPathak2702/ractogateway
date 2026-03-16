"""Ollama and llama-cpp-python backends for LocalLLMKit."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ._registry import LocalModelInfo

_OLLAMA_DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODELS_DIR = Path.home() / ".ractogateway" / "models"


# ── Lazy import helpers ───────────────────────────────────────────────────────

def _require_ollama_sdk() -> Any:
    try:
        import ollama  # noqa: PLC0415
        return ollama
    except ImportError as exc:
        raise ImportError(
            "The 'ollama' Python package is required for OllamaBackend.\n"
            "Install it with:  pip install ractogateway[ollama]\n"
            "Also ensure Ollama app is running: https://ollama.com/download"
        ) from exc


def _require_llama_cpp() -> Any:
    try:
        from llama_cpp import Llama  # noqa: PLC0415
        return Llama
    except ImportError as exc:
        raise ImportError(
            "The 'llama-cpp-python' package is required for LlamaCppBackend.\n"
            "Install it with:  pip install ractogateway[local-llamacpp]\n"
            "GPU acceleration:  pip install ractogateway[local-llamacpp-gpu]"
        ) from exc


def _require_hf_hub() -> Any:
    try:
        import huggingface_hub  # noqa: PLC0415
        return huggingface_hub
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required for GGUF model downloads.\n"
            "Install it with:  pip install ractogateway[local-llamacpp]"
        ) from exc


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseLocalBackend(ABC):
    """Abstract interface that every local backend must implement."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used on this machine."""

    @abstractmethod
    def is_model_ready(self, model: LocalModelInfo) -> bool:
        """Return True if the model is downloaded / pulled and ready to run."""

    @abstractmethod
    def download(
        self,
        model: LocalModelInfo,
        *,
        verbose: bool = True,
        on_progress: Any | None = None,
    ) -> None:
        """Download or pull the model. May print progress to stdout."""

    @abstractmethod
    def chat(
        self,
        model: LocalModelInfo,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Run a non-streaming chat and return the full reply text."""

    @abstractmethod
    def stream(
        self,
        model: LocalModelInfo,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield reply tokens one at a time."""

    @abstractmethod
    def delete_model(self, model: LocalModelInfo) -> bool:
        """Remove the model from local storage. Return True on success."""


# ── Ollama backend ────────────────────────────────────────────────────────────

def _ollama_is_reachable(base_url: str = _OLLAMA_DEFAULT_URL) -> bool:
    try:
        with urllib.request.urlopen(  # noqa: S310
            f"{base_url}/api/tags", timeout=2
        ):
            return True
    except Exception:
        return False


class OllamaBackend(BaseLocalBackend):
    """Backend that delegates inference to a running Ollama server.

    Auto-starts Ollama if ``auto_start=True`` (default) and the binary is
    on PATH.  Downloads models with ``ollama pull``.
    """

    def __init__(
        self,
        *,
        base_url: str = _OLLAMA_DEFAULT_URL,
        auto_start: bool = True,
        ollama_bin: str = "ollama",
    ) -> None:
        self._base_url = base_url
        self._auto_start = auto_start
        self._ollama_bin = ollama_bin

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        if _ollama_is_reachable(self._base_url):
            return True
        if self._auto_start:
            return self._try_start_ollama()
        return False

    def _try_start_ollama(self) -> bool:
        """Attempt to start `ollama serve` in the background."""
        try:
            subprocess.Popen(  # noqa: S603
                [self._ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time  # noqa: PLC0415
            for _ in range(24):  # up to 6 seconds
                time.sleep(0.25)
                if _ollama_is_reachable(self._base_url):
                    return True
        except FileNotFoundError:
            pass
        return False

    def is_model_ready(self, model: LocalModelInfo) -> bool:
        try:
            with urllib.request.urlopen(  # noqa: S310
                f"{self._base_url}/api/tags", timeout=3
            ) as resp:
                data: dict[str, Any] = json.loads(resp.read())
            names = {m["name"] for m in data.get("models", [])}
            tag = model.ollama_tag
            return (
                tag in names
                or f"{tag}:latest" in names
                or any(n.startswith(tag.split(":")[0] + ":") for n in names)
            )
        except Exception:
            return False

    def download(
        self,
        model: LocalModelInfo,
        *,
        verbose: bool = True,
        on_progress: Any | None = None,
    ) -> None:
        """Pull the model via ``ollama pull``."""
        if not _ollama_is_reachable(self._base_url):
            if not self._try_start_ollama():
                raise RuntimeError(
                    "Ollama is not running.\n"
                    "Install it from https://ollama.com/download and run `ollama serve`."
                )
        if verbose:
            print(
                f"\nDownloading {model.display_name} via Ollama "
                f"(~{model.size_gb:.1f} GB)..."
            )
            print(f"  Command: ollama pull {model.ollama_tag}\n")

        env = {**os.environ, "OLLAMA_HOST": self._base_url.replace("http://", "")}
        proc = subprocess.Popen(  # noqa: S603
            [self._ollama_bin, "pull", model.ollama_tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        if proc.stdout:
            for line in proc.stdout:
                stripped = line.rstrip()
                if verbose and stripped:
                    print(f"  {stripped}")
                if on_progress:
                    on_progress(stripped)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"`ollama pull {model.ollama_tag}` exited with code {rc}."
            )
        if verbose:
            print(f"\n✓ {model.display_name} is ready.\n")

    def chat(
        self,
        model: LocalModelInfo,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        ollama = _require_ollama_sdk()
        client = ollama.Client(host=self._base_url)
        options: dict[str, Any] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        response = client.chat(
            model=model.ollama_tag,
            messages=messages,
            options=options or None,
        )
        return response["message"]["content"]  # type: ignore[index]

    def stream(
        self,
        model: LocalModelInfo,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        ollama = _require_ollama_sdk()
        client = ollama.Client(host=self._base_url)
        options: dict[str, Any] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            options["num_predict"] = kwargs["max_tokens"]
        for chunk in client.chat(
            model=model.ollama_tag,
            messages=messages,
            stream=True,
            options=options or None,
        ):
            delta: str = chunk.get("message", {}).get("content", "")  # type: ignore[union-attr]
            if delta:
                yield delta

    def delete_model(self, model: LocalModelInfo) -> bool:
        try:
            env = {**os.environ, "OLLAMA_HOST": self._base_url.replace("http://", "")}
            subprocess.run(  # noqa: S603
                [self._ollama_bin, "rm", model.ollama_tag],
                env=env,
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


# ── LlamaCpp backend ──────────────────────────────────────────────────────────

class LlamaCppBackend(BaseLocalBackend):
    """Backend that runs GGUF models directly via ``llama-cpp-python``.

    Models are downloaded from HuggingFace Hub and stored in ``models_dir``
    (defaults to ``~/.ractogateway/models/``).  No server process is needed.
    """

    def __init__(
        self,
        *,
        models_dir: Path | str | None = None,
        context_length: int = 4096,
        gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        self._models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        self._context_length = context_length
        self._gpu_layers = gpu_layers
        self._verbose = verbose
        self._models_dir.mkdir(parents=True, exist_ok=True)
        # Cache loaded Llama instances to avoid re-loading on every call
        self._loaded: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "llamacpp"

    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False

    def _model_path(self, model: LocalModelInfo) -> Path:
        return self._models_dir / model.hf_filename

    def is_model_ready(self, model: LocalModelInfo) -> bool:
        return self._model_path(model).exists()

    def download(
        self,
        model: LocalModelInfo,
        *,
        verbose: bool = True,
        on_progress: Any | None = None,
    ) -> None:
        """Download the GGUF file from HuggingFace Hub."""
        hf_hub = _require_hf_hub()
        dest = self._model_path(model)
        if dest.exists():
            if verbose:
                print(f"✓ {model.display_name} already downloaded at {dest}")
            return

        if verbose:
            print(
                f"\nDownloading {model.display_name} from HuggingFace Hub\n"
                f"  Repo:   {model.hf_repo}\n"
                f"  File:   {model.hf_filename}\n"
                f"  Size:   ~{model.size_gb:.1f} GB\n"
                f"  Saving: {dest}\n"
            )

        downloaded: str = hf_hub.hf_hub_download(
            repo_id=model.hf_repo,
            filename=model.hf_filename,
            local_dir=str(self._models_dir),
            local_dir_use_symlinks=False,
        )
        # Ensure it lands at the expected path
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != dest.resolve():
            downloaded_path.rename(dest)

        if verbose:
            print(f"\n✓ {model.display_name} ready at {dest}\n")
        if on_progress:
            on_progress(f"Downloaded {model.display_name}")

    def _load(self, model: LocalModelInfo) -> Any:
        """Return a cached Llama instance, loading from disk if necessary."""
        if model.name not in self._loaded:
            Llama = _require_llama_cpp()
            path = self._model_path(model)
            if not path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {path}\n"
                    "Call kit.download() first, or set auto_download=True."
                )
            self._loaded[model.name] = Llama(
                model_path=str(path),
                n_ctx=self._context_length,
                n_gpu_layers=self._gpu_layers,
                verbose=self._verbose,
                chat_format="chatml",
            )
        return self._loaded[model.name]

    def chat(
        self,
        model: LocalModelInfo,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        llm = self._load(model)
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response["choices"][0]["message"]["content"]  # type: ignore[index]

    def stream(
        self,
        model: LocalModelInfo,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Iterator[str]:
        llm = self._load(model)
        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            stream=True,
        ):
            delta: str = chunk["choices"][0]["delta"].get("content", "")  # type: ignore[index]
            if delta:
                yield delta

    def delete_model(self, model: LocalModelInfo) -> bool:
        path = self._model_path(model)
        if path.exists():
            if model.name in self._loaded:
                del self._loaded[model.name]
            path.unlink()
            return True
        return False
