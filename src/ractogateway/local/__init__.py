"""RactoGateway Local LLM — privacy-first inference on your laptop.

All models run 100% locally — no API keys, no data leaves your machine.

Quick start::

    from ractogateway.local import LocalLLMKit

    # Auto-detects Ollama or llama-cpp — downloads the model on first use
    kit = LocalLLMKit("llama3.2")          # Llama 3.2 3B (default)
    response = kit.chat("Summarise this contract for me.")
    print(response.content)

    # Stream token-by-token
    for token in kit.stream("Explain GST in simple terms"):
        print(token, end="", flush=True)

    # Async
    response = await kit.achat("What is the penalty for late GST filing?")

Supported backends:

* **Ollama** (recommended) — install from https://ollama.com and run
  ``ollama serve``.  :class:`LocalLLMKit` auto-starts it and downloads
  the model with ``ollama pull``.

* **llama-cpp-python** — pure-Python GGUF inference, no server needed.
  Install with ``pip install ractogateway[local-llamacpp]``.
  Models are downloaded from HuggingFace Hub into ``~/.ractogateway/models/``.

Integration with RactoMailKit::

    from ractogateway.local import LocalLLMKit
    from ractogateway.mail import RactoMailKit

    kit = LocalLLMKit("qwen2.5:3b")
    mail = RactoMailKit(connectors=[...], kit=kit)
    mail.sync()

    # LLM-enhanced answer generation
    response = mail.ask("Which invoices are still outstanding?")
    print(response.answer)

Available models::

    llama3.2:1b    — Meta Llama 3.2 1B   (0.8 GB)
    llama3.2:3b    — Meta Llama 3.2 3B   (2.0 GB)  ← default
    llama3.1:8b    — Meta Llama 3.1 8B   (4.7 GB)
    qwen2.5:0.5b   — Alibaba Qwen 2.5 0.5B (0.4 GB)
    qwen2.5:3b     — Alibaba Qwen 2.5 3B   (2.0 GB)
    qwen2.5:7b     — Alibaba Qwen 2.5 7B   (4.5 GB)
    mistral:7b     — Mistral 7B           (4.1 GB)
    phi3.5         — Microsoft Phi 3.5    (2.3 GB)
    gemma2:2b      — Google Gemma 2 2B    (1.6 GB)
    deepseek-r1:7b — DeepSeek R1 7B       (4.7 GB)  ← reasoning
    deepseek-r1:1.5b — DeepSeek R1 1.5B  (1.0 GB)  ← reasoning tiny

Install extras::

    pip install ractogateway[ollama]            # Ollama backend
    pip install ractogateway[local-llamacpp]   # llama-cpp-python + HF Hub
    pip install ractogateway[local-llamacpp-gpu]  # + GPU support
    pip install ractogateway[local]            # ollama + llamacpp (all)
"""

from ._backends import LlamaCppBackend, OllamaBackend
from ._registry import LocalModelInfo, MODEL_REGISTRY, resolve_model
from .kit import BackendType, DownloadStatus, LocalLLMKit, LocalResponse

__all__ = [
    "BackendType",
    "DownloadStatus",
    "LlamaCppBackend",
    "LocalLLMKit",
    "LocalModelInfo",
    "LocalResponse",
    "MODEL_REGISTRY",
    "OllamaBackend",
    "resolve_model",
]
