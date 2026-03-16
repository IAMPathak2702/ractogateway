"""Model registry for LocalLLMKit — curated list of laptop-friendly models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LocalModelInfo:
    """Metadata for a single local model."""

    name: str               # canonical shorthand, e.g. "llama3.2:3b"
    display_name: str       # human-readable label
    description: str        # one-line description
    size_gb: float          # approximate GGUF download size in GB
    # Ollama
    ollama_tag: str         # tag passed to `ollama pull / run`
    # llama-cpp / HuggingFace
    hf_repo: str            # HuggingFace repo ID
    hf_filename: str        # exact GGUF filename inside the repo
    # Metadata
    family: str             # llama | qwen | mistral | phi | gemma | deepseek
    context_length: int = 4096
    supports_thinking: bool = False
    aliases: tuple[str, ...] = field(default_factory=tuple)


# ── Master registry ───────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, LocalModelInfo] = {}


def _r(info: LocalModelInfo) -> None:
    """Register a model under its canonical name and all aliases."""
    MODEL_REGISTRY[info.name] = info
    for alias in info.aliases:
        MODEL_REGISTRY[alias] = info


# ── Llama 3.x ─────────────────────────────────────────────────────────────────

_r(LocalModelInfo(
    name="llama3.2:1b",
    display_name="Llama 3.2 1B Instruct",
    description="Meta's smallest Llama 3.2 — fits in under 1 GB RAM.",
    size_gb=0.8,
    ollama_tag="llama3.2:1b",
    hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
    hf_filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    family="llama",
    context_length=131072,
    aliases=("llama3.2-1b", "llama-3.2-1b"),
))

_r(LocalModelInfo(
    name="llama3.2:3b",
    display_name="Llama 3.2 3B Instruct",
    description="Meta's Llama 3.2 3B — best quality-per-GB under 2 GB.",
    size_gb=2.0,
    ollama_tag="llama3.2:3b",
    hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
    hf_filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    family="llama",
    context_length=131072,
    aliases=("llama3.2", "llama3.2:latest", "llama-3.2-3b"),
))

_r(LocalModelInfo(
    name="llama3.1:8b",
    display_name="Llama 3.1 8B Instruct",
    description="Meta's Llama 3.1 8B — high quality, needs ~5 GB RAM.",
    size_gb=4.7,
    ollama_tag="llama3.1:8b",
    hf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    hf_filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    family="llama",
    context_length=131072,
    aliases=("llama3.1", "llama3.1:latest", "llama-3.1-8b"),
))

# ── Qwen 2.5 ─────────────────────────────────────────────────────────────────

_r(LocalModelInfo(
    name="qwen2.5:0.5b",
    display_name="Qwen 2.5 0.5B Instruct",
    description="Alibaba's ultra-light Qwen model — under 400 MB.",
    size_gb=0.4,
    ollama_tag="qwen2.5:0.5b",
    hf_repo="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    hf_filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    family="qwen",
    context_length=32768,
    aliases=("qwen2.5-0.5b",),
))

_r(LocalModelInfo(
    name="qwen2.5:1.5b",
    display_name="Qwen 2.5 1.5B Instruct",
    description="Alibaba's Qwen 2.5 1.5B — fast, ~1 GB.",
    size_gb=1.0,
    ollama_tag="qwen2.5:1.5b",
    hf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    hf_filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    family="qwen",
    context_length=32768,
    aliases=("qwen2.5-1.5b",),
))

_r(LocalModelInfo(
    name="qwen2.5:3b",
    display_name="Qwen 2.5 3B Instruct",
    description="Alibaba's Qwen 2.5 3B — excellent multilingual support.",
    size_gb=2.0,
    ollama_tag="qwen2.5:3b",
    hf_repo="Qwen/Qwen2.5-3B-Instruct-GGUF",
    hf_filename="qwen2.5-3b-instruct-q4_k_m.gguf",
    family="qwen",
    context_length=32768,
    aliases=("qwen2.5", "qwen2.5:latest", "qwen2.5-3b"),
))

_r(LocalModelInfo(
    name="qwen2.5:7b",
    display_name="Qwen 2.5 7B Instruct",
    description="Alibaba's Qwen 2.5 7B — excellent at code and reasoning, ~4.5 GB.",
    size_gb=4.5,
    ollama_tag="qwen2.5:7b",
    hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
    hf_filename="qwen2.5-7b-instruct-q4_k_m.gguf",
    family="qwen",
    context_length=128000,
    aliases=("qwen2.5-7b",),
))

# ── Mistral ───────────────────────────────────────────────────────────────────

_r(LocalModelInfo(
    name="mistral:7b",
    display_name="Mistral 7B Instruct v0.3",
    description="Mistral AI's flagship 7B — strong instruction following.",
    size_gb=4.1,
    ollama_tag="mistral:7b",
    hf_repo="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    hf_filename="Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
    family="mistral",
    context_length=32768,
    aliases=("mistral", "mistral:latest", "mistral-7b"),
))

# ── Microsoft Phi ─────────────────────────────────────────────────────────────

_r(LocalModelInfo(
    name="phi3.5",
    display_name="Phi-3.5-mini Instruct (3.8B)",
    description="Microsoft's 3.8B model — punches above its weight in reasoning.",
    size_gb=2.3,
    ollama_tag="phi3.5",
    hf_repo="bartowski/Phi-3.5-mini-instruct-GGUF",
    hf_filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
    family="phi",
    context_length=128000,
    aliases=("phi3.5:latest", "phi-3.5", "phi3.5-mini"),
))

# ── Google Gemma ──────────────────────────────────────────────────────────────

_r(LocalModelInfo(
    name="gemma2:2b",
    display_name="Gemma 2 2B IT",
    description="Google's Gemma 2 2B — small but surprisingly capable.",
    size_gb=1.6,
    ollama_tag="gemma2:2b",
    hf_repo="bartowski/gemma-2-2b-it-GGUF",
    hf_filename="gemma-2-2b-it-Q4_K_M.gguf",
    family="gemma",
    context_length=8192,
    aliases=("gemma2", "gemma2:latest", "gemma-2-2b"),
))

_r(LocalModelInfo(
    name="gemma2:9b",
    display_name="Gemma 2 9B IT",
    description="Google's Gemma 2 9B — high quality, needs ~5.5 GB RAM.",
    size_gb=5.4,
    ollama_tag="gemma2:9b",
    hf_repo="bartowski/gemma-2-9b-it-GGUF",
    hf_filename="gemma-2-9b-it-Q4_K_M.gguf",
    family="gemma",
    context_length=8192,
    aliases=("gemma2-9b",),
))

# ── DeepSeek ──────────────────────────────────────────────────────────────────

_r(LocalModelInfo(
    name="deepseek-r1:7b",
    display_name="DeepSeek-R1-Distill Qwen 7B",
    description="DeepSeek R1 distilled 7B — chain-of-thought reasoning built-in.",
    size_gb=4.7,
    ollama_tag="deepseek-r1:7b",
    hf_repo="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
    hf_filename="DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    family="deepseek",
    supports_thinking=True,
    context_length=131072,
    aliases=("deepseek-r1", "deepseek-r1:latest"),
))

_r(LocalModelInfo(
    name="deepseek-r1:1.5b",
    display_name="DeepSeek-R1-Distill Qwen 1.5B",
    description="DeepSeek R1 distilled 1.5B — reasoning in under 1 GB.",
    size_gb=1.0,
    ollama_tag="deepseek-r1:1.5b",
    hf_repo="bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    hf_filename="DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
    family="deepseek",
    supports_thinking=True,
    context_length=131072,
    aliases=("deepseek-r1-1.5b",),
))


# ── Resolver ──────────────────────────────────────────────────────────────────

def resolve_model(name: str) -> LocalModelInfo:
    """Resolve a model shorthand or alias to its ``LocalModelInfo`` record.

    Raises ``ValueError`` with a helpful list of available models if not found.
    """
    info = MODEL_REGISTRY.get(name)
    if info is not None:
        return info
    available = sorted({m.name for m in MODEL_REGISTRY.values()})
    raise ValueError(
        f"Unknown local model {name!r}.\n"
        f"Available models: {available}\n"
        "Tip: use LocalLLMKit.list_available_models() for descriptions."
    )
