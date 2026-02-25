# Developer Kits

Developer kits provide a high-level, provider-specific interface with sync and
async parity across all five supported backends.

| Kit | Import alias | Backend | API key? |
| --- | --- | --- | --- |
| `OpenAIDeveloperKit` | `gpt` | OpenAI GPT models | Yes — `OPENAI_API_KEY` |
| `GoogleDeveloperKit` | `gemini` | Google Gemini models | Yes — `GEMINI_API_KEY` |
| `AnthropicDeveloperKit` | `claude` | Anthropic Claude models | Yes — `ANTHROPIC_API_KEY` |
| `OllamaDeveloperKit` | `local` | Local Ollama server | **No** |
| `HuggingFaceDeveloperKit` | `hf` | HF Inference API / TGI / vLLM | Optional — `HF_TOKEN` |

Every kit exposes the same six methods:

| Method | Returns |
| --- | --- |
| `chat(config)` | `LLMResponse` |
| `achat(config)` | `LLMResponse` (async) |
| `stream(config)` | `Iterator[StreamChunk]` |
| `astream(config)` | `AsyncIterator[StreamChunk]` |
| `embed(config)` | `EmbeddingResponse` |
| `aembed(config)` | `EmbeddingResponse` (async) |

> Anthropic does not offer a native embedding API.
> Ollama embeddings require a separate embedding model (e.g. `nomic-embed-text`).

## Local Model Kits

See the dedicated guides for setup and usage:

- {doc}`ollama` — zero-config local inference with Ollama
- {doc}`huggingface` — HuggingFace cloud + TGI / vLLM local servers

## Detailed API

Use these pages for full class/method signatures and docstrings:

- {doc}`../api/openai_kit`
- {doc}`../api/google_kit`
- {doc}`../api/anthropic_kit`
- {doc}`../api/modules/index` (all modules, auto-generated)
