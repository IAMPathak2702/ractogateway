# RactoGateway — Project Memory

## Architecture
- **Src layout**: `src/ractogateway/` — package root
- **Five developer kits** (as of 0.1.4+):
  - `openai_developer_kit` → `gpt` alias, cloud GPT models
  - `google_developer_kit` → `gemini` alias, Gemini models
  - `anthropic_developer_kit` → `claude` alias, Claude models
  - `ollama_developer_kit` → `local` alias, local Ollama server (no API key)
  - `huggingface_developer_kit` → `hf` alias, HF Inference API + TGI/vLLM
- **Low-level adapters**: `adapters/{openai,google,anthropic,ollama,huggingface}_kit.py`
  all extend `adapters/base.py:BaseLLMAdapter`
- **Lazy imports**: `__init__.py` uses `_MODULE_EXPORTS` dict + `__getattr__` — no heavy
  deps pulled on `import ractogateway`

## Key Patterns
- Every kit: `Chat` alias = full class. Methods: `chat/achat/stream/astream/embed/aembed`
- `ChatConfig` is the single input model; `LLMResponse` is the single output model
- Streaming uses `StreamChunk` with `.delta.text`, `.accumulated_text`, `.is_final`, `.usage`
- All adapters call `_wrap_provider_error(exc, provider)` to unify exceptions
- Middleware chain: truncate → exact cache → semantic cache → route → API → write caches → telemetry
- `_require_xxx()` guard functions for lazy optional import with friendly error messages
- `# type: ignore[import-untyped]` on ollama/huggingface_hub imports (no stubs)

## Optional Deps
- `pip install ractogateway[ollama]` → `ollama>=0.3,<1.0`
- `pip install ractogateway[huggingface]` → `huggingface_hub>=0.23,<2.0`
- Both included in `ractogateway[all]` and `ractogateway[dev]`
- Mypy overrides: `ollama.*` and `huggingface_hub.*` → `ignore_missing_imports = true`
- Ruff per-file ignores: `PLR0911`, `PLR0912` on both new kit files

## Ollama Kit Specifics
- `Client(host=base_url)` / `AsyncClient(host=base_url)` — default `http://localhost:11434`
- `client.chat(**request)` where request has `model`, `messages`, `options`, `tools`, `stream`
- `options = {"temperature": ..., "num_predict": max_tokens}`
- Streaming: each event has `event.message.content` (delta), `event.done` (bool),
  `event.prompt_eval_count`, `event.eval_count` (usage, final only)
- Embeddings: `client.embed(model=model, input=texts)` → `.embeddings: list[list[float]]`
- Default embedding model: `nomic-embed-text`

## HuggingFace Kit Specifics
- `InferenceClient(token=..., base_url=...)` / `AsyncInferenceClient(...)`
- Token resolution: `api_key` → `HF_TOKEN` env → `HUGGINGFACE_TOKEN` env
- Chat: `client.chat_completion(**request)` — OpenAI-compatible response shape
- Streaming: `stream=True` yields OpenAI-compatible SSE events (choices[0].delta.content)
- Embeddings: `client.feature_extraction(texts, model=model)` → `list[float]` or `list[list[float]]`
- Default chat model: `meta-llama/Llama-3.2-3B-Instruct`
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Local TGI: pass `base_url="http://localhost:8080"`, model label `"tgi"`

## Docs
- Sphinx, MyST markdown, sphinx-rtd-theme
- `docs/conf.py` — `autodoc_mock_imports` must include all optional deps
- `docs/guide/ollama.md` and `docs/guide/huggingface.md` — added in 0.1.4+
- `docs/index.md` toctree includes both new guide pages

## Files Changed in Ollama+HF PR
- NEW: `src/ractogateway/adapters/ollama_kit.py`
- NEW: `src/ractogateway/adapters/huggingface_kit.py`
- NEW: `src/ractogateway/ollama_developer_kit/__init__.py`
- NEW: `src/ractogateway/ollama_developer_kit/kit.py`
- NEW: `src/ractogateway/huggingface_developer_kit/__init__.py`
- NEW: `src/ractogateway/huggingface_developer_kit/kit.py`
- NEW: `docs/guide/ollama.md`
- NEW: `docs/guide/huggingface.md`
- UPDATED: `src/ractogateway/__init__.py` (module + attr exports, __all__)
- UPDATED: `src/ractogateway/adapters/__init__.py` (comments)
- UPDATED: `pyproject.toml` (deps, mypy overrides, ruff ignores, keywords)
- UPDATED: `README.md` (install, ToC, developer kits section, method table)
- UPDATED: `docs/index.md` (description, toctree)
- UPDATED: `docs/conf.py` (autodoc_mock_imports)
- UPDATED: `docs/guide/developer_kits.md` (kit table, cross-links)
