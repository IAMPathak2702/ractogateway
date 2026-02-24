# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.4] — 2026-02-24

### Fixed

- **mypy strict** — `history_turns` in all three developer kits (`OpenAIDeveloperKit`,
  `GoogleDeveloperKit`, `AnthropicDeveloperKit`) now correctly typed as
  `list[ChatTurn] | None` instead of `list[dict[str, str]] | None`, eliminating
  18 `[arg-type]` errors. `m.role.value` is used in place of `m.role` so the
  string literal matches `ChatTurn.role: Literal["system","user","assistant"]`.
- Removed stale `# type: ignore[return-value]` comment from `adapters/base.py`
  that mypy flagged as `[unused-ignore]`.
- Removed redundant `# noqa: PLC0415` from `adapters/google_kit.py` (rule
  already globally suppressed in `pyproject.toml`).

### Changed

- Version bumped from `0.1.3` to `0.1.4` in `pyproject.toml`, `__init__.py`,
  `docs/conf.py`, and `README.md` badge.
- User guide (`docs/guide/userguide.md`) encoding artifacts corrected:
  em-dashes (`—`) and degree symbol (`°`) in the Tool Calling section restored
  from corrupted `?` characters; output code block re-labelled `text` instead
  of `json` to fix Sphinx highlighting warning.

---

## [0.1.3] — 2026-02-22

### Added

- **OpenAI Structured Outputs support** (`ractogateway.adapters._openai_schema`):
  - `sanitize_for_openai(schema)` — recursively strips all keywords rejected by the
    OpenAI `json_schema` response format (`default`, `title`, `minimum`, `maximum`,
    `minLength`, `maxLength`, `pattern`, `format`, `minItems`, `maxItems`, etc.) and
    enforces strict-mode invariants (`additionalProperties: false`, all properties
    listed in `required`) throughout the schema tree including `$defs`.
  - `validate_schema_for_openai(schema, model_name)` — early sanity check that raises
    a descriptive `ValueError` for constructs that cannot be auto-fixed (`not`,
    `if/then/else`, `allOf`, unsupported primitive types) *before* any API call is made.
  - `build_response_format(model)` — validates, sanitises, and returns the
    `{"type": "json_schema", "json_schema": {"name": ..., "schema": ..., "strict": True}}`
    dict ready to pass as `response_format` to OpenAI Chat Completions.
- **41 unit tests** in `tests/test_openai_schema.py` covering keyword stripping,
  strict-mode enforcement, `Optional` field handling, nested models, early validation
  errors, and `build_response_format` output.

### Changed

- `OpenAILLMKit._build_request` now automatically sets `response_format` via
  `build_response_format` whenever `prompt.output_format` is a Pydantic `BaseModel`
  subclass. Users can override by passing `response_format=...` in `kwargs`.
- `_schema_from_model` in `ractogateway.prompts.engine` now strips the same
  Pydantic-generated noise keywords from prompt-embedded schemas (previously only
  `title` was removed).
- Version bumped from `0.1.2` to `0.1.3` in `pyproject.toml`, `__init__.py`, and
  `docs/conf.py` (docs was incorrectly pinned to `0.1.1`).

---

## [0.1.1] — 2026-02-19

### Added

- **RAG pipeline** — full `ractogateway.rag` subsystem:
  - File readers: `TextReader`, `PdfReader`, `WordReader`, `SpreadsheetReader`, `ImageReader`, `HtmlReader`, `FileReaderRegistry`
  - Chunkers: `FixedChunker`, `RecursiveChunker`, `SentenceChunker`, `SemanticChunker`
  - Text processors: `TextCleaner`, `Lemmatizer`, `ProcessingPipeline`
  - Embedders: `OpenAIEmbedder`, `GoogleEmbedder`, `VoyageEmbedder`
  - Vector stores: `InMemoryVectorStore`, `ChromaStore`, `FAISSStore`, `PineconeStore`, `QdrantStore`, `WeaviateStore`, `MilvusStore`, `PGVectorStore`
  - `RactoRAG` main pipeline class
  - RAG install extras: `[rag]`, `[rag-all]`, `[rag-pdf]`, `[rag-word]`, `[rag-excel]`, `[rag-image]`, `[rag-nlp]`, `[rag-chroma]`, `[rag-faiss]`, `[rag-pinecone]`, `[rag-qdrant]`, `[rag-weaviate]`, `[rag-milvus]`, `[rag-pgvector]`, `[rag-voyage]`
- **Fine-tuning pipeline** — `ractogateway.finetune` subsystem:
  - `RactoDataset` with validation and JSONL export
  - `OpenAIFineTuner` — dataset upload and job management
  - `GeminiFineTuner` — single-turn text-pair fine-tuning
  - `AnthropicFineTuner` stub
- **`RactoFile`** — file attachment support in `RactoPrompt` for multimodal inputs
- **Short aliases** for `Chat` classes in each developer kit (`__init__.py` re-exports)
- **CI/CD workflows** — `python-app.yml` (tests) and `python-publish.yml` (PyPI publish)

### Changed

- Version bumped from `0.1.0` to `0.1.1` in `pyproject.toml` and `src/ractogateway/__init__.py`
- Developer kit `__init__.py` files updated with usage examples and cleaner public API surface
- Refactored top-level imports across all kits for consistency
- Updated linting rules in `pyproject.toml` (`PLC0415`, `PLR0913` ignores formalised)

---

## [0.1.0] — 2026-02-18

### Added

- Initial release of **RactoGateway** — a unified, production-ready AI SDK
- **RACTO Prompt Engine** (`ractogateway.prompts.engine`):
  - `RactoPrompt` — structured prompt with role, task, context, constraints, and `output_format`
  - Anti-hallucination guardrails built into every prompt
- **Developer Kits** with sync and async parity:
  - `OpenAIDeveloperKit` — chat, streaming, embeddings, tool calling
  - `GoogleDeveloperKit` — chat, streaming, embeddings, tool calling
  - `AnthropicDeveloperKit` — chat, streaming, tool calling
- **Adapter layer** (`ractogateway.adapters`):
  - `BaseLLMAdapter` abstract base class
  - `OpenAILLMKit`, `GoogleLLMKit`, `AnthropicLLMKit` provider adapters
- **Tool Registry** (`ractogateway.tools.registry`):
  - `@tool` decorator for registering callable tools
  - `ToolRegistry` for managing and passing tools to LLMs
- **Gateway runner** (`ractogateway.gateway`) — low-level provider-agnostic execution layer
- **Pydantic models** (`ractogateway._models`): `ChatResponse`, `EmbeddingResponse`, `StreamChunk`
- `py.typed` marker — full PEP 561 type information
- Install extras: `[openai]`, `[google]`, `[anthropic]`, `[all]`
- Apache 2.0 license

---

[0.1.4]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.1...v0.1.3
[0.1.1]: https://github.com/IAMPathak2702/RactoGateway/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/IAMPathak2702/RactoGateway/releases/tag/v0.1.0
